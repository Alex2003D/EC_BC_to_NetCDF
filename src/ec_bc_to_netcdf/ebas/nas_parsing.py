"""Utilities for parsing EBAS NAS headers, dates, and column descriptors.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence
import os
import numpy as np

__all__ = [
    "START_MARKER_COL_DESC",
    "END_MARKER_COL_DESC",
    "DEFAULT_UNCERTAINTY_PERCENTAGE",
    "FILL_VALUE_STRINGS",
    "get_station_code_from_nas",
    "get_content_between_markers",
    "parse_nas_date",
    "parse_metadata_from_header",
    "find_data_block_start_index",
    "parse_column_description_details",
]


# Markers defining the column description block in NAS files
START_MARKER_COL_DESC = "end_time of measurement, days from the file reference point"
END_MARKER_COL_DESC = "0"

# Default uncertainty used when none is provided in the file or column descriptor
DEFAULT_UNCERTAINTY_PERCENTAGE = 18.13

# Commonly used fill strings in EBAS files
FILL_VALUE_STRINGS = [
    "9.99", "9.999",
    "99.9", "99.99", "99.999", "99.9999",
    "999", "999.9", "999.99", "999.999",
    "9999", "9999.9", "9999.99", "9999.999",
    "-999", "-9999"
]
FILL_VALUE_SET = set(FILL_VALUE_STRINGS)


def _is_numeric_string(text: str) -> bool:
    """Return True if the given string looks like a number (int/float with optional sign)."""
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def get_station_code_from_nas(raw_lines: Sequence[str], filename: str) -> Optional[str]:
    """Extract station code from the NAS header or fallback to filename pattern.

    Returns the station code in uppercase if found, otherwise None.
    """
    # Prefer header value if present
    for line_content in raw_lines[:150]:
        line = line_content.lower().strip()
        if line.startswith("station code:"):
            try:
                return line.split(":", 1)[1].strip().upper()
            except IndexError:
                continue

    # Fallback to filename pattern: AA9999A... → first 7 chars
    name_part = os.path.basename(filename).split(".")[0]
    if len(name_part) >= 7:
        candidate = name_part[:7]
        if candidate[:2].isalpha() and candidate[2:6].isdigit() and candidate[6].isalpha():
            return candidate.upper()
    return None


def get_content_between_markers(lines: Sequence[str], start_marker: str, end_marker_text: str) -> Optional[List[str]]:
    """Return lines between start and end markers, trimmed; None if not found."""
    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if start_marker in line and start_idx == -1:
            start_idx = i
            continue
        if start_idx != -1 and line.strip() == end_marker_text:
            end_idx = i
            break

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return [l.strip() for l in lines[start_idx + 1:end_idx]]
    return None


def parse_nas_date(
    date_str: str,
    reference_date_obj: Optional[datetime] = None,
    initial_day_reading_for_offset: Optional[float] = None
) -> Optional[datetime]:
    """Parse EBAS NAS date strings..
    """
    try:
        return datetime.strptime(date_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        if not reference_date_obj:
            return None
        try:
            current_day_reading = float(date_str)
            if initial_day_reading_for_offset is not None:
                day_offset_from_series_start = current_day_reading - float(initial_day_reading_for_offset)
                return reference_date_obj + timedelta(days=day_offset_from_series_start)
            ref_date_midnight = reference_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
            return ref_date_midnight + timedelta(days=current_day_reading)
        except ValueError:
            return None


def parse_metadata_from_header(lines: Sequence[str], filename: str) -> Dict[str, Any]:
    """Parse basic metadata from NAS header lines.

    Extracts station location, start date, detection limits.
    """
    metadata: Dict[str, Any] = {
        "start_date_obj": None,
        "latitude": np.nan,
        "longitude": np.nan,
        "altitude": np.nan,
        "file_detection_limit": np.nan,
        "station_code_from_header": None,
        "dependent_var_fill_values": []
    }

    # Identify the column description start line
    start_marker_col_desc_line_idx = -1
    for i, line_content in enumerate(lines):
        if START_MARKER_COL_DESC in line_content:
            start_marker_col_desc_line_idx = i
            break

    # Attempt to read a line of dependent fill values immediately above the marker
    if start_marker_col_desc_line_idx > 0:
        potential_fill_line_idx = start_marker_col_desc_line_idx - 1
        if potential_fill_line_idx >= 0:
            potential_fill_line_content = lines[potential_fill_line_idx].strip()
            parts = potential_fill_line_content.split()
            if len(parts) > 1:
                is_likely_fills = True
                for part_val in parts:
                    if not (_is_numeric_string(part_val) or all(c in '9.-' for c in part_val) or part_val in FILL_VALUE_SET):
                        if any(c.isalpha() for c in part_val):
                            is_likely_fills = False
                            break
                if is_likely_fills:
                    metadata["dependent_var_fill_values"] = parts

    # Parse core metadata from header (first ~100 lines)
    for line_num, line_content in enumerate(lines):
        line = line_content.lower().strip()
        if line.startswith("startdate:"):
            try:
                metadata["start_date_obj"] = parse_nas_date(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("station latitude:"):
            try:
                metadata["latitude"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("station longitude:"):
            try:
                metadata["longitude"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("station altitude:"):
            try:
                alt_str = line.split(":")[1].strip().split(" ")[0]
                metadata["altitude"] = float(alt_str)
            except (ValueError, IndexError):
                pass
        elif line.startswith("station code:"):
            try:
                metadata["station_code_from_header"] = line.split(":")[1].strip().upper()
            except IndexError:
                pass
        elif "detection limit:" in line and "detection limit expl:" not in line:
            try:
                val_str = line.split("detection limit:")[1].strip().split(" ")[0]
                metadata["file_detection_limit"] = float(val_str)
            except (IndexError, ValueError):
                pass

        if line_num > 100:
            break
    return metadata


def find_data_block_start_index(lines: Sequence[str], col_desc_block_end_line_idx: int) -> int:
    """Return index where numeric data block starts, or -1 if not found."""
    try:
        # Fast-path: use the declared number of key-value pairs (if present)
        after_desc_idx = col_desc_block_end_line_idx + 1
        if after_desc_idx < len(lines):
            try:
                num_kv_lines = int(lines[after_desc_idx].strip())
                potential_header_idx = after_desc_idx + num_kv_lines + 1
                if potential_header_idx < len(lines):
                    header_line = lines[potential_header_idx].strip()
                    lower_header = header_line.lower()
                    if "starttime" in lower_header and ("endtime" in lower_header or "ec" in lower_header):
                        return potential_header_idx + 1
                    parts = header_line.split()
                    if len(parts) >= 2 and all(_is_numeric_string(p) for p in parts[:2]):
                        return potential_header_idx
            except ValueError:
                pass

        # Fallback: scan forward with heuristics
        for i in range(after_desc_idx, len(lines)):
            candidate = lines[i].strip()
            if not candidate or candidate.startswith("#"):
                continue
            lower_candidate = candidate.lower()
            if "starttime" in lower_candidate and ("endtime" in lower_candidate or "ec" in lower_candidate):
                if ":" not in candidate:
                    return i + 1
            parts = candidate.split()
            if len(parts) >= 2 and all(_is_numeric_string(p) for p in parts[:2]):
                if i > 0 and (":" not in lines[i - 1] or "endtime" in lines[i - 1].lower()):
                    return i
            if i > col_desc_block_end_line_idx + 200 and col_desc_block_end_line_idx != -1:
                break
    except Exception:
        pass
    return -1


def parse_column_description_details(col_desc_content_lines: Sequence[str]) -> Dict[str, Any]:
    """Identify EC/uncertainty/flag column indices and QA metadata from descriptors."""
    details: Dict[str, Any] = {
        "ec_col_idx": -1, "unc_col_idx": -1, "flag_col_idx": -1,
        "col_desc_det_limit": np.nan, "col_desc_qa_variability_percent": np.nan,
        "num_data_columns": len(col_desc_content_lines)
    }

    def _try_extract_float(line_lower: str, prefix: str, terminators: Sequence[str]) -> Optional[float]:
        try:
            tail = line_lower.split(prefix, 1)[1]
            for term in terminators:
                if term in tail:
                    tail = tail.split(term, 1)[0]
            return float(tail.strip())
        except (IndexError, ValueError):
            return None

    for idx, line_content in enumerate(col_desc_content_lines):
        lower_line = line_content.lower()
        if "elemental_carbon" in lower_line and "uncertainty" not in lower_line and "numflag" not in lower_line:
            if details["ec_col_idx"] == -1:
                details["ec_col_idx"] = idx
            if details["ec_col_idx"] == idx:
                det = _try_extract_float(lower_line, "detection limit=", terminators=[",", "ug"])
                if det is not None:
                    details["col_desc_det_limit"] = det
                qa = _try_extract_float(lower_line, "qa1 variability=", terminators=["%", ","])
                if qa is not None:
                    details["col_desc_qa_variability_percent"] = qa
        elif "elemental_carbon" in lower_line and "uncertainty" in lower_line:
            if details["unc_col_idx"] == -1:
                details["unc_col_idx"] = idx
        elif "numflag" in lower_line or "flag" in lower_line:
            if details["flag_col_idx"] == -1:
                details["flag_col_idx"] = idx

    if (
        details["ec_col_idx"] == -1 and
        details["num_data_columns"] == 1 and
        details["unc_col_idx"] == -1 and
        details["flag_col_idx"] == -1
    ):
        details["ec_col_idx"] = 0
    elif details["ec_col_idx"] == -1 and details["num_data_columns"] > 0:
        excluded = {details["unc_col_idx"], details["flag_col_idx"]}
        possible_ec_indices = [i for i in range(details["num_data_columns"]) if i not in excluded]
        if len(possible_ec_indices) == 1:
            details["ec_col_idx"] = possible_ec_indices[0]
        elif not possible_ec_indices and "elemental_carbon" in col_desc_content_lines[0].lower():
            details["ec_col_idx"] = 0
    return details


