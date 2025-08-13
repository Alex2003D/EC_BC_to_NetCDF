import os
from datetime import datetime, timezone
import argparse
import numpy as np
from netCDF4 import Dataset
from collections import defaultdict
from .nas_parsing import (
    START_MARKER_COL_DESC, END_MARKER_COL_DESC, DEFAULT_UNCERTAINTY_PERCENTAGE, FILL_VALUE_STRINGS,
    get_station_code_from_nas, get_content_between_markers, parse_nas_date,
    parse_metadata_from_header, find_data_block_start_index, parse_column_description_details
)
from .shared import build_id_array_from_unix_seconds, list_nas_files


MIN_DATE_CUTOFF = datetime(2013, 1, 1, tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EBAS EC NAS files to NetCDF, grouped by station.")
    parser.add_argument("source", help="Folder containing .nas files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for .nc files")
    args = parser.parse_args()

    source_dir = args.source
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    station_file_groups = {}
    processed_files_count = 0
    skipped_files_count = 0

    source_dir_abs = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir_abs):
        print(f"Error: Source directory {source_dir_abs} not found.")
        print(f"Please ensure the EBAS NAS files are in the correct location or pass a valid folder path.")
        return

    all_nas_files = list_nas_files(source_dir_abs, sort=False, case_insensitive_ext=True)
    if not all_nas_files:
        print(f"Error: Could not list files in {source_dir_abs}. Check path and permissions.")
        return

    print(f"Found {len(all_nas_files)} .nas files in {source_dir_abs}")

    for nas_file_path in all_nas_files:
        filename = os.path.basename(nas_file_path)
        try:
            with open(nas_file_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_lines = f.readlines()
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            skipped_files_count += 1
            continue

        header_metadata = parse_metadata_from_header(raw_lines, filename)

        station_code = header_metadata.get("station_code_from_header")
        if not station_code:
            station_code = get_station_code_from_nas(raw_lines, filename)

        if not station_code:
            print(f"  DEBUG SKIP ({filename}): Could not determine station code. Skipping.")
            skipped_files_count += 1
            continue

        col_desc_content_lines = get_content_between_markers(raw_lines, START_MARKER_COL_DESC, END_MARKER_COL_DESC)
        if not col_desc_content_lines:
            print(f"  DEBUG SKIP ({filename}): Could not find column description block (markers: '{START_MARKER_COL_DESC}' to '{END_MARKER_COL_DESC}'). Skipping.")
            skipped_files_count += 1
            continue

        col_desc_details = parse_column_description_details(col_desc_content_lines)
        ec_col_idx = col_desc_details["ec_col_idx"]
        unc_col_idx = col_desc_details["unc_col_idx"]
        num_data_cols = col_desc_details["num_data_columns"]

        if ec_col_idx == -1:
            print(f"  DEBUG SKIP ({filename}): Could not identify an elemental_carbon data column. Descriptions found: {col_desc_content_lines[:5]}")
            skipped_files_count += 1
            continue

        col_desc_end_line_idx = -1
        for i, line in enumerate(raw_lines):
            if START_MARKER_COL_DESC in line:
                start_desc_idx = i
                for j in range(start_desc_idx + 1, len(raw_lines)):
                    if raw_lines[j].strip() == END_MARKER_COL_DESC:
                        col_desc_end_line_idx = j
                        break
                break

        if col_desc_end_line_idx == -1:
            print(f"  DEBUG SKIP ({filename}): Could not find end marker ('{END_MARKER_COL_DESC}') for column description (after finding start). Skipping.")
            skipped_files_count += 1
            continue

        data_block_start_idx = find_data_block_start_index(raw_lines, col_desc_end_line_idx)

        if data_block_start_idx == -1:
            print(f"  DEBUG SKIP ({filename}): Could not find data block start after col_desc_end_line_idx: {col_desc_end_line_idx}. Skipping.")
            skipped_files_count += 1
            continue

        first_data_line_actual_time_value = None
        if data_block_start_idx < len(raw_lines):
            search_idx = data_block_start_idx
            while search_idx < len(raw_lines):
                line_content = raw_lines[search_idx].strip()
                if not line_content or line_content.startswith("#"):
                    search_idx += 1
                    continue

                parts_first_line = line_content.split()
                if len(parts_first_line) > 0:
                    try:
                        first_data_line_actual_time_value = float(parts_first_line[0])
                    except ValueError:
                        pass
                break

        file_data = {
            "timestamps": [], "ec_values": [], "uncertainty_values": [],
            "latitude": header_metadata["latitude"],
            "longitude": header_metadata["longitude"],
            "altitude": header_metadata["altitude"],
            "source_file": filename,
            "start_date_obj": header_metadata["start_date_obj"]
        }

        for line_num in range(data_block_start_idx, len(raw_lines)):
            line = raw_lines[line_num].strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2 + num_data_cols:
                print(f"    DEBUG DATALINE SKIP ({filename} L:{line_num + 1}): Malformed data line (too few parts). Expected {2 + num_data_cols}, got {len(parts)}. Line: '{line}'")
                continue

            try:
                start_time_str = parts[0]
                dt_obj = parse_nas_date(start_time_str,
                                        header_metadata["start_date_obj"],
                                        initial_day_reading_for_offset=first_data_line_actual_time_value)

                if dt_obj is None:
                    dt_obj = parse_nas_date(start_time_str, None)
                    if dt_obj is None:
                        print(f"    DEBUG DATALINE SKIP ({filename} L:{line_num + 1}): Could not parse time '{start_time_str}' with available methods.")
                        continue

                if dt_obj < MIN_DATE_CUTOFF:
                    continue

                timestamp_sec = int(dt_obj.timestamp())

                ec_val_str = parts[2 + ec_col_idx]
                file_specific_ec_fill = None
                dependent_fills = header_metadata.get("dependent_var_fill_values")
                if dependent_fills:
                    offset = 0
                    if len(dependent_fills) == num_data_cols + 1:
                        offset = 1

                    target_idx_ec = ec_col_idx + offset
                    if target_idx_ec < len(dependent_fills):
                        file_specific_ec_fill = dependent_fills[target_idx_ec]

                if (file_specific_ec_fill and ec_val_str == file_specific_ec_fill) or \
                   ec_val_str in FILL_VALUE_STRINGS:
                    ec_val = np.nan
                else:
                    ec_val = float(ec_val_str)

                unc_val = np.nan
                if unc_col_idx != -1:
                    if (2 + unc_col_idx) < len(parts):
                        unc_val_str = parts[2 + unc_col_idx]
                        file_specific_unc_fill = None
                        if dependent_fills:
                            target_idx_unc = unc_col_idx + offset
                            if target_idx_unc < len(dependent_fills):
                                file_specific_unc_fill = dependent_fills[target_idx_unc]

                        if (file_specific_unc_fill and unc_val_str == file_specific_unc_fill) or \
                           unc_val_str in FILL_VALUE_STRINGS:
                            unc_val = np.nan
                        else:
                            unc_val = float(unc_val_str)
                elif not np.isnan(col_desc_details["col_desc_qa_variability_percent"]):
                    if not np.isnan(ec_val):
                        unc_val = ec_val * (col_desc_details["col_desc_qa_variability_percent"] / 100.0)
                elif not np.isnan(ec_val):
                    unc_val = ec_val * (DEFAULT_UNCERTAINTY_PERCENTAGE / 100.0)

                file_data["timestamps"].append(timestamp_sec)
                file_data["ec_values"].append(ec_val)
                file_data["uncertainty_values"].append(unc_val)

            except ValueError as e:
                print(f"    DEBUG DATALINE SKIP ({filename} L:{line_num + 1}): ValueError. Line: '{line}'. Error: {e}")
                continue
            except IndexError as e:
                print(f"    DEBUG DATALINE SKIP ({filename} L:{line_num + 1}): IndexError. Line: '{line}'. Error: {e}. ec_idx={2 + ec_col_idx}, unc_idx={2 + unc_col_idx if unc_col_idx != -1 else 'N/A'}, num_parts={len(parts)}")
                continue

        if file_data["timestamps"]:
            if station_code not in station_file_groups:
                station_file_groups[station_code] = []
            station_file_groups[station_code].append(file_data)
            processed_files_count += 1
        else:
            print(f"  DEBUG SKIP ({filename}): No valid data extracted (timestamps list is empty). Skipping file.")
            skipped_files_count += 1

    print(f"Finished parsing phase. Processed {processed_files_count} files, skipped {skipped_files_count} files.")
    print(f"Aggregating data for {len(station_file_groups)} stations.")

    created_nc_files = 0
    for station_code, files_data_list in station_file_groups.items():
        print(f"  Processing station: {station_code} ({len(files_data_list)} file(s))")

        files_data_list.sort(key=lambda fd: fd["start_date_obj"] if fd["start_date_obj"] else datetime.min.replace(tzinfo=timezone.utc))

        all_timestamps = []
        all_ec_values = []
        all_uncertainty_values = []

        for file_data in files_data_list:
            all_timestamps.extend(file_data["timestamps"])
            all_ec_values.extend(file_data["ec_values"])
            all_uncertainty_values.extend(file_data["uncertainty_values"])

        if not all_timestamps:
            print(f"    No data to write for station {station_code}. Skipping NetCDF creation.")
            continue

        data_by_timestamp = defaultdict(list)
        for i in range(len(all_timestamps)):
            ts = all_timestamps[i]
            ec = all_ec_values[i]
            unc = all_uncertainty_values[i]
            data_by_timestamp[ts].append({'ec': ec, 'unc': unc})

        unique_data_tuples = []
        sorted_unique_timestamps = sorted(list(data_by_timestamp.keys()))

        for ts in sorted_unique_timestamps:
            records_for_ts = data_by_timestamp[ts]

            selected_ec = np.nan
            selected_unc = np.nan
            found_valid_ec = False

            for record in records_for_ts:
                if not np.isnan(record['ec']):
                    selected_ec = record['ec']
                    selected_unc = record['unc']
                    found_valid_ec = True
                    break

            if not found_valid_ec and records_for_ts:
                selected_ec = records_for_ts[0]['ec']
                selected_unc = records_for_ts[0]['unc']

            unique_data_tuples.append((ts, selected_ec, selected_unc))

        if not unique_data_tuples:
            print(f"    No data after improved de-duplication for station {station_code}. Skipping NetCDF creation.")
            continue

        final_timestamps, final_ec_values, final_uncertainty_values = zip(*unique_data_tuples)

        ref_file_data = files_data_list[0]
        latitude = ref_file_data["latitude"]
        longitude = ref_file_data["longitude"]
        altitude = ref_file_data["altitude"]

        if np.isnan(latitude):
            for fd in files_data_list[1:]:
                if not np.isnan(fd["latitude"]):
                    latitude = fd["latitude"]
                    print(f"    Note: Used latitude from {fd['source_file']} for station {station_code}")
                    break
        if np.isnan(longitude):
            for fd in files_data_list[1:]:
                if not np.isnan(fd["longitude"]):
                    longitude = fd["longitude"]
                    print(f"    Note: Used longitude from {fd['source_file']} for station {station_code}")
                    break
        if np.isnan(altitude):
            for fd in files_data_list[1:]:
                if not np.isnan(fd["altitude"]):
                    altitude = fd["altitude"]
                    print(f"    Note: Used altitude from {fd['source_file']} for station {station_code}")
                    break

        output_dir_abs = os.path.abspath(output_dir)
        if not os.path.exists(output_dir_abs):
            os.makedirs(output_dir_abs)

        nc_file_path = os.path.join(output_dir_abs, f"{station_code}.nc")
        try:
            with Dataset(nc_file_path, 'w', format='NETCDF4') as ncfile:
                ncfile.createDimension('time_components', len(final_timestamps))

                ec_var = ncfile.createVariable('ec', 'f8', ('time_components',), fill_value=np.nan)
                ec_var[:] = np.array(final_ec_values, dtype='f8')
                ec_var.coordinates = "id"

                num_time_points = len(final_timestamps)

                lat_var = ncfile.createVariable('latitude', 'f8', ('time_components',), fill_value=np.nan)
                lat_val_to_write = latitude if not np.isnan(latitude) else np.nan
                lat_var[:] = np.full(num_time_points, lat_val_to_write, dtype='f8')
                lat_var.coordinates = "id"

                lon_var = ncfile.createVariable('longitude', 'f8', ('time_components',), fill_value=np.nan)
                lon_val_to_write = longitude if not np.isnan(longitude) else np.nan
                lon_var[:] = np.full(num_time_points, lon_val_to_write, dtype='f8')
                lon_var.coordinates = "id"

                alt_var = ncfile.createVariable('altitude', 'f8', ('time_components',), fill_value=np.nan)
                alt_val_to_write = altitude if not np.isnan(altitude) else np.nan
                alt_var[:] = np.full(num_time_points, alt_val_to_write, dtype='f8')
                alt_var.coordinates = "id"

                unc_var = ncfile.createVariable('uncertainty', 'f8', ('time_components',), fill_value=np.nan)
                unc_var[:] = np.array(final_uncertainty_values, dtype='f8')
                unc_var.coordinates = "id"

                time_var = ncfile.createVariable('time_components', 'i8', ('time_components',))
                time_var[:] = np.array(final_timestamps, dtype='i8')

                id_var_new = ncfile.createVariable('id', 'i8', ('time_components',))
                id_var_new[:] = build_id_array_from_unix_seconds(final_timestamps)

            created_nc_files += 1
        except Exception as e:
            print(f"    Error creating NetCDF for station {station_code} at {nc_file_path}: {e}")

    print(f"Finished. Created {created_nc_files} NetCDF files in {os.path.abspath(output_dir)}.")


if __name__ == "__main__":
    main()


