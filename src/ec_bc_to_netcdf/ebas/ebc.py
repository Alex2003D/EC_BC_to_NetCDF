import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
import re
import argparse
import os
from io import StringIO
from typing import Any, Dict, List, Optional
from .shared import build_id_array_from_unix_seconds, list_nas_files


def parse_variable_definition(line_content: str) -> Dict[str, Any]:
    """Parse one variable definition line from an EBAS NAS file.

    Extracts base fields plus optional wavelength and statistic metadata.
    """
    parts = [p.strip() for p in line_content.split(',')]
    definition = {
        'raw': line_content,
        'name_desc': parts[0],
        'unit': None,
        'wavelength': None,
        'statistic': None,
        'original_index': -1
    }
    if len(parts) > 1:
        definition['unit'] = parts[1]

    for part in parts[2:]:
        if 'Wavelength=' in part:
            match = re.search(r'Wavelength=([\d.]+)\s*nm', part, re.IGNORECASE)
            if match:
                definition['wavelength'] = float(match.group(1))
        elif 'Statistics=' in part:
            match = re.search(r'Statistics=([\w:.-]+)', part, re.IGNORECASE)
            if match:
                definition['statistic'] = match.group(1).lower()
    return definition


def _resolve_endtime_column_name(df: pd.DataFrame, variable_definitions: List[Dict[str, Any]], *, filepath: str) -> str:
    """Resolve the DataFrame column name that represents end time.
    """
    endtime_var_def = next((vd for vd in variable_definitions if 'end_time' in vd['name_desc'].lower()), None)
    if not endtime_var_def or 'df_column_name' not in endtime_var_def:
        found_endtime_col = None
        for col in df.columns:
            if col.lower() == 'endtime' or col.lower() == 'end_time':
                found_endtime_col = col
                break
        if not found_endtime_col:
            raise ValueError(f"Could not identify 'endtime' column in DataFrame for {filepath}")
        return found_endtime_col
    return endtime_var_def['df_column_name']


def _select_bc_variable(variable_definitions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the preferred equivalent_black_carbon variable definition.

    Preference order:
    - wavelength == 880.0 and statistic == 'arithmetic mean'
    - statistic == 'arithmetic mean'
    - first candidate if any
    """
    ebc_candidates = [vd for vd in variable_definitions if 'equivalent_black_carbon' in vd['name_desc'].lower()]
    chosen_bc_vd = None
    for vd_cand in ebc_candidates:
        if vd_cand['wavelength'] == 880.0 and vd_cand['statistic'] == 'arithmetic mean':
            chosen_bc_vd = vd_cand
            break
    if not chosen_bc_vd:
        for vd_cand in ebc_candidates:
            if vd_cand['statistic'] == 'arithmetic mean':
                chosen_bc_vd = vd_cand
                break
    if not chosen_bc_vd and ebc_candidates:
        chosen_bc_vd = ebc_candidates[0]
    return chosen_bc_vd


def parse_nas_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Parse a single EBAS eBC NAS file into structured metadata and a DataFrame.

    Returns a dict with keys 'metadata', 'variable_definitions', and 'data_df'.
    On parse errors, returns None and prints an error message.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f.readlines()]

        parsed_content = {'metadata': {}, 'variable_definitions': []}

        n_header_lines, parsed_content['metadata']['format_code'] = map(int, lines[0].split())

        ref_date_parts = lines[6].split()
        ref_year, ref_month, ref_day = int(ref_date_parts[0]), int(ref_date_parts[1]), int(ref_date_parts[2])
        reference_date = datetime(ref_year, ref_month, ref_day, tzinfo=timezone.utc)
        parsed_content['metadata']['reference_date'] = reference_date

        n_variables = int(lines[9])
        parsed_content['metadata']['n_variables'] = n_variables

        current_line_idx = 10

        scale_factors_str = []
        while len(scale_factors_str) < n_variables:
            scale_factors_str.extend(lines[current_line_idx].split())
            current_line_idx += 1
        parsed_content['metadata']['scale_factors'] = [float(sf) for sf in scale_factors_str[:n_variables]]

        missing_values_str = []
        while len(missing_values_str) < n_variables:
            missing_values_str.extend(lines[current_line_idx].split())
            current_line_idx += 1
        parsed_content['metadata']['missing_values'] = [mv for mv in missing_values_str[:n_variables]]

        variable_definitions = []
        for i in range(n_variables):
            parsed_def = parse_variable_definition(lines[current_line_idx])
            parsed_def['original_index'] = i
            variable_definitions.append(parsed_def)
            current_line_idx += 1
        parsed_content['variable_definitions'] = variable_definitions

        n_special_comments = int(lines[current_line_idx])
        current_line_idx += 1 + n_special_comments

        n_normal_comments = int(lines[current_line_idx])
        current_line_idx += 1

        for i in range(n_normal_comments):
            line_content = lines[current_line_idx + i]
            if ':' in line_content:
                key, value = line_content.split(':', 1)
                parsed_content['metadata'][key.strip()] = value.strip()

        raw_data_column_names = lines[n_header_lines - 1].split()

        data_column_names = []
        counts = {}
        for name in raw_data_column_names:
            if name in counts:
                counts[name] += 1
                data_column_names.append(f"{name}_{counts[name]}")
            else:
                counts[name] = 0
                data_column_names.append(name)

        data_lines_str = "\n".join(lines[n_header_lines:])
        data_io = StringIO(data_lines_str)

        df = pd.read_csv(data_io, sep=r"\s+", header=None, names=data_column_names, na_filter=False, dtype=str)

        col_offset = 0
        first_defined_var_name = variable_definitions[0]['name_desc'].lower()

        if 'starttime' in df.columns[0].lower() and ('endtime' in first_defined_var_name or 'end_time' in first_defined_var_name):
            if len(df.columns) > 1 and (df.columns[1].lower() == 'endtime' or df.columns[1].lower() == 'end_time'):
                col_offset = 1
        elif 'endtime' in first_defined_var_name and df.columns[0].lower() == 'endtime':
            col_offset = 0
        elif 'end_time' in first_defined_var_name and df.columns[0].lower() == 'end_time':
            col_offset = 0

        if len(df.columns) < n_variables + col_offset:
            raise ValueError(
                f"DataFrame has {len(df.columns)} columns, but expected at least {n_variables} "
                f"(for defined vars) + {col_offset} (offset). File: {filepath}"
            )

        mapped_df_cols_for_defs = list(df.columns[col_offset: col_offset + n_variables])
        if len(mapped_df_cols_for_defs) != n_variables:
            raise ValueError(
                f"Could not map {n_variables} variable definitions to DataFrame columns. "
                f"Deduced {len(mapped_df_cols_for_defs)} target columns from offset {col_offset}. File: {filepath}"
            )

        for i in range(n_variables):
            variable_definitions[i]['df_column_name'] = mapped_df_cols_for_defs[i]
            col_name_in_df = mapped_df_cols_for_defs[i]
            missing_val_str = parsed_content['metadata']['missing_values'][i]
            scale_factor = parsed_content['metadata']['scale_factors'][i]

            df[col_name_in_df] = df[col_name_in_df].replace(missing_val_str, np.nan)
            df[col_name_in_df] = pd.to_numeric(df[col_name_in_df], errors='coerce')
            df[col_name_in_df] = df[col_name_in_df] * scale_factor

        endtime_col_name_in_df = _resolve_endtime_column_name(df, variable_definitions, filepath=filepath)

        df[endtime_col_name_in_df] = pd.to_numeric(df[endtime_col_name_in_df], errors='coerce')
        df['abs_time'] = df[endtime_col_name_in_df].apply(
            lambda x: reference_date + timedelta(days=x) if pd.notnull(x) else pd.NaT
        )
        df.dropna(subset=['abs_time'], inplace=True)

        chosen_bc_vd = _select_bc_variable(variable_definitions)

        if not chosen_bc_vd:
            print(f"Warning: No suitable 'equivalent_black_carbon' column found in {filepath}. This file's data might be incomplete.")
            df['parsed_bc'] = np.nan
        else:
            df['parsed_bc'] = df[chosen_bc_vd['df_column_name']]

        p15_vd, p84_vd = None, None
        if chosen_bc_vd and chosen_bc_vd['wavelength'] is not None:
            same_wavelength_candidates = [
                vd for vd in variable_definitions
                if 'equivalent_black_carbon' in vd['name_desc'].lower() and vd['wavelength'] == chosen_bc_vd['wavelength']
            ]
            for vd_cand in same_wavelength_candidates:
                if vd_cand['statistic'] == 'percentile:15.87':
                    p15_vd = vd_cand
                elif vd_cand['statistic'] == 'percentile:84.13':
                    p84_vd = vd_cand

        df['parsed_p15'] = df[p15_vd['df_column_name']] if p15_vd else np.nan
        df['parsed_p84'] = df[p84_vd['df_column_name']] if p84_vd else np.nan

        essential_cols = ['abs_time', 'parsed_bc', 'parsed_p15', 'parsed_p84']
        df_essential = df[[col for col in essential_cols if col in df]].copy()

        parsed_content['data_df'] = df_essential

        lat_str = parsed_content['metadata'].get('Station latitude', parsed_content['metadata'].get('Measurement latitude'))
        lon_str = parsed_content['metadata'].get('Station longitude', parsed_content['metadata'].get('Measurement longitude'))
        alt_str_m = parsed_content['metadata'].get('Measurement altitude', parsed_content['metadata'].get('Station altitude'))

        if lat_str is None:
            raise ValueError(f"Latitude not found in metadata. File: {filepath}")
        if lon_str is None:
            raise ValueError(f"Longitude not found in metadata. File: {filepath}")
        if alt_str_m is None:
            raise ValueError(f"Altitude not found in metadata. File: {filepath}")

        parsed_content['metadata']['final_latitude'] = float(lat_str)
        parsed_content['metadata']['final_longitude'] = float(lon_str)
        alt_val_str = re.sub(r'\s*m$', '', str(alt_str_m), flags=re.IGNORECASE)
        parsed_content['metadata']['final_altitude'] = float(alt_val_str)

        parsed_content['metadata']['endtime_data_column'] = _resolve_endtime_column_name(df, variable_definitions, filepath=filepath)

        station_code = parsed_content['metadata'].get('Station code')
        if not station_code:
            print(f"Warning: 'Station code' not found in metadata for {filepath}. Skipping this file.")
            return None
        parsed_content['metadata']['station_code'] = station_code

        return parsed_content

    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_station_netcdf(
    station_code: str,
    list_of_parsed_data: List[Dict[str, Any]],
    output_filepath: str,
    source_nas_filenames: List[str]
) -> None:
    """Aggregate parsed items for a station and write an xarray NetCDF file.
    """
    if not list_of_parsed_data:
        print(f"No data to process for station {station_code}. Skipping NetCDF creation.")
        return

    all_dfs = [pd.DataFrame(p['data_df']) for p in list_of_parsed_data if p['data_df'] is not None and not p['data_df'].empty]
    if not all_dfs:
        print(f"No valid DataFrames to concatenate for station {station_code}. Skipping.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if combined_df.empty or 'abs_time' not in combined_df.columns:
        print(f"Warning: Combined DataFrame is empty or missing 'abs_time' for station {station_code}. Skipping.")
        return

    cutoff_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
    combined_df = combined_df[combined_df['abs_time'] <= cutoff_date]

    if combined_df.empty:
        print(f"Warning: Combined DataFrame is empty after filtering for dates <= 2024-12-31 for station {station_code}. Skipping.")
        return

    combined_df['time_unix_float'] = (combined_df['abs_time'] - datetime(1970, 1, 1, tzinfo=timezone.utc)).dt.total_seconds()
    combined_df.dropna(subset=['time_unix_float'], inplace=True)
    combined_df['time_unix'] = combined_df['time_unix_float'].astype(np.int64)

    combined_df.drop_duplicates(subset=['time_unix'], keep='first', inplace=True)
    combined_df.sort_values(by='time_unix', inplace=True)

    if combined_df.empty:
        print(f"Warning: No valid time data after processing for station {station_code}. Skipping.")
        return

    time_components_data = combined_df['time_unix'].values

    id_data = build_id_array_from_unix_seconds(combined_df['time_unix'].values)

    first_meta = list_of_parsed_data[0]['metadata']
    latitude_val = first_meta.get('final_latitude', np.nan)
    longitude_val = first_meta.get('final_longitude', np.nan)
    altitude_val = first_meta.get('final_altitude', np.nan)

    for parsed_item in list_of_parsed_data[1:]:
        meta = parsed_item['metadata']
        if meta.get('final_latitude') != latitude_val:
            print(f"Warning: Inconsistent latitude for station {station_code} in file contributing {meta.get('Station name', '')}")
        if meta.get('final_longitude') != longitude_val:
            print(f"Warning: Inconsistent longitude for station {station_code} in file contributing {meta.get('Station name', '')}")
        if meta.get('final_altitude') != altitude_val:
            print(f"Warning: Inconsistent altitude for station {station_code} in file contributing {meta.get('Station name', '')}")

    latitude_data = np.full(len(time_components_data), latitude_val, dtype=np.float64)
    longitude_data = np.full(len(time_components_data), longitude_val, dtype=np.float64)
    altitude_data = np.full(len(time_components_data), altitude_val, dtype=np.float64)

    bc_data = combined_df['parsed_bc'].values.astype(np.float64)

    uncertainty_data = np.full(len(bc_data), np.nan, dtype=np.float64)
    has_p15 = 'parsed_p15' in combined_df.columns
    has_p84 = 'parsed_p84' in combined_df.columns

    for i in range(len(bc_data)):
        if has_p15 and has_p84 and pd.notnull(combined_df['parsed_p15'].iloc[i]) and pd.notnull(combined_df['parsed_p84'].iloc[i]):
            uncertainty_data[i] = (combined_df['parsed_p84'].iloc[i] - combined_df['parsed_p15'].iloc[i]) / 2.0
        elif pd.notnull(bc_data[i]):
            uncertainty_data[i] = np.maximum(0.25 * np.abs(bc_data[i]), 0.005)

    data_vars = {
        'bc':          (('time_components',), bc_data,          {'_FillValue': np.nan}),
        'latitude':    (('time_components',), latitude_data,    {'_FillValue': np.nan}),
        'longitude':   (('time_components',), longitude_data,   {'_FillValue': np.nan}),
        'altitude':    (('time_components',), altitude_data,    {'_FillValue': np.nan}),
        'uncertainty': (('time_components',), uncertainty_data, {'_FillValue': np.nan}),
    }

    coordinate_vars = {
        'time_components': ('time_components', time_components_data, {}),
        'id':              (('time_components',), id_data,              {}),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coordinate_vars)

    ds.attrs = {}

    ds.to_netcdf(output_filepath, format='NETCDF4')
    print(f"Successfully converted and aggregated data for station {station_code} to {output_filepath}")


def main() -> None:
    """CLI entry point for converting EBAS eBC NAS files to NetCDF by station."""
    parser = argparse.ArgumentParser(description="Convert and aggregate EBAS NAS files to NetCDF format by station.")
    parser.add_argument("source", help="Folder containing .nas files")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save output .nc files.")

    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Error: Source directory {args.source} not found or is not a directory.")
        return

    expanded_inputs = []
    try:
        expanded_inputs = list_nas_files(args.source, sort=True, case_insensitive_ext=True)
    except Exception as e:
        print(f"Error: Could not read directory {args.source}: {e}")
        return

    station_data_map = {}

    for input_filepath in expanded_inputs:
        if not os.path.exists(input_filepath):
            print(f"Error: Input file not found: {input_filepath}")
            continue

        print(f"Processing {input_filepath}...")
        parsed_data = parse_nas_file(input_filepath)

        if parsed_data and 'station_code' in parsed_data['metadata']:
            station_code = parsed_data['metadata']['station_code']
            if station_code not in station_data_map:
                station_data_map[station_code] = {'parsed_items': [], 'source_nas_filenames': []}

            station_data_map[station_code]['parsed_items'].append(parsed_data)
            station_data_map[station_code]['source_nas_filenames'].append(os.path.basename(input_filepath))
        else:
            print(f"Skipping {input_filepath} due to parsing errors or missing station code.")

    for station_code, station_info in station_data_map.items():
        list_of_parsed_data = station_info['parsed_items']
        source_nas_filenames = station_info['source_nas_filenames']

        if not list_of_parsed_data:
            print(f"No successfully parsed data for station {station_code}. Skipping.")
            continue

        output_filename = f"{station_code}.nc"

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_filepath = os.path.join(args.output_dir, output_filename)

        try:
            create_station_netcdf(station_code, list_of_parsed_data, output_filepath, source_nas_filenames)
        except Exception as e:
            print(f"Error creating NetCDF for station {station_code} from files {source_nas_filenames}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()


