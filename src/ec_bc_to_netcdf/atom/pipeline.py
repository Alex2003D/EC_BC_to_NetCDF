import pandas as pd
import os
import xarray as xr
import numpy as np
from datetime import datetime


def read_ict_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    comments = int(lines[0].split(',')[0])
    column_names = [col.strip() for col in lines[comments - 1].strip().split(',')]
    start_date_list = lines[6].strip().split(',')

    start_date = f"{start_date_list[0].strip()}-{start_date_list[1].strip()}-{start_date_list[2].strip()} 00:00:00"
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp())

    df = pd.read_csv(filename, skiprows=comments + 1, names=column_names)
    df['UTC_start'] = pd.to_numeric(df['UTC_start'], errors='coerce')
    df.replace(-9999.99, float('nan'), inplace=True)

    df['time'] = start_timestamp + df['UTC_start'].astype(int)

    required_columns = ['time', 'BC_mass_90_550_nm']
    for col in required_columns:
        if col not in df.columns:
            df[col] = float('nan')

    df = df[required_columns]
    return df


def add_additional_data(df, additional_data_filename):
    df2 = pd.read_csv(additional_data_filename)
    df2['time'] = pd.to_datetime(df2['Time']).astype(np.int64) // 10**9
    df2 = df2.drop('Time', axis=1)
    merged_df = pd.merge(df, df2[['time', 'Altitude', 'Latitude', 'Longitude']], on='time', how='inner')
    return merged_df


def process_all_ict_files(directory):
    files = os.listdir(directory)
    ict_files = [f for f in files if f.endswith('.ict')]
    all_dfs = []

    for ict_file in ict_files:
        filename = os.path.join(directory, ict_file)
        df = read_ict_file(filename)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)
    return combined_df.sort_values(by='time')


def convert_to_netcdf(df, output_file):
    ids = [int("2" + str(t)[1:]) for t in df['time']]

    uncertainty = df['BC_mass_90_550_nm'].values * 0.30

    ds = xr.Dataset(
        data_vars={
            'BC': ('time_components', df['BC_mass_90_550_nm'].values),
            'latitude': ('time_components', df['Latitude'].values),
            'longitude': ('time_components', df['Longitude'].values),
            'altitude': ('time_components', df['Altitude'].values),
            'uncertainty': ('time_components', uncertainty),
        },
        coords={
            'time_components': df['time'].values,
            'id': ('time_components', ids)
        }
    )

    _, unique_indices = np.unique(ds['time_components'], return_index=True)
    ds = ds.isel(time_components=unique_indices)

    ds.to_netcdf(output_file)

    print(f"NetCDF file created successfully:")
    print(f"First timestamp: {ds['time_components'][0].values}")
    print(f"Number of timestamps: {len(ds['time_components'])}")

    return ds


