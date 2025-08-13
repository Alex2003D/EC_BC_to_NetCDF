import pandas as pd
import xarray as xr
import numpy as np


def create_dataframe_from_txt(file_path):
    df = pd.read_csv(file_path)

    columns_to_drop = ['Dataset', 'SiteCode', 'POC', 'ParamCode']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df['time'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').astype('int64') // 10**9
    df = df.drop('Date', axis=1)

    df = df.rename(columns={
        'ECf_Val': 'ec',
        'ECf_Unc': 'uncertainty',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Elevation': 'altitude'
    })

    df['ec'] = df['ec'].replace(-999, np.nan)
    df['uncertainty'] = df['uncertainty'].replace(-999, np.nan)

    df = df.sort_values('time')
    return df


def create_netcdf(df) -> xr.Dataset:
    data_vars = {
        'ec': ('time_components', df['ec'].values),
        'latitude': ('time_components', df['latitude'].values),
        'longitude': ('time_components', df['longitude'].values),
        'altitude': ('time_components', df['altitude'].values),
        'uncertainty': ('time_components', df['uncertainty'].values)
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'time_components': df['time'].values}
    )

    ds.to_netcdf('improve.nc')
    return ds


