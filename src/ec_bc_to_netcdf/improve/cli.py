import pandas as pd
from .pipeline import create_dataframe_from_txt, create_netcdf


def main():
    file_path = "improve_data.txt"
    df = create_dataframe_from_txt(file_path)
    nan_count = df['uncertainty'].isna().sum()
    print(f"\nNumber of NaN values in uncertainty column: {nan_count}")
    ds = create_netcdf(df)
    print(f"First timestamp: {ds['time_components'][0].values}")
    print(f"Number of timestamps: {len(ds['time_components'])}")


if __name__ == "__main__":
    main()


