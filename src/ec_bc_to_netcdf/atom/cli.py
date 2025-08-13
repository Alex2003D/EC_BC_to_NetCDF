import os
from .pipeline import process_all_ict_files, add_additional_data, convert_to_netcdf


def main():
    directory = os.getcwd()
    combined_df = process_all_ict_files(directory)
    df = add_additional_data(combined_df, "flight_tracks.csv")
    output_nc = "total_data.nc"
    convert_to_netcdf(df, output_nc)


if __name__ == "__main__":
    main()


