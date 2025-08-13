# EC_BC_to_NetCDF (EBAS, IMPROVE, NASA ATOM)

EBAS eBC (NAS to NetCDF):
- python -m ec_bc_to_netcdf.ebas.ebc SOURCE_DIR -o OUTPUT_DIR

EBAS EC (NAS to NetCDF):
- python -m ec_bc_to_netcdf.ebas.ec SOURCE_DIR -o OUTPUT_DIR

IMPROVE EC (CSV to NetCDF):
- python -m ec_bc_to_netcdf.improve.cli
- Expects improve_data.txt in CWD, writes improve.nc

NASA ATOM BC (ICT + tracks to NetCDF):
- python -m ec_bc_to_netcdf.atom.cli
- Expects .ict files and flight_tracks.csv in CWD, writes total_data.nc


