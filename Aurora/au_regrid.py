#!/usr/bin/env python3
import os
import xarray as xr
import numpy as np
import xesmf as xe
from tqdm import tqdm

# --- Paths ---
forecast_path = '/home/project/17001770/weather_department/nwp/zach/aurora_folder/aurora_forecasts'
truth_path = '/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data/era5_pl_splits_6hr_final'
weights_path = "/home/project/17001770/weather_department/nwp/zach/aurora_folder/aurora_bilinear_0p25_weights.nc"
output_path = "/home/project/17001770/weather_department/nwp/zach/aurora_folder/aurora_0p25deg"
out_dir = "/home/users/industry/connect/zachloy/scratch/aurora_regridded"
os.makedirs(output_path, exist_ok=True)

# --- Prepare target grid from ERA5 truth ---
truth_sample = xr.open_dataset(os.path.join(truth_path, 'era5_pl_2024-05-19_06.grib'), engine='cfgrib',backend_kwargs={"indexpath": ""} )
lat_new = np.arange(truth_sample.latitude.min(), truth_sample.latitude.max() + 0.25, 0.25)
lon_new = np.arange(truth_sample.longitude.min(), truth_sample.longitude.max() + 0.25, 0.25)
grid_out = xr.Dataset(
    {
        "latitude": (["latitude"], lat_new),
        "longitude": (["longitude"], lon_new)
    }
)

# --- Load one forecast sample to define input grid ---
forecast_sample = xr.open_dataset(os.path.join(forecast_path, 'aurora_forecast_2024-02-06_12-out-6.nc'))

# --- Create regridder ---
regridder = xe.Regridder(
    forecast_sample,
    grid_out,
    method='bilinear',
    filename=weights_path,
    reuse_weights=True
)

# --- Process all forecast files serially ---
forecast_files = [
    f for f in os.listdir(forecast_path)
    if f.endswith(".nc") and not os.path.exists(os.path.join(output_path, f))
]

for fc_file in tqdm(forecast_files, desc="Regridding forecasts"):
    fc_path = os.path.join(forecast_path, fc_file)
    out_file = os.path.join(output_path, fc_file)
    print("processing: ", fc_file)

    if os.path.exists(out_file):
        continue  # skip already regridded

    try:
        ds = xr.open_dataset(fc_path)
        ds = ds.assign_coords(longitude=(ds.longitude % 360))

        ds_regrid = regridder(ds)
        ds_regrid = ds_regrid.sortby("latitude", ascending=False)
        ds_regrid.to_netcdf(os.path.join(out_dir, fc_file))
        ds.close()
        ds_regrid.close()

    except Exception as e:
        print(f"Failed {fc_file}: {e}")

print(f"All regridded forecasts saved to {output_path}")

