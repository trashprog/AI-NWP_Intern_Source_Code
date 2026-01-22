#!/usr/bin/env python3
import xarray as xr
import matplotlib.pyplot as plt
import cfgrib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import re
import torch
import pickle
from aurora import AuroraHighRes, Batch, Metadata, rollout
from huggingface_hub import hf_hub_download

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================
# running the inference
# ==========================
MERGED_DIR = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/au_input_files"
OUTPUT_DIR = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/au_forecast_files"
MODEL_CKPT = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/aurora-0.1-finetuned.ckpt"
FORECAST_HOURS = 48
STEP_HOURS = 6
N_STEPS = FORECAST_HOURS // STEP_HOURS

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("STEP 1. Running model Inference...")

# ==========================
# Helper functions
# ==========================
def fix_coordinates(ds):
    lon_values = ds.longitude.values.astype(np.float32)
    new_lons = np.mod(lon_values, 360)
    new_lons[new_lons == 360] = 0.0
    ds = ds.assign_coords(longitude=new_lons)
    ds = ds.sortby("longitude")
    return ds

def lon_360_to_180(lon):
    lon = lon.copy()
    lon[lon > 180] -= 360
    return lon

# ==========================
# Load static variables
# ==========================
print("Downloading static variables...")
static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.1-static.pickle")
with open(static_path, "rb") as f:
    static_vars = pickle.load(f)
print("Static variables loaded.")

# ==========================
# Load model
# ==========================
print("Loading model checkpoint...")
model = AuroraHighRes()
model.load_checkpoint_local(MODEL_CKPT)
model.eval()
model = model.to("cuda")
print("Model ready on device:", next(model.parameters()).device)

# ==========================
# Process each merged GRIB file
# ==========================
for fname in sorted(os.listdir(MERGED_DIR)):
    if not fname.endswith(".grib"):
        continue

    fpath = os.path.join(MERGED_DIR, fname)
    base_name = os.path.splitext(fname)[0]  # e.g. aurora_merged_2024-01-15_18
    date_str = base_name.replace("aurora_merged_", "")

    # Skip if all forecast steps already exist
    # --- Robust incremental forecasting check ---
    init_prefix = f"aurora_forecast_{date_str}"

    expected_files = {
        f"{init_prefix}-out-{step*STEP_HOURS}.nc"
        for step in range(1, N_STEPS + 1)
    }

    existing_files = set(os.listdir(OUTPUT_DIR))

    already_done = expected_files.issubset(existing_files)

    if already_done:
        print(f"Skipping {fname}: all {N_STEPS} forecast steps already exist.")
        continue

    # --- Load datasets ---
    sfc_ds = xr.load_dataset(fpath, engine="cfgrib",
                             filter_by_keys={"typeOfLevel": "surface"},
                             decode_timedelta=True)
    pl_ds = xr.load_dataset(fpath, engine="cfgrib",
                             filter_by_keys={"typeOfLevel": "isobaricInhPa"},
                             decode_timedelta=True)

    # --- Fix longitudes ---
    pl_ds = fix_coordinates(pl_ds)
    sfc_ds = fix_coordinates(sfc_ds)

    # --- Determine forecast initialization from filename ---
    file_hour = int(base_name.split("_")[-1])  # e.g., "18" from "aurora_merged_2024-01-01_18"
    valid_times = pd.to_datetime(pl_ds.valid_time.values)
    matching_times = [t for t in valid_times if t.hour == file_hour]

    if not matching_times:
        print(f"Skipping {fname}: no matching timestep found")
        continue

    date = matching_times[0].to_pydatetime()
    print(f"Forecast initialization: {date}")

    # --- Rename surface vars ---
    sfc_ds_renamed = sfc_ds.rename({"t2m": "2t", "u10": "10u", "v10": "10v"})
    surf_vars = {
        "2t": torch.from_numpy(sfc_ds_renamed["2t"].values[:2][None]),
        "10u": torch.from_numpy(sfc_ds_renamed["10u"].values[:2][None]),
        "10v": torch.from_numpy(sfc_ds_renamed["10v"].values[:2][None]),
        "msl": torch.from_numpy(sfc_ds_renamed["msl"].values[:2][None]),
    }

    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
    atmos_vars = {
        "t": torch.from_numpy(pl_ds["t"].sel(isobaricInhPa=list(levels)).values[:2][None]),
        "u": torch.from_numpy(pl_ds["u"].sel(isobaricInhPa=list(levels)).values[:2][None]),
        "v": torch.from_numpy(pl_ds["v"].sel(isobaricInhPa=list(levels)).values[:2][None]),
        "q": torch.from_numpy(pl_ds["q"].sel(isobaricInhPa=list(levels)).values[:2][None]),
        "z": torch.from_numpy(pl_ds["z"].sel(isobaricInhPa=list(levels)).values[:2][None]),
    }

    metadata = Metadata(
        lat=torch.from_numpy(pl_ds.latitude.values),
        lon=torch.from_numpy(pl_ds.longitude.values),
        time=(date,),
        atmos_levels=levels,
    )

    batch = Batch(
        surf_vars=surf_vars,
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        # static_vars={k: torch.from_numpy(v).to("cpu") for k, v in static_vars.items()},
        atmos_vars=atmos_vars,
        metadata=metadata,
    )

    # --- Run forecast rollout ---
    print(f"Running {FORECAST_HOURS}h forecast (steps of {STEP_HOURS}h)...")
    # with torch.inference_mode():
    #     preds = [p.to("cuda") for p in rollout(model, batch, steps=N_STEPS)]
    with torch.inference_mode():
        preds = []
        for p in rollout(model, batch, steps=N_STEPS):
            p = p.to("cpu")          # move OFF GPU immediately
            preds.append(p)
            torch.cuda.empty_cache()  # free memory for next step

    # --- Save outputs and print per truth file ---
    print(f"Truth file: {fname}")
    # --- Save outputs ---
    for step_idx, batch_step in enumerate(preds, start=1):
        step_time = np.datetime64(date) + np.timedelta64(step_idx * STEP_HOURS, "h")
        lat = batch_step.metadata.lat.cpu().numpy()
        lon = lon_360_to_180(batch_step.metadata.lon.cpu().numpy())

        # Atmospheric variables (z, t, u, v, q)
        atm_data = {v: (("isobaricInhPa", "latitude", "longitude"),
                         np.squeeze(t.cpu().numpy()).astype(np.float32))
                    for v, t in batch_step.atmos_vars.items()}
        
        # Surface variables (2t, 10u, 10v, msl)
        surf_data = {v: (("latitude", "longitude"),
                          np.squeeze(t.cpu().numpy()).astype(np.float32))
                     for v, t in batch_step.surf_vars.items()}
        
        # Static variables (Rename 'z' to 'z_sfc' to avoid collision)
        static_data = {}
        for v, t in batch_step.static_vars.items():
            var_name = "z_sfc" if v == "z" else v  # RENAME STATIC Z
            static_data[var_name] = (("latitude", "longitude"), 
                                      t.cpu().numpy().astype(np.float32))

        # Combine dictionaries safely
        combined_vars = {**atm_data, **surf_data, **static_data}

        ds = xr.Dataset(
            data_vars=combined_vars,
            coords={
                "time": ("time", [step_time]),
                "valid_time": ("time", [step_time]),
                "isobaricInhPa": ("isobaricInhPa", levels),
                "latitude": ("latitude", lat),
                "longitude": ("longitude", lon),
            },
        )

        lead_hours = step_idx * STEP_HOURS
        out_name = f"aurora_forecast_{date.strftime('%Y-%m-%d_%H')}-out-{lead_hours}.nc"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        ds.to_netcdf(out_path)
        print(f"  Saved forecast: {out_name}")

print("\nAll forecasts completed successfully!")


print("STEP 2. Regrid the forecasts")
import xesmf as xe

forecast_path = '/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/au_forecast_files'
truth_path = '/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Truth/truth_files'
weights_path = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/aurora_bilinear_0p25_weights.nc"
output_path = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/au_forecast_regridded_files"

# --- Prepare target grid from ERA5 truth ---
truth_sample = xr.open_dataset(os.path.join(truth_path, 'era5_pl_2024-05-15_06.grib'), engine='cfgrib',backend_kwargs={"indexpath": ""} )
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
    if f.endswith(".nc") and not os.path.exists(os.path.join(output_path, f"{os.path.splitext(f)[0]}_regridded.nc"))
]

for fc_file in forecast_files:
    fc_path = os.path.join(forecast_path, fc_file)
    out_file = os.path.join(output_path, f"{base_name}_regridded.nc")
    print("processing: ", fc_file)

    if os.path.exists(out_file):
        continue  # skip already regridded

    try:
        ds = xr.open_dataset(fc_path)
        ds = ds.assign_coords(longitude=(ds.longitude % 360))

        ds_regrid = regridder(ds)
        ds_regrid = ds_regrid.sortby("latitude", ascending=False)
        ds_regrid.to_netcdf(os.path.join(output_path, fc_file))
        ds.close()
        ds_regrid.close()

    except Exception as e:
        print(f"Failed {fc_file}: {e}")

print(f"All regridded forecasts saved to {output_path}")


print("STEP 3. Compute the weatherbench metrics")

from metric_utils import (
    slice_domain,
    load_dataset,
    compute_drwb_pl,
    compute_drwb_sfc,
    compute_wind_rmse,
    extract_trwb_vars,
    compute_weatherbenches
)

truth_dir = '/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Truth/truth_files'


