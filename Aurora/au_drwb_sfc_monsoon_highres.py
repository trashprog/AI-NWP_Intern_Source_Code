#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ---
truth_sfc_path = '/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data/era5_sfc_splits_6hr'
truth_t2m_path = '/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data/era5_sfc_splits_6hr_t2m'
forecast_path = '/home/project/17001770/weather_department/nwp/zach/aurora_folder/aurora_forecasts'
output_dir = '/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_dynamic_weatherbench_rmse'
os.makedirs(output_dir, exist_ok=True)

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]

truth_pattern = re.compile(r"era5_sfc_(\d{4}-\d{2}-\d{2})_(\d{2})\.grib")

forecast_files = [f for f in os.listdir(forecast_path) if f.endswith('.nc')]
truth_sfc_files = [f for f in os.listdir(truth_sfc_path) if f.endswith('.grib')]
truth_t2m_files = [f for f in os.listdir(truth_t2m_path) if f.endswith('.grib')]


# ------------------------------------------------------------------
# Extract month helper
# ------------------------------------------------------------------
def extract_month(fname):
    m = truth_pattern.match(fname)
    if not m:
        return None
    date_str, hour = m.groups()
    return int(date_str.split("-")[1])

def lon_deg_to_360(ds):
    lon = ds.longitude.astype(np.float32)
    lon = np.mod(lon, 360.0)
    lon = xr.where(lon == 360.0, 0.0, lon)
    ds = ds.assign_coords(longitude=lon)
    ds = ds.sortby("longitude").sortby("latitude")
    return ds


# ------------------------------------------------------------------
# Physics
# ------------------------------------------------------------------
def compute_t2m_energy(u10, v10, lats, lons, t2m, cp=1004, density_sfc=1.2):
    cp_t2m = cp * t2m * density_sfc
    Re = 6371000
    dlat = np.deg2rad(np.gradient(lats))
    dlon = np.deg2rad(np.gradient(lons))
    coslat = np.clip(np.cos(np.deg2rad(lats)), 1e-3, 1.0)

    dudx = np.gradient(u10 * cp_t2m, axis=-1) / (dlon[None, :] * Re * coslat[:, None])
    dvdy = np.gradient(v10 * cp_t2m, axis=-2) / (dlat[:, None] * Re)
    return dudx + dvdy


def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f.ravel() - t.ravel())**2))


# ------------------------------------------------------------------
# Per-file computation
# ------------------------------------------------------------------
def process_file(sfc_file):
    rmse_result = {lead: [] for lead in lead_times}

    match = truth_pattern.match(sfc_file)
    if not match:
        return rmse_result

    date_str, hour_str = match.groups()
    valid_time = pd.to_datetime(f"{date_str} {hour_str}:00")

    # Truth data
    truth_sfc_ds = xr.open_dataset(os.path.join(truth_sfc_path, sfc_file), engine='cfgrib').sortby("longitude").sortby("latitude")
    truth_sfc_ds = truth_sfc_ds.sel(latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']), longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))

    t2m_file = f"era5_sfc_{date_str}_{hour_str}_t2m.grib"
    truth_t2m_ds = xr.open_dataset(os.path.join(truth_t2m_path, t2m_file), engine='cfgrib').sortby("longitude").sortby("latitude")
    truth_t2m_ds = truth_t2m_ds.sel(latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']), longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))

    u10_truth = truth_sfc_ds['u10'].values
    v10_truth = truth_sfc_ds['v10'].values
    t2m_truth = truth_t2m_ds['t2m'].values

    for lead in lead_times:
        fc_valid_time = valid_time - pd.Timedelta(hours=lead)
        fc_file_name = f"aurora_forecast_{fc_valid_time.strftime('%Y-%m-%d')}_{fc_valid_time.strftime('%H')}-out-{lead}.nc"

        if fc_file_name not in forecast_files:
            print(fc_file_name, " not found!")
            continue

        fc_ds = lon_deg_to_360(xr.open_dataset(os.path.join(forecast_path, fc_file_name)))

        fc_ds = fc_ds.interp(
            longitude=truth_sfc_ds['longitude'].values,
            latitude=truth_sfc_ds['latitude'].values,
            method="nearest").sel(latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']),
                                  longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))

        u10_fc = fc_ds['10u'].values
        v10_fc = fc_ds['10v'].values
        t2m_fc = fc_ds['2t'].values

        energy_truth = compute_t2m_energy(
            u10_truth, v10_truth, truth_sfc_ds['latitude'].values, truth_sfc_ds['longitude'].values, t2m_truth
        )
        energy_fc = compute_t2m_energy(
            u10_fc, v10_fc, fc_ds['latitude'].values, fc_ds['longitude'].values, t2m_fc
        )

        rmse_result[lead].append(compute_scalar_rmse(energy_fc, energy_truth))

    return rmse_result


# ------------------------------------------------------------------
# Season runner
# ------------------------------------------------------------------
def run_season(month_set, label):
    print(f"\n=== Running season: {label} ===")

    season_files = [f for f in truth_sfc_files if extract_month(f) in month_set]
    print(f"{label}: {len(season_files)} files")

    rmse_dict = {lead: [] for lead in lead_times}

    with ProcessPoolExecutor(max_workers=16) as exe:
        futures = [exe.submit(process_file, f) for f in season_files]
        for fut in tqdm(as_completed(futures), total=len(season_files), desc=f"{label}"):
            res = fut.result()
            for lead in lead_times:
                rmse_dict[lead].extend(res[lead])

    # Save
    npz_path = os.path.join(output_dir, f"era5_auro_surface_energy_rmse_{label}_highres.npz")
    rmse_dict_str = {str(k): v for k, v in rmse_dict.items()}
    np.savez(npz_path, **rmse_dict_str)
    print(f"Saved: {npz_path}")

    # Plot
    plt.figure(figsize=(10,6))
    mean_rmse = [np.nanmean(rmse_dict[lead]) for lead in lead_times]
    plt.plot(lead_times, mean_rmse, marker='o')
    plt.xlabel("Lead time (h)")
    plt.ylabel("RMSE")
    plt.title(f"Surface Temperature Energy RMSE 0.1 Degrees({label})")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"era5_auro_surface_energy_rmse_{label}_highres.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_season({4, 5, 10, 11}, "Intermonsoon")
    run_season({12, 1, 2, 3}, "NE")
    run_season({6, 7, 8, 9}, "SW")