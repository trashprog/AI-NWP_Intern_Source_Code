import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Config
# ----------------------------------------------------
truth_sfc_path = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/era5_2024to2025/era5_sfc_splits_6hr_regridded'
truth_t2m_path = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/era5_2024to2025/era5_sfc_splits_6hr_t2m_regridded'
forecast_path = '/home/project/17001770/ccrs_dwr/nwp/zach/aifs_folder/aifs_output_regridded'

save_dir = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/era5_dynamic_weatherbench_rmse'
os.makedirs(save_dir, exist_ok=True)

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]

# Define Monsoon Periods
MONSOON_PERIODS = {
    "Inter-monsoon": {4, 5, 10, 11},
    "Southwest-monsoon": {6, 7, 8, 9},
    "Northeast-monsoon": {12, 1, 2, 3},
}

truth_pattern = re.compile(r"era5_sfc_(\d{4}-\d{2}-\d{2})_(\d{2})_regridded\.grib")

forecast_files = [f for f in os.listdir(forecast_path) if f.endswith('.nc')]
truth_sfc_files = [f for f in os.listdir(truth_sfc_path) if f.endswith('.grib')]
truth_t2m_files = [f for f in os.listdir(truth_t2m_path) if f.endswith('.grib')]


# ----------------------------------------------------
# Computation functions
# ----------------------------------------------------
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
    return np.sqrt(np.nanmean((f.ravel() - t.ravel()) ** 2))

# ----------------------------------------------------
# Worker function
# ----------------------------------------------------
def process_truth_file(sfc_file):
    match = truth_pattern.match(sfc_file)
    if not match:
        return []

    date_str, hour_str = match.groups()
    valid_time = pd.to_datetime(f"{date_str} {hour_str}:00")

    # Load truth
    truth_sfc_ds = xr.open_dataset(os.path.join(truth_sfc_path, sfc_file), engine='cfgrib')
    truth_sfc_ds = truth_sfc_ds.sel(
        latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']),
        longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max'])
    ).sortby("latitude", ascending=False)

    t2m_file = f"era5_sfc_{date_str}_{hour_str}_t2m_regridded.grib"
    if t2m_file not in truth_t2m_files:
        return []

    truth_t2m_ds = xr.open_dataset(os.path.join(truth_t2m_path, t2m_file), engine='cfgrib')
    truth_t2m_ds = truth_t2m_ds.sel(
        latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']),
        longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max'])
    ).sortby("latitude", ascending=False)

    results = []

    u10_truth = truth_sfc_ds['u10'].values
    v10_truth = truth_sfc_ds['v10'].values
    t2m_truth = truth_t2m_ds['t2m'].values
    lats = truth_sfc_ds['latitude'].values
    lons = truth_sfc_ds['longitude'].values

    truth_energy = compute_t2m_energy(u10_truth, v10_truth, lats, lons, t2m_truth)

    for lead in lead_times:
        fc_valid_time = valid_time - pd.Timedelta(hours=lead)
        fc_file = f"aifs_gridded_{fc_valid_time:%Y-%m-%d}_{fc_valid_time:%H}-out-{lead}_regridded.nc"

        if fc_file not in forecast_files:
            continue

        fc_ds = xr.open_dataset(os.path.join(forecast_path, fc_file))
        fc_ds = fc_ds.sel(
            latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']),
            longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max'])
        )
        fc_ds = fc_ds.sel(latitude=fc_ds['latitude'][::-1])

        if not (np.array_equal(lats, fc_ds['latitude'].values) and
                np.array_equal(lons, fc_ds['longitude'].values)):
            continue

        u10_fc = fc_ds['10u'].values
        v10_fc = fc_ds['10v'].values
        t2m_fc = fc_ds['2t'].values
        fc_energy = compute_t2m_energy(u10_fc, v10_fc, lats, lons, t2m_fc)

        rmse_val = compute_scalar_rmse(fc_energy, truth_energy)
        results.append((lead, rmse_val))

    return results


# ----------------------------------------------------
# Main Execution (Season Loop)
# ----------------------------------------------------

# 1. Categorize files by season
files_by_season = {season: [] for season in MONSOON_PERIODS}

for f in truth_sfc_files:
    match = truth_pattern.match(f)
    if match:
        date_str = match.group(1)
        # Parse date to extract month
        file_date = pd.to_datetime(date_str)
        month = file_date.month
        
        # Assign file to appropriate season list
        for season_name, season_months in MONSOON_PERIODS.items():
            if month in season_months:
                files_by_season[season_name].append(f)
                break

# 2. Process each season
for season_name, season_files in files_by_season.items():
    if not season_files:
        print(f"No files found for {season_name}, skipping.")
        continue

    print(f"Processing {season_name} with {len(season_files)} files...")
    
    # Reset results container for this season
    season_results = {lead: [] for lead in lead_times}

    # Parallel Execution
    with ProcessPoolExecutor(max_workers=1) as ex:
        futures = [ex.submit(process_truth_file, f) for f in season_files]

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {season_name}"):
            res = fut.result()
            for lead, val in res:
                season_results[lead].append(val)

    # Generate safe filename prefix
    safe_season_name = season_name.replace("-", "_").lower()

    # ----------------------------------------------------
    # Save NPZ for this season
    # ----------------------------------------------------
    npz_path = os.path.join(save_dir, f"era5_aifs_drwb_mse_ste_{safe_season_name}_rmse_dict.npz")
    string_key_results = {str(k): np.array(v) for k, v in season_results.items()}
    np.savez(npz_path, **string_key_results)
    print(f"Saved NPZ to: {npz_path}")

    # ----------------------------------------------------
    # Plot for this season
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    means = [np.nanmean(season_results[lead]) if season_results[lead] else np.nan for lead in lead_times]
    plt.plot(lead_times, means, marker='o')
    plt.title(f"ERA5 vs AIFS: STE RMSE ({season_name})")
    plt.xlabel("Lead Time (hours)")
    plt.ylabel("RMSE")
    plt.grid(True)

    plot_path = os.path.join(save_dir, f"era5_aifs_drwb_mse_ste_{safe_season_name}_rmse.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {plot_path}")