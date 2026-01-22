import os
import re
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import xarray as xr
import pandas as pd

# --- Config ---
truth_path = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/era5_2024to2025/era5_pl_splits_6hr_final'
forecast_path = '/home/project/17001770/ccrs_dwr/nwp/zach/aifs_folder/aifs_output_regridded'
DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]
PRESSURE_LEVELS_MSE = [850, 200]
file_prefix = 'era5_aifs_drwb_'
output_dir = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/era5_dynamic_weatherbench_rmse'
os.makedirs(output_dir, exist_ok=True)

# Define Monsoon Periods
MONSOON_PERIODS = {
    "Inter-monsoon": {4, 5, 10, 11},
    "Southwest-monsoon": {6, 7, 8, 9},
    "Northeast-monsoon": {12, 1, 2, 3},
}

truth_files = [f for f in os.listdir(truth_path) if f.endswith('.grib')]
forecast_files = [f for f in os.listdir(forecast_path) if f.endswith('.nc')]

# --helpers
def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f.ravel() - t.ravel())**2))

def compute_mse(t, z, q, cp = 1004, g = 9.81, Lv = 2.5e6):
    # temperature, geopotential z, specific humidity q
    g_z = z/g
    return cp*t + g*g_z + Lv*q

def compute_mse_vorticity_convergence(u, v, lats, lons, mse):
    """
    Compute horizontal MSE convergence:
    - u, v: wind components (2D: lat x lon or 3D: lev x lat x lon)
    - lats, lons: 1D arrays
    - mse: MSE array of same shape as u/v
    Returns: horizontal convergence array (same shape)
    """
    Re = 6371000  # Earth radius in meters

    # Convert lat/lon spacing to radians
    dlat = np.deg2rad(np.gradient(lats))          # 1D array
    dlon = np.deg2rad(np.gradient(lons))          # 1D array
    coslat = np.cos(np.deg2rad(lats))
    coslat = np.clip(coslat, 1e-3, 1.0)
    # print("cos(lat) min/max:", coslat.min(), coslat.max())

    # vorticity calculation
    zeta = np.gradient(v, axis=-1) / (dlon[None, :] * Re * coslat[:, None]) - np.gradient(u, axis=-2) / (dlat[:, None] * Re)
    
    # MSE fluxes
    Fx = u
    Fy = v
    
    # mse
    mse_dudx = np.gradient(Fx*mse, axis=-1) / (dlon[None, :] * Re * coslat[:, None])
    mse_dvdy = np.gradient(Fy*mse, axis=-2) / (dlat[:, None] * Re)

    # vorticity
    zeta_dudx = np.gradient(Fx*mse*zeta, axis=-1) / (dlon[None, :] * Re * coslat[:, None])
    zeta_dvdy = safe_gradient(Fy*zeta*mse, axis=-2, spacing=(dlat[:, None] * Re))
    nan_mask = np.isnan(v*zeta*mse)

    # convert vorticity 0.0 to nans
    clean_array = np.where(nan_mask, 0.0, v*zeta*mse) # Replace NaNs with 0 for gradient computation
    zeta_dvdy = np.gradient(clean_array, axis=-2) / (dlat[:, None] * Re) # Compute gradient safely
    zeta_dvdy[nan_mask] = np.nan  # Restore NaNs

    # Horizontal divergence
    mse_div_F = mse_dudx + mse_dvdy
    zeta_div_F = zeta_dudx + zeta_dvdy

    # Horizontal convergence = - divergence
    mse_convergence = np.where(mse_div_F < 0, mse_div_F, np.nan)
    mse_divergence = np.where(mse_div_F > 0, mse_div_F, np.nan)
    zeta_convergence = np.where(zeta_div_F < 0, zeta_div_F, np.nan)
    zeta_divergence = np.where(zeta_div_F > 0, zeta_div_F, np.nan)
    
    return mse_convergence, mse_divergence, zeta_convergence, zeta_divergence


def safe_gradient(arr, axis, spacing):
    """
    Compute gradient while ignoring NaNs:
    - Replaces NaNs with nearest valid value along the axis.
    """
    # Make a copy so original array isn’t modified
    arr_copy = arr.copy()
    
    # Fill NaNs along the axis
    # Forward-fill
    arr_copy = np.where(np.isnan(arr_copy), np.nan_to_num(arr_copy, nan=0.0), arr_copy)
    
    # Or better: interpolate NaNs
    # But simple approach: mask them with zeros
    grad = np.gradient(arr_copy, axis=axis) / spacing
    return grad


# --- Function to compute RMSE per truth file ---
def process_file(truth_file):
    rmse_result = {
        'mse_convergence': {lead: [] for lead in lead_times},
        'mse_divergence': {lead: [] for lead in lead_times},
        'vorticity_convergence': {lead: [] for lead in lead_times}
    }

    match = re.match(r"era5_pl_(\d{4}-\d{2}-\d{2})_(\d{2})\.grib", truth_file)
    if not match:
        return rmse_result
    truth_date_str, truth_hour_str = match.groups()
    truth_valid_time = pd.to_datetime(f"{truth_date_str} {truth_hour_str}:00")

    truth_ds = xr.open_dataset(
        os.path.join(truth_path, truth_file),
        engine='cfgrib',
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName":["u","v","t","z","q"]}, "indexpath": ""}
    )
    truth_ds = truth_ds.sel(latitude=slice(DOMAIN['lat_max'], DOMAIN['lat_min']),
                            longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))

    for lead in lead_times:
        fc_valid_time = truth_valid_time - pd.Timedelta(hours=lead)
        fc_file_name = f"aifs_gridded_{fc_valid_time.strftime('%Y-%m-%d')}_{fc_valid_time.strftime('%H')}-out-{lead}_regridded.nc"
        if fc_file_name not in forecast_files:
            continue

        fc_ds = xr.open_dataset(os.path.join(forecast_path, fc_file_name))
        fc_ds = fc_ds.sel(latitude=slice(DOMAIN['lat_min'], DOMAIN['lat_max']),
                          longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))
        fc_ds = fc_ds.sel(latitude=fc_ds['latitude'][::-1])  # flip if necessary

        # Check lat/lon match
        if not (np.array_equal(truth_ds['latitude'].values, fc_ds['latitude'].values) and
                np.array_equal(truth_ds['longitude'].values, fc_ds['longitude'].values)):
            continue

        for level in PRESSURE_LEVELS_MSE:
            if level not in truth_ds['isobaricInhPa'].values:
                continue

            truth_u = truth_ds['u'].sel(isobaricInhPa=level).values
            truth_v = truth_ds['v'].sel(isobaricInhPa=level).values
            truth_t = truth_ds['t'].sel(isobaricInhPa=level).values
            truth_z = truth_ds['z'].sel(isobaricInhPa=level).values
            truth_q = truth_ds['q'].sel(isobaricInhPa=level).values

            fc_u = fc_ds['u'].sel(level=level).squeeze().values
            fc_v = fc_ds['v'].sel(level=level).squeeze().values
            fc_t = fc_ds['t'].sel(level=level).squeeze().values
            fc_z = fc_ds['z'].sel(level=level).squeeze().values
            fc_q = fc_ds['q'].sel(level=level).squeeze().values

            mse_truth = compute_mse(truth_t, truth_z, truth_q)
            mse_fc = compute_mse(fc_t, fc_z, fc_q)

            mse_conv_truth, mse_div_truth, zeta_conv_truth, _ = compute_mse_vorticity_convergence(
                truth_u, truth_v, truth_ds.latitude.values, truth_ds.longitude.values, mse_truth
            )
            mse_conv_fc, mse_div_fc, zeta_conv_fc, _ = compute_mse_vorticity_convergence(
                fc_u, fc_v, fc_ds.latitude.values, fc_ds.longitude.values, mse_fc
            )

            if level == 850:
                rmse_result['mse_convergence'][lead].append(compute_scalar_rmse(mse_conv_fc, mse_conv_truth))
                rmse_result['vorticity_convergence'][lead].append(compute_scalar_rmse(zeta_conv_fc, zeta_conv_truth))
            if level == 200:
                rmse_result['mse_divergence'][lead].append(compute_scalar_rmse(mse_div_fc, mse_div_truth))

    return rmse_result

# --- Categorize files by Season ---
files_by_season = {season: [] for season in MONSOON_PERIODS}

for f in truth_files:
    match = re.match(r"era5_pl_(\d{4}-\d{2}-\d{2})_(\d{2})\.grib", f)
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

# --- Loop through each season and process ---
for season_name, season_files in files_by_season.items():
    if not season_files:
        print(f"No files found for {season_name}, skipping.")
        continue

    print(f"Processing {season_name} with {len(season_files)} files...")

    # Initialize container for this season
    season_rmse = {
        'mse_convergence': {lead: [] for lead in lead_times},
        'mse_divergence': {lead: [] for lead in lead_times},
        'vorticity_convergence': {lead: [] for lead in lead_times}
    }

    # Parallel execution for the current season's files
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_file, season_files), 
                            total=len(season_files), 
                            desc=f"Processing {season_name}"))

    # Combine results
    for res in results:
        for var in season_rmse:
            for lead in lead_times:
                season_rmse[var][lead].extend(res[var][lead])

    # --- Save NPZ for this season ---
    # Create a safe filename (replace spaces or hyphens if preferred, though standard here is fine)
    safe_season_name = season_name.replace("-", "_").lower()
    npz_file = os.path.join(output_dir, f"{file_prefix}{safe_season_name}_rmse_dict.npz")
    np.savez(npz_file, **season_rmse)
    print(f"Saved RMSE dictionary for {season_name}: {npz_file}")

    # --- Plot RMSE for this season ---
    for var in season_rmse:
        plt.figure(figsize=(10,6))
        mean_rmse = [np.mean(season_rmse[var][lead]) if season_rmse[var][lead] else np.nan for lead in lead_times]
        
        plt.plot(lead_times, mean_rmse, marker='o')
        plt.xlabel("Lead time (h)")
        plt.ylabel("RMSE")
        plt.title(f"ERA5 vs AIFS RMSE: {var} ({season_name})")
        plt.grid(True)
        
        plot_file = os.path.join(output_dir, f"{file_prefix}{safe_season_name}_{var}_rmse.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_file}")