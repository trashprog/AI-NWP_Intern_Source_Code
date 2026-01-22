#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# --- Config ---
truth_path = '/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data/era5_pl_splits_6hr_final'
forecast_path = '/home/project/17001770/weather_department/nwp/zach/gc_folder/gc_forecasts'
output_dir = '/home/project/17001770/weather_department/nwp/zach/era5_folder/regional_google_weatherbench_plots'
os.makedirs(output_dir, exist_ok=True)

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
PRESSURE_LEVELS = {"t": 850, "q": 700, "z": 500, "wind": 850, "u": 850, "v": 850}
# PRESSURE_LEVELS = {"t": 850, "q": 700, "wind": 850, "u": 850, "v": 850}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]
num_workers = 16

truth_pattern = re.compile(r"era5_pl_(\d{4}-\d{2}-\d{2})_(\d{2})\.grib")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helpers ---
def load_grib_uvqtz(filepath):
    """Load u, v, q, t, z from GRIB file."""
    try:
        ds = xr.open_dataset(
            filepath,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": ["u", "v", "q", "t", "z"]},
                "indexpath": ""
            }
        )
        return ds
    except Exception as e:
        print(f"⚠️ Could not open {filepath}: {e}")
        return xr.Dataset()

def ensure_descending_latitude(ds):
    if ds['latitude'][0] < ds['latitude'][-1]:
        return ds.sel(latitude=ds['latitude'][::-1])
    return ds

def compute_wind_rmse(u_truth, v_truth, u_forecast, v_forecast):
    rmse_u = np.sqrt(np.nanmean((u_truth.ravel() - u_forecast.ravel())**2))
    rmse_v = np.sqrt(np.nanmean((v_truth.ravel() - v_forecast.ravel())**2))
    rmse_vec = np.sqrt(rmse_u**2 + rmse_v**2)
    return rmse_u, rmse_v, rmse_vec

def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f.ravel() - t.ravel())**2))

# --- RMSE computation per file ---
def compute_rmse_for_file(truth_file, forecast_path):
    match = truth_pattern.match(os.path.basename(truth_file))
    if not match:
        return None
    truth_date_str, truth_hour_str = match.groups()
    truth_valid_time = pd.to_datetime(f"{truth_date_str} {truth_hour_str}:00")

    ds_truth = load_grib_uvqtz(truth_file)
    if not ds_truth:
        return None
    ds_truth = ensure_descending_latitude(ds_truth)

    lat_slice = slice(DOMAIN["lat_max"], DOMAIN["lat_min"])
    lon_slice = slice(DOMAIN["lon_min"], DOMAIN["lon_max"])
    ds_truth = ds_truth.sel(latitude=lat_slice, longitude=lon_slice)

    results = {var: {} for var in ["wind", "u", "v", "q", "t", "z"]}

    
    for lead in lead_times:
        fc_valid_time = truth_valid_time - pd.Timedelta(hours=lead)
        fc_date_str = fc_valid_time.strftime("%Y-%m-%d")
        fc_hour_str = fc_valid_time.strftime("%H")
        fc_file_name = f"gc_merged_{fc_date_str}_{fc_hour_str}-out-{lead}.grib"
        fc_file = os.path.join(forecast_path, fc_file_name)
        if not os.path.exists(fc_file):
            print(f"  Lead {lead}h -> Forecast file not found: {fc_file_name}")
            continue

        ds_fc = load_grib_uvqtz(fc_file)
        if not ds_fc:
            continue
        ds_fc = ensure_descending_latitude(ds_fc)
        ds_fc = ds_fc.sel(latitude=lat_slice, longitude=lon_slice)

        try:
            # Wind
            u_t = ds_truth["u"].sel(isobaricInhPa=PRESSURE_LEVELS["wind"]).values
            v_t = ds_truth["v"].sel(isobaricInhPa=PRESSURE_LEVELS["wind"]).values
            u_f = ds_fc["u"].sel(isobaricInhPa=PRESSURE_LEVELS["wind"]).values
            v_f = ds_fc["v"].sel(isobaricInhPa=PRESSURE_LEVELS["wind"]).values
            rmse_u, rmse_v, rmse_wind = compute_wind_rmse(u_t, v_t, u_f, v_f)
            results["u"][lead] = rmse_u
            results["v"][lead] = rmse_v
            results["wind"][lead] = rmse_wind

            # Temperature 850
            t_t = ds_truth["t"].sel(isobaricInhPa=PRESSURE_LEVELS["t"]).values
            t_f = ds_fc["t"].sel(isobaricInhPa=PRESSURE_LEVELS["t"]).values
            results["t"][lead] = compute_scalar_rmse(t_f, t_t)

            # Specific humidity 700
            q_t = ds_truth["q"].sel(isobaricInhPa=PRESSURE_LEVELS["q"]).values
            q_f = ds_fc["q"].sel(isobaricInhPa=PRESSURE_LEVELS["q"]).values
            results["q"][lead] = compute_scalar_rmse(q_f, q_t)

            # Geopotential 500
            z_t = ds_truth["z"].sel(isobaricInhPa=PRESSURE_LEVELS["z"]).values
            z_f = ds_fc["z"].sel(isobaricInhPa=PRESSURE_LEVELS["z"]).values
            results["z"][lead] = compute_scalar_rmse(z_f, z_t)

        except Exception:
            for var in results:
                results[var][lead] = np.nan

        ds_fc.close()

    ds_truth.close()
    return os.path.basename(truth_file), results

# --- Parallel execution ---
def main():
    truth_files = [os.path.join(truth_path, f) for f in os.listdir(truth_path) 
                   if truth_pattern.match(f) and f.endswith('.grib')]

    # RMSE storage, only for variables in PRESSURE_LEVELS
    rmse_store = {var: {level: {lead: [] for lead in lead_times} 
                        for level in PRESSURE_LEVELS.values()} 
                  for var in PRESSURE_LEVELS}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_rmse_for_file, tf, forecast_path): tf for tf in truth_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ERA5–Pangu RMSE"):
            result = fut.result()
            if result is None:
                continue
            _, data = result
            for var in data:
                if var not in PRESSURE_LEVELS:
                    continue
                for lead in data[var]:
                    rmse_store[var][PRESSURE_LEVELS[var]][lead].append(data[var][lead])

    # Compute mean RMSE
    mean_rmse = {var: {level: {} for level in PRESSURE_LEVELS.values()} for var in PRESSURE_LEVELS}
    for var in rmse_store:
        for level in PRESSURE_LEVELS.values():
            for lead in lead_times:
                arr = np.array(rmse_store[var][level][lead])
                mean_rmse[var][level][lead] = np.nanmean(arr) if arr.size > 0 else np.nan

    # Save results
    npz_path = os.path.join(output_dir, "era5_vs_gc_rmse_allvars.npz")
    np.savez(npz_path, rmse=mean_rmse, lead_times=np.array(lead_times), pressure_levels=np.array(list(PRESSURE_LEVELS.values())))
    print(f"✅ Aggregated RMSE saved to {npz_path}")

    # Plotting
    for var, title in [("wind", "850 hPa Wind Vector"), 
                       ("u", "850 hPa U-component"), 
                       ("v", "850 hPa V-component"), 
                       ("q", "700 hPa Specific Humidity"), 
                       ("t", "850 hPa Temperature"),
                        ("z", "500 hPa Geopotential")]:
        plt.figure(figsize=(10,6))
        for level in PRESSURE_LEVELS.values():
            y = [mean_rmse[var][level][lt] for lt in lead_times]
            plt.plot(lead_times, y, marker='o', label=f"{level} hPa")
        plt.title(f"ERA5 vs Graphcast RMSE: {title} (SEA Region)")
        plt.xlabel("Lead Time [hours]")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"era5_vs_gc_rmse_{var}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print("All plots saved to:", output_dir)


if __name__ == "__main__":
    main()
