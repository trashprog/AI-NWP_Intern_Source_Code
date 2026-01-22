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

# ---------------- Config ----------------
truth_path = "/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data/era5_pl_splits_6hr_final"
forecast_path = "/home/project/17001770/weather_department/nwp/zach/aurora_folder/aurora_0p25deg"
output_dir = "/home/project/17001770/weather_department/nwp/zach/era5_folder/regional_google_weatherbench_plots"
os.makedirs(output_dir, exist_ok=True)

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
PRESSURE_LEVELS = {
    "wind": 850,
    "u": 850,
    "v": 850,
    "t": 850,
    "q": 700,
    "z": 500,
}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]
num_workers = 16

# --- Monsoon definitions ---
MONSOONS = {
    "NE_monsoon": {12, 1, 2, 3},
    "SW_monsoon": {6, 7, 8, 9},
    "Inter_monsoon": {4, 5, 10, 11},
}

truth_pattern = re.compile(r"era5_pl_(\d{4}-\d{2}-\d{2})_(\d{2})\.grib")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Helpers ----------------
def load_grib_uvqtz(filepath):
    try:
        return xr.open_dataset(
            filepath,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs={
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "shortName": ["u", "v", "q", "t", "z"],
                },
                "indexpath": "",
            },
        )
    except Exception as e:
        print(f"⚠️ Could not open {filepath}: {e}")
        return xr.Dataset()

def ensure_descending_latitude(ds):
    if ds.latitude[0] < ds.latitude[-1]:
        return ds.sel(latitude=ds.latitude[::-1])
    return ds

def compute_wind_rmse(u_t, v_t, u_f, v_f):
    rmse_u = np.sqrt(np.nanmean((u_t - u_f) ** 2))
    rmse_v = np.sqrt(np.nanmean((v_t - v_f) ** 2))
    return rmse_u, rmse_v, np.sqrt(rmse_u**2 + rmse_v**2)

def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f - t) ** 2))

# ---------------- RMSE per file ----------------
def compute_rmse_for_file(truth_file):
    m = truth_pattern.match(os.path.basename(truth_file))
    if not m:
        return None

    date_str, hour_str = m.groups()
    truth_time = pd.to_datetime(f"{date_str} {hour_str}:00")

    ds_t = load_grib_uvqtz(truth_file)
    if not ds_t:
        return None

    ds_t = ensure_descending_latitude(ds_t)
    ds_t = ds_t.sel(
        latitude=slice(DOMAIN["lat_max"], DOMAIN["lat_min"]),
        longitude=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]),
    )

    results = {v: {} for v in PRESSURE_LEVELS}

    for lead in lead_times:
        fc_time = truth_time - pd.Timedelta(hours=lead)
        fc_name = f"aurora_forecast_{fc_time:%Y-%m-%d}_{fc_time:%H}-out-{lead}.nc"
        fc_path = os.path.join(forecast_path, fc_name)

        if not os.path.exists(fc_path):
            continue

        ds_f = xr.open_dataset(fc_path)
        ds_f = ensure_descending_latitude(ds_f)
        ds_f = ds_f.sel(
            latitude=slice(DOMAIN["lat_max"], DOMAIN["lat_min"]),
            longitude=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]),
        )

        try:
            u_t = ds_t.u.sel(isobaricInhPa=850).values
            v_t = ds_t.v.sel(isobaricInhPa=850).values
            u_f = ds_f.u.sel(isobaricInhPa=850).values
            v_f = ds_f.v.sel(isobaricInhPa=850).values

            ru, rv, rw = compute_wind_rmse(u_t, v_t, u_f, v_f)
            results["u"][lead] = ru
            results["v"][lead] = rv
            results["wind"][lead] = rw

            results["t"][lead] = compute_scalar_rmse(
                ds_f.t.sel(isobaricInhPa=850).values,
                ds_t.t.sel(isobaricInhPa=850).values,
            )
            results["q"][lead] = compute_scalar_rmse(
                ds_f.q.sel(isobaricInhPa=700).values,
                ds_t.q.sel(isobaricInhPa=700).values,
            )
            results["z"][lead] = compute_scalar_rmse(
                ds_f.z.sel(isobaricInhPa=500).values,
                ds_t.z.sel(isobaricInhPa=500).values,
            )

        except Exception:
            for v in results:
                results[v][lead] = np.nan

        ds_f.close()

    ds_t.close()
    return results

# ---------------- Seasonal runner ----------------
def run_monsoon(label, months):
    print(f"\n=== Running {label} ===")

    truth_files = []
    for f in os.listdir(truth_path):
        m = truth_pattern.match(f)
        if not m:
            continue
        month = int(m.group(1).split("-")[1])
        if month in months:
            truth_files.append(os.path.join(truth_path, f))

    rmse_store = {
        v: {PRESSURE_LEVELS[v]: {lt: [] for lt in lead_times}}
        for v in PRESSURE_LEVELS
    }

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(compute_rmse_for_file, tf) for tf in truth_files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
            data = fut.result()
            if not data:
                continue
            for v in data:
                for lt, val in data[v].items():
                    rmse_store[v][PRESSURE_LEVELS[v]][lt].append(val)

    mean_rmse = {
        v: {
            lvl: {
                lt: np.nanmean(rmse_store[v][lvl][lt])
                for lt in lead_times
            }
            for lvl in rmse_store[v]
        }
        for v in rmse_store
    }

    np.savez(
        os.path.join(output_dir, f"era5_vs_auro_rmse_{label}.npz"),
        rmse=mean_rmse,
        lead_times=np.array(lead_times),
    )

    for v, title in [
        ("wind", "850 hPa Wind Vector"),
        ("u", "850 hPa U-component"),
        ("v", "850 hPa V-component"),
        ("q", "700 hPa Specific Humidity"),
        ("t", "850 hPa Temperature"),
        ("z", "500 hPa Geopotential"),
    ]:
        plt.figure(figsize=(10, 6))
        lvl = PRESSURE_LEVELS[v]
        plt.plot(
            lead_times,
            [mean_rmse[v][lvl][lt] for lt in lead_times],
            marker="o",
            label=f"{lvl} hPa",
        )
        plt.title(f"ERA5 vs Aurora RMSE — {title} ({label})")
        plt.xlabel("Lead Time [hours]")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(output_dir, f"era5_vs_auro_rmse_{v}_{label}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print(f"=== Finished {label} ===")

# ---------------- Main ----------------
def main():
    for label, months in MONSOONS.items():
        run_monsoon(label, months)

if __name__ == "__main__":
    main()