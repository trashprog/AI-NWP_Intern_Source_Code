import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Paths and configuration ---
truth_path = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/gpm/gpm_data/accumulated'
forecast_path = '/home/project/17001770/ccrs_dwr/nwp/zach/aifs_folder/aifs_output_regridded'
output_dir = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/tp_plots'

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]

os.makedirs(output_dir, exist_ok=True)
npz_path = os.path.join(output_dir, 'rmse_results_aifs_gpm.npz')
plot_path = os.path.join(output_dir, 'rmse_vs_leadtime_aifs_gpm.png')

truth_pattern = re.compile(r"gpm_acc_(\d{4}-\d{2}-\d{2})_(\d{2})\.nc")
forecast_pattern = re.compile(r"aifs_gridded_(\d{4}-\d{2}-\d{2})_(\d{2})-out-(\d+)_regridded\.nc")

# --- Helper function ---
def compute_rmse(fc, obs):
    mask = np.isfinite(fc) & np.isfinite(obs)
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((fc[mask] - obs[mask]) ** 2))

def process_truth_file(truth_file):
    match = truth_pattern.match(truth_file)
    if not match:
        return None
    truth_date_str, truth_hour_str = match.groups()
    truth_valid_time = pd.to_datetime(f"{truth_date_str} {truth_hour_str}:00")

    truth_ds = xr.open_dataset(os.path.join(truth_path, truth_file),
                               decode_timedelta=False).sel(
                            lat=slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
                            lon=slice(DOMAIN["lon_min"], DOMAIN["lon_max"])
                        )

    lat_truth = truth_ds['lat'].values
    lon_truth = truth_ds['lon'].values

    rmse_results = {}

    for lead in lead_times:
        init_time = truth_valid_time - pd.Timedelta(hours=lead)
        init_date_str = init_time.strftime("%Y-%m-%d")
        init_hour_str = init_time.strftime("%H")
        fc_file_name = f"aifs_gridded_{init_date_str}_{init_hour_str}-out-{lead}_regridded.nc"
        fc_path = os.path.join(forecast_path, fc_file_name)

        if not os.path.exists(fc_path):
            continue

        try:
            fc_ds = xr.open_dataset(fc_path, decode_timedelta=False).sel(
                        latitude=slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
                        longitude=slice(DOMAIN["lon_min"], DOMAIN["lon_max"])
                    )

            # Align longitudes to 0-360
            fc_ds = fc_ds.assign_coords(longitude=(fc_ds['longitude'].values % 360))
            lat_fc = fc_ds['latitude'].values
            lon_fc = fc_ds['longitude'].values

            # Skip if grids don't match
            if not (np.array_equal(lat_truth, lat_fc) and np.array_equal(lon_truth, lon_fc)):
                return None

            rmse_value = compute_rmse(fc_ds['tp'].squeeze().values * 1000,
                                      truth_ds['precipitation'].values)
            rmse_results.setdefault(lead, []).append(rmse_value)
        except Exception as e:
            print(f"⚠️ Error in {fc_file_name}: {e}")
            continue

    return rmse_results

# --- Main execution ---
if __name__ == "__main__":
    all_truth = sorted([f for f in os.listdir(truth_path) if truth_pattern.match(f)])
    rmse_dict_align = {}

    print(f"🔹 Starting RMSE computation for {len(all_truth)} truth files using 16 workers...")

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_truth_file, f): f for f in all_truth}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing truth files"):
            res = future.result()
            if res is None:
                continue
            for lead, values in res.items():
                rmse_dict_align.setdefault(lead, []).extend(values)

    # --- Save results ---
    np.savez(npz_path, **{str(k): v for k, v in rmse_dict_align.items()})
    print(f"✅ RMSE computation complete. Results saved to: {npz_path}")

    # --- Plot mean RMSE per lead time ---
    mean_rmse = []
    for lead in lead_times:
        vals = np.array(rmse_dict_align.get(lead, []))
        vals = vals[np.isfinite(vals)]
        mean_rmse.append(np.nan if len(vals)==0 else np.mean(vals))

    plt.figure(figsize=(8, 5))
    plt.plot(lead_times, mean_rmse, 'o-', lw=2)
    plt.grid(alpha=0.3)
    plt.xlabel('Forecast Lead Time (hours)')
    plt.ylabel('RMSE (mm)')
    plt.title('AIFS vs GPM Precipitation RMSE SEA Region')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"📊 Plot saved to: {plot_path}")
