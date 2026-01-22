import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Config ---
truth_path = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/era5_2024to2025/era5_pl_splits_6hr_final'
forecast_path = '/home/project/17001770/ccrs_dwr/nwp/zach/aifs_folder/aifs_output_regridded'
PLOT_DIR = '/home/project/17001770/ccrs_dwr/nwp/zach/era5_reanalysis/regional_google_weatherbench_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]

# Target levels per variable
PRESSURE_LEVELS = {
    "wind": 850,
    "u": 850,
    "v": 850,
    "temperature": 850,
    "humidity": 700,
    "q": 500,
    "z": 500,
}

# Define monsoon periods
MONSOON_PERIODS = {
    "Inter-monsoon": {4, 5, 10, 11},
    "Southwest-monsoon": {6, 7, 8, 9},
    "Northeast-monsoon": {12, 1, 2, 3},
}

truth_pattern = re.compile(r"era5_pl_(\d{4}-\d{2}-\d{2})_(\d{2})\.grib")
WORKERS = 1

# --- RMSE helpers ---
def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f.ravel() - t.ravel())**2))

def compute_wind_rmse(u_t, v_t, u_f, v_f):
    rmse_u = np.sqrt(np.nanmean((u_t.ravel() - u_f.ravel())**2))
    rmse_v = np.sqrt(np.nanmean((v_t.ravel() - v_f.ravel())**2))
    rmse_wind = np.sqrt(rmse_u**2 + rmse_v**2)
    return rmse_u, rmse_v, rmse_wind

# --- Process one truth file ---
def process_truth_file(truth_file):
    vars_rmse = {var: {PRESSURE_LEVELS[var]: {lead: [] for lead in lead_times}} for var in PRESSURE_LEVELS}

    match = truth_pattern.match(truth_file)
    if not match:
        return vars_rmse, None
    date_str, hour_str = match.groups()
    truth_valid_time = pd.to_datetime(f"{date_str} {hour_str}:00")
    month = truth_valid_time.month

    try:
        truth_ds = xr.load_dataset(
            os.path.join(truth_path, truth_file),
            engine='cfgrib',
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}, "indexpath": ""}
        )
        truth_ds = truth_ds.sel(latitude=slice(DOMAIN['lat_max'], DOMAIN['lat_min']),
                                longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))
    except Exception as e:
        print(f"[ERROR] Loading truth {truth_file}: {e}")
        return vars_rmse, month

    for lead in lead_times:
        fc_valid_time = truth_valid_time - pd.Timedelta(hours=lead)
        fc_file_name = f"aifs_gridded_{fc_valid_time.strftime('%Y-%m-%d')}_{fc_valid_time.strftime('%H')}-out-{lead}_regridded.nc"
        fc_path = os.path.join(forecast_path, fc_file_name)
        if not os.path.exists(fc_path):
            continue

        try:
            fc_ds = xr.open_dataset(fc_path)
            fc_ds = fc_ds.sel(latitude=fc_ds['latitude'][::-1])
            fc_ds = fc_ds.sel(latitude=slice(DOMAIN['lat_max'], DOMAIN['lat_min']),
                              longitude=slice(DOMAIN['lon_min'], DOMAIN['lon_max']))

            if not (np.array_equal(truth_ds['latitude'].values, fc_ds['latitude'].values) and
                    np.array_equal(truth_ds['longitude'].values, fc_ds['longitude'].values)):
                continue

            # Wind vector
            u_t = truth_ds['u'].sel(isobaricInhPa=PRESSURE_LEVELS['u']).values
            v_t = truth_ds['v'].sel(isobaricInhPa=PRESSURE_LEVELS['v']).values
            u_f = fc_ds['u'].sel(level=PRESSURE_LEVELS['u']).squeeze().values
            v_f = fc_ds['v'].sel(level=PRESSURE_LEVELS['v']).squeeze().values
            rmse_u, rmse_v, rmse_wind = compute_wind_rmse(u_t, v_t, u_f, v_f)
            vars_rmse['u'][PRESSURE_LEVELS['u']][lead].append(rmse_u)
            vars_rmse['v'][PRESSURE_LEVELS['v']][lead].append(rmse_v)
            vars_rmse['wind'][PRESSURE_LEVELS['wind']][lead].append(rmse_wind)

            # Temperature 850
            t_t = truth_ds['t'].sel(isobaricInhPa=PRESSURE_LEVELS['temperature']).values
            t_f = fc_ds['t'].sel(level=PRESSURE_LEVELS['temperature']).squeeze().values
            vars_rmse['temperature'][PRESSURE_LEVELS['temperature']][lead].append(compute_scalar_rmse(t_f, t_t))

            # Humidity 700
            q_t = truth_ds['q'].sel(isobaricInhPa=PRESSURE_LEVELS['humidity']).values
            q_f = fc_ds['q'].sel(level=PRESSURE_LEVELS['humidity']).squeeze().values
            vars_rmse['humidity'][PRESSURE_LEVELS['humidity']][lead].append(compute_scalar_rmse(q_f, q_t))

            # Geopotential 500
            z_t = truth_ds['z'].sel(isobaricInhPa=PRESSURE_LEVELS['z']).values
            z_f = fc_ds['z'].sel(level=PRESSURE_LEVELS['z']).squeeze().values
            vars_rmse['z'][PRESSURE_LEVELS['z']][lead].append(compute_scalar_rmse(z_f, z_t))

        except Exception as e:
            continue

        fc_ds.close()

    truth_ds.close()
    return vars_rmse, month

# --- Main ---
all_truth_files = [f for f in os.listdir(truth_path) if truth_pattern.match(f) and f.endswith('.grib')]

for monsoon_name, months in MONSOON_PERIODS.items():
    print(f"Processing monsoon period: {monsoon_name}")
    rmse_dict = {var: {PRESSURE_LEVELS[var]: {lead: [] for lead in lead_times}} for var in PRESSURE_LEVELS}

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_truth_file, f): f for f in all_truth_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {monsoon_name}"):
            result, month = future.result()
            if month not in months:
                continue
            for var in rmse_dict:
                for level in rmse_dict[var]:
                    for lead in lead_times:
                        rmse_dict[var][level][lead].extend(result[var][level][lead])

    # Average per lead
    for var in rmse_dict:
        for level in rmse_dict[var]:
            for lead in lead_times:
                rmse_dict[var][level][lead] = np.nanmean(rmse_dict[var][level][lead])

    # Save
    np.savez(os.path.join(PLOT_DIR, f"era5_vs_aifs_rmse_{monsoon_name.replace(' ','_')}.npz"),
             lead_times=np.array(lead_times),
             pressure_levels=np.array(list(PRESSURE_LEVELS.values())),
             rmse_dict=rmse_dict)

    # Plot
    plot_titles = {
        'wind': 'Wind Vector (850 hPa)',
        'u': 'U-component (850 hPa)',
        'v': 'V-component (850 hPa)',
        'temperature': 'Temperature (850 hPa)',
        'humidity': 'Specific Humidity (700 hPa)',
        'z': 'Geopotential (500 hPa)',
    }

    for var, title in plot_titles.items():
        plt.figure(figsize=(10,6))
        level = PRESSURE_LEVELS[var]
        y_vals = [rmse_dict[var][level][lead] for lead in lead_times]
        plt.plot(lead_times, y_vals, marker='o')
        plt.title(f"ERA5 vs AIFS RMSE: {title} ({monsoon_name})")
        plt.xlabel("Lead Time [hours]")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"era5_vs_aifs_rmse_{var}_{monsoon_name.replace(' ','_')}.png"))
        plt.close()
