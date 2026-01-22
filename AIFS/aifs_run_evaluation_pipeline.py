#!/usr/bin/env python3
import sys
import os
import re
import datetime
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

# === CONFIG ===
print("STEP 1 running inference")
checkpoint = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/AIFS/aifs-single-mse-1.0.ckpt"
input_dir = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/AIFS/aifs_input_files"
output_dir = "/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/AIFS/aifs_forecast_files"

grid_res = 0.25  # degrees
lat_full = np.arange(-90, 90 + grid_res, grid_res)
lon_full = np.arange(0, 360, grid_res)
grid_lon, grid_lat = np.meshgrid(lon_full, lat_full)

pattern = re.compile(r"([a-zA-Z]+)_([0-9]+)")
suffix_pattern = re.compile(r"^(?P<base>.+)\.0$")
special_renames = {'t2m': '2t', 'd2m': '2d', 'u10': '10u', 'v10': '10v'}

unit_map = {
    "u": "m s-1", "v": "m s-1", "w": "m s-1",
    "10u": "m s-1", "10v": "m s-1", "100u": "m s-1", "100v": "m s-1",
    "t": "K", "2t": "K", "z": "m",
    "tp": "m", "cp": "m", "sf": "m", "ro": "m",
    "msl": "Pa", "sp": "Pa"
}

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = '16'

# === GPU slicing ===
gpu_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# List all NPZ files
all_files = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith(".npz") and f.startswith("input_state_")
])

# Split into 4 chunks (or adjust if needed)
chunk_size = (len(all_files) + 3) // 4
start = gpu_index * chunk_size
end = start + chunk_size
files = all_files[start:end]

print(f"GPU {gpu_index} processing {len(files)} files...", flush=True)

# === Helper function ===
def fix_var_names(fields):
    renamed = {}
    for k, v in fields.items():
        if k in special_renames:
            renamed[special_renames[k]] = v
            continue
        m = suffix_pattern.match(k)
        if m:
            renamed[m.group('base')] = v
        else:
            renamed[k] = v
    return renamed

# === Process single file ===
def process_file(npz_path):
    try:
        print(f"Processing {npz_path} ...", flush=True)
        loaded = np.load(npz_path, allow_pickle=True)
        input_state = {k: loaded[k] for k in loaded.files}

        if isinstance(input_state['fields'], np.ndarray):
            input_state['fields'] = input_state['fields'].item()

        input_state['fields'] = fix_var_names(input_state['fields'])

        dt = input_state['date']
        if isinstance(dt, np.ndarray):
            dt = dt.item()
        if isinstance(dt, np.datetime64):
            dt = dt.astype('M8[ms]').astype(datetime.datetime)
        elif isinstance(dt, str):
            dt = datetime.datetime.fromisoformat(dt)
        input_state['date'] = dt
        initial_dt = dt

        runner = SimpleRunner(checkpoint, device="cuda")

        for state in runner.run(input_state=input_state, lead_time=48):
            print_state(state)

            scat_points = np.column_stack((state["longitudes"], state["latitudes"]))
            fields = state["fields"]

            groups, surface_keys = {}, []
            for k in fields.keys():
                m = pattern.match(k)
                if m:
                    var, level = m.groups()
                    groups.setdefault(var, []).append(int(level))
                else:
                    surface_keys.append(k)

            ds = xr.Dataset(coords={"latitude": lat_full, "longitude": lon_full})
            if groups:
                levels = sorted({lvl for lvls in groups.values() for lvl in lvls})
                ds["level"] = ("level", np.array(levels, dtype=np.int32))

            tri = Delaunay(scat_points)

            # 3D variables
            for var, lvls in groups.items():
                arr = np.empty((len(lvls), len(lat_full), len(lon_full)), dtype=np.float32)
                for i, lvl in enumerate(sorted(lvls)):
                    key = f"{var}_{lvl}"
                    interp = LinearNDInterpolator(tri, fields[key])
                    arr[i] = interp(grid_lon, grid_lat)
                ds[var] = (("level", "latitude", "longitude"), arr)

            # Surface variables
            for key in surface_keys:
                interp = LinearNDInterpolator(tri, fields[key])
                ds[key] = (("latitude", "longitude"), interp(grid_lon, grid_lat))

            ds = ds.expand_dims({"time": [np.datetime64(str(state["date"]))]})

            for var in ds.data_vars:
                m = re.match(r"([a-zA-Z]+)", var)
                base = m.group(1) if m else var
                ds[var].attrs["units"] = unit_map.get(base, "unknown")

            lead_hours = int((state["date"] - initial_dt).total_seconds() / 3600)
            out_nc = os.path.join(
                output_dir,
                f"aifs_gridded_{initial_dt.strftime('%Y-%m-%d_%H')}-out-{lead_hours}.nc"
            )
            ds.to_netcdf(out_nc)
            print(f"Saved: {out_nc}", flush=True)

    except Exception as e:
        print(f"Error processing {npz_path}: {e}", flush=True)

# === Main loop ===
for f in files:
    process_file(f)


print("STEP 2 processing outputs")

# --- Paths ---
FORECAST_PATH = '/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/AIFS/aifs_forecast_files'
OUTPUT_FORECAST_PATH = '/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/AIFS/aifs_forecast_regridded_files'

# Create output directory
os.makedirs(OUTPUT_FORECAST_PATH, exist_ok=True)

# --- Helper function ---
def fix_grid(ds):
    """Ensure lat descending and longitude -180..180, then sort."""
    if ds.latitude.values[0] < ds.latitude.values[-1]:
        ds = ds.reindex(latitude=list(reversed(ds.latitude)))
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    return ds.sortby("latitude").sortby("longitude")

def process_forecast_file(f_in):
    """Process a single forecast NetCDF: fix grid and save, overwrite existing files."""
    f_out = os.path.join(OUTPUT_FORECAST_PATH, os.path.basename(f_in).replace(".nc","_regridded.nc"))
    try:
        ds = xr.load_dataset(f_in, engine="netcdf4")
        ds = fix_grid(ds)
        ds.to_netcdf(f_out, mode="w")  # overwrite
        ds.close()
        return f_out, "done"
    except Exception as e:
        return f_in, f"error: {e}"

# --- Main execution ---
if __name__=="__main__":
    print("Processing forecast NetCDF files...")
    forecast_files = [os.path.join(FORECAST_PATH, f) for f in os.listdir(FORECAST_PATH) if f.endswith(".nc")]
    results = []
    for f in forecast_files:
        results.append(process_forecast_file(f))

    # Save log
    with open("regrid_log.txt","w") as f:
        for fpath, status in results:
            f.write(f"{fpath}: {status}\n")

    print("All forecast files processed and regridded successfully.")