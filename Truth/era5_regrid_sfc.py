#!/usr/bin/env python3

# era5 surface data is in 0.1 degrees so it must be regrided 
# when evaualting against 0.25 degrees resolution models such as AIFS
import subprocess
import time
from pathlib import Path
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Paths ---
base_dir = Path("/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data") # change as needed since aifs and aurora might be ran in different supecomputers
input_dir = base_dir / "era5_sfc_splits_6hr"
output_dir = base_dir / "era5_sfc_splits_6hr_regridded"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Settings ---
target_grid = "r1440x721"  # 0.25° grid
log_path = base_dir / "regrid_sfc_log.txt"
max_workers = 16

# --- File discovery ---
grib_files = sorted(glob.glob(str(input_dir / "*.grib*")))
print(f"[INFO] Found {len(grib_files)} files to regrid")

# --- Helper: log to both console and file ---
def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

# --- Worker function ---
def regrid_file(f):
    f_path = Path(f)
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})", f_path.stem)
    if not match:
        return ("skip", f"[WARN] Cannot parse date/hour from {f_path.name}, skipping.")
    
    date_str, hour_str = match.groups()
    out_file = output_dir / f"era5_sfc_{date_str}_{hour_str}_regridded.grib"

    # Skip if already exists
    if out_file.exists() and out_file.stat().st_size > 0:
        return ("skip", f"[SKIP] {out_file.name} already exists.")

    try:
        result = subprocess.run(
            ["cdo", f"remapcon,{target_grid}", str(f_path), str(out_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return ("ok", f"[OK] {out_file.name} written successfully.")
    except subprocess.CalledProcessError as e:
        if out_file.exists():
            out_file.unlink(missing_ok=True)
        return ("fail", f"[ERR] Failed on {f_path.name}\n       CDO output:\n{e.stderr.strip()}")

# --- Main parallel section ---
start_time = time.time()
n_done = n_skipped = n_failed = 0

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(regrid_file, f): f for f in grib_files}
    for future in as_completed(futures):
        status, message = future.result()
        log(message)
        if status == "ok":
            n_done += 1
        elif status == "skip":
            n_skipped += 1
        else:
            n_failed += 1

end_time = time.time()

# --- Summary ---
log("\n========== SUMMARY ==========")
log(f"Completed: {n_done}")
log(f"Skipped:   {n_skipped}")
log(f"Failed:    {n_failed}")
log(f"Total time: {(end_time - start_time)/60:.1f} min")
log("=============================\n")

print(f"[INFO] Regridding finished. Log saved to {log_path}")
