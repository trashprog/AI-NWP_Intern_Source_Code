#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path
import glob


# =====================================================
# Helpers
# =====================================================
def run(cmd):
    subprocess.run(cmd, check=True)


# =====================================================
# SURFACE (SFC) SPLITTING
# =====================================================
def split_surface():
    print("\n=== STARTING SURFACE SPLITTING ===")

    base_dir = Path("/home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder")
    surface_data = base_dir / "aifs_sfc.grib"
    output_dir = base_dir / "aifs_sfc_splits_6hr"

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print(f"[INFO] Splitting {surface_data.name} into daily files...")
    run(["cdo", "splitdate", str(surface_data), str(output_dir / "date_")])

    date_files = sorted(output_dir.glob("date_*.grb*"))
    print(f"[INFO] Found {len(date_files)} daily files")

    for date_file in date_files:
        date_name = date_file.stem.replace("date_", "")
        print(f"[INFO] Processing {date_name}")

        for hour in (0, 6, 12, 18):
            out_file = output_dir / f"aifs_sfc_{date_name}_{hour:02d}.grib"
            if out_file.exists():
                continue

            run(["cdo", f"selhour,{hour}", str(date_file), str(out_file)])

    print(f"[DONE] Surface splitting completed in {time.time() - t0:.2f} seconds")
    print(f"[INFO] Surface outputs in: {output_dir}")


# =====================================================
# SOIL SPLITTING
# =====================================================
def split_soil():
    print("\n=== STARTING SOIL SPLITTING ===")

    base_dir = Path("/home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder")
    soil_data = base_dir / "aifs_soil.grib"
    output_dir = base_dir / "aifs_soil_splits_6hr"

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print(f"[INFO] Splitting {soil_data.name} into daily files...")
    run(["cdo", "splitdate", str(soil_data), str(output_dir / "date_")])

    date_files = sorted(output_dir.glob("date_*.grb*"))
    print(f"[INFO] Found {len(date_files)} daily files")

    for date_file in date_files:
        date_name = date_file.stem.replace("date_", "")
        print(f"[INFO] Processing {date_name}")

        for hour in (0, 6, 12, 18):
            out_file = output_dir / f"aifs_soil_{date_name}_{hour:02d}.grib"
            if out_file.exists():
                continue

            run(["cdo", f"selhour,{hour}", str(date_file), str(out_file)])

    print(f"[DONE] Soil splitting completed in {time.time() - t0:.2f} seconds")
    print(f"[INFO] Soil outputs in: {output_dir}")


# =====================================================
# PRESSURE LEVEL (PL) SPLITTING
# =====================================================
def split_pressure():
    print("\n=== STARTING PRESSURE LEVEL SPLITTING ===")

    base_dir = Path("/home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder")
    output_dir = base_dir / "aifs_pl_splits_6hr"

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    pressure_files = sorted(base_dir.glob("aifs_pl_*.grib"))
    print(f"[INFO] Found {len(pressure_files)} pressure-level files")

    for pressure_path in pressure_files:
        print(f"\n[INFO] Processing {pressure_path.name}")

        daily_dir = output_dir / f"temp_daily_{pressure_path.stem}"
        daily_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------
        # Step 1: split into daily files
        # -------------------------------
        if list(daily_dir.glob("date_*.grb*")):
            print("[SKIP] Daily split already exists")
        else:
            print("[RUN] Splitting into daily date files")
            run(["cdo", "splitdate", str(pressure_path), str(daily_dir / "date_")])

        # -------------------------------
        # Step 2: daily → 6-hour files
        # -------------------------------
        date_files = sorted(daily_dir.glob("date_*.grb*"))
        print(f"[INFO] Found {len(date_files)} daily files")

        for date_file in date_files:
            date_name = date_file.stem.replace("date_", "")
            print(f"[INFO] Processing {date_name}")

            for hour in (0, 6, 12, 18):
                out_file = output_dir / f"aifs_pl_{date_name}_{hour:02d}.grib"
                if out_file.exists():
                    continue

                run(["cdo", f"selhour,{hour}", str(date_file), str(out_file)])

    print(f"[DONE] Pressure splitting completed in {time.time() - t0:.2f} seconds")
    print(f"[INFO] Pressure outputs in: {output_dir}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    overall_start = time.time()

    split_surface()
    split_soil()
    split_pressure()

    print(f"\nALL SPLITTING COMPLETED IN {time.time() - overall_start:.2f} seconds")
