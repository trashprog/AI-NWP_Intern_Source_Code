#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path
import glob

def split_grib_6hour(
    input_grib: Path,
    output_dir: Path,
    prefix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Splitting {input_grib.name} into daily files...")
    subprocess.run(
        ["cdo", "splitdate", str(input_grib), str(output_dir / "date_")],
        check=True,
    )

    date_files = sorted(output_dir.glob("date_*.grb*"))
    print(f"[INFO] Found {len(date_files)} daily files")

    for date_file in date_files:
        date_name = date_file.stem.replace("date_", "")
        print(f"[INFO] Processing {date_name}...")

        for hour in (0, 6, 12, 18):
            out_file = output_dir / f"{prefix}_{date_name}_{hour:02d}.grib"
            subprocess.run(
                ["cdo", f"selhour,{hour}", str(date_file), str(out_file)],
                check=True,
            )


if __name__ == "__main__":
    base_dir = Path("/home/project/17001770/weather_department/nwp/zach/era5_folder/era5_data")

    start_time = time.time()

    print("starting ERA5 splitting...")

    # surface
    split_grib_6hour(
        input_grib=base_dir / "era5_sfc.grib",
        output_dir=base_dir / "era5_sfc_splits_6hr",
        prefix="era5_sfc",
    )

    # pressure levels
    split_grib_6hour(
        input_grib=base_dir / "era5_pl_final.grib",
        output_dir=base_dir / "era5_pl_splits_6hr_final",
        prefix="era5_pl",
    )

    end_time = time.time()
    print(f"All splitting completed in {end_time - start_time:.2f} seconds.")
