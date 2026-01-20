#!/usr/bin/env python3

import subprocess
from pathlib import Path

# --- Directories ---
base_dir = Path("/home/project/17001770/weather_department/nwp/zach/aurora_folder")
sfc_splits = base_dir / "aurora_sfc_splits_6hr"
pl_splits  = base_dir / "aurora_pl_splits_6hr"
merged_dir = base_dir / "aurora_merged"
merged_dir.mkdir(parents=True, exist_ok=True)

print("Scanning for SFC and PL files...")

# ---------------------------------------------
# Find all SFC 6-hour files
# aurora_sfc_YYYY-MM-DD_HH.grib
# ---------------------------------------------
sfc_files = sorted(sfc_splits.glob("aurora_sfc_*.grib"))
print(f"Found {len(sfc_files)} SFC files.")

merged_count = 0

for sfc_file in sfc_files:
    # Extract timestamp part: YYYY-MM-DD_HH.grib
    timestamp = sfc_file.name.replace("aurora_sfc_", "")

    # Matching PL file
    pl_file = pl_splits / f"aurora_pl_{timestamp}"

    # Output merged file
    merged_file = merged_dir / f"aurora_merged_{timestamp}"

    # Skip if already merged
    if merged_file.exists():
        print(f"[SKIP] Already merged: {merged_file.name}")
        continue

    # Skip if PL missing
    if not pl_file.exists():
        print(f"[WARN] Missing PL file for {timestamp}, skipping")
        continue

    print(f"[MERGE] {sfc_file.name} + {pl_file.name} → {merged_file.name}")
    subprocess.run(
        ["cdo", "merge", str(sfc_file), str(pl_file), str(merged_file)],
        check=True
    )

    merged_count += 1

print(f"\nDone! Total merged files created: {merged_count}")
print(f"Merged files stored in: {merged_dir}")
