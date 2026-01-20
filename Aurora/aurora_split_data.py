#!/usr/bin/env python3

import subprocess
import glob
import time
from pathlib import Path
from datetime import datetime

# =====================================================
# Helpers
# =====================================================
def run(cmd):
    """Run a command and fail loudly."""
    subprocess.run(cmd, check=True)


def parse_datetime_from_name(path: Path):
    """
    aurora_*_YYYY-MM-DD_HH.grib -> datetime
    """
    _, _, date, hour = path.stem.split("_")
    return datetime.strptime(f"{date}_{hour}", "%Y-%m-%d_%H")


# =====================================================
# SFC PIPELINE
# =====================================================
def process_sfc():
    print("\n=== SFC SPLIT + MERGE PIPELINE ===")

    base_dir = Path(
        "/home/project/77010001/weather_department/nwp/intern_sharing/aurora_inputs/sfc" # change path as needed
    )
    input_pattern = base_dir / "aurora_sfc_month_*_highres.grib"

    out_dir = Path(
        "/home/project/77010001/weather_department/nwp/zach/aurora_folder" # change path as needed
    )

    split_dir = out_dir / "aurora_sfc_split_individual"
    merged_dir = out_dir / "aurora_sfc_splits_6hr"

    split_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # -------------------------------
    # PHASE 1: MONTH → DAY → HOUR
    # -------------------------------
    for month_file in sorted(glob.glob(str(input_pattern))):
        month_path = Path(month_file)
        print(f"\n[INFO] Processing {month_path.name}")

        daily_dir = split_dir / f"temp_daily_{month_path.stem}"
        daily_dir.mkdir(parents=True, exist_ok=True)

        # Month → daily
        if list(daily_dir.glob("date_*.grb*")):
            print("[SKIP] Daily split exists")
        else:
            print("[RUN] Monthly → daily")
            run(["cdo", "splitdate", str(month_path), str(daily_dir / "date_")])

        # Daily → hourly
        for daily_file in sorted(daily_dir.glob("date_*.grb*")):
            date_str = daily_file.stem.replace("date_", "")
            hour_dir = split_dir / date_str
            hour_dir.mkdir(parents=True, exist_ok=True)

            existing = list(hour_dir.glob(f"aurora_sfc_{date_str}_??.grib"))
            if len(existing) == 24:
                print(f"[SKIP] Hourly files exist for {date_str}")
                continue

            print(f"[RUN] {date_str} → hourly")
            run(["cdo", "splithour", str(daily_file), str(hour_dir / "h_")])

            for tmp in hour_dir.glob("h_*.grb*"):
                hour = tmp.stem.split("_")[1]
                out = hour_dir / f"aurora_sfc_{date_str}_{hour}.grib"
                if out.exists():
                    tmp.unlink()
                else:
                    out.write_bytes(tmp.read_bytes())
                    tmp.unlink()

    print(f"[INFO] SFC splitting done in {time.time() - t0:.1f}s")

    # -------------------------------
    # PHASE 2: MERGE CONSECUTIVE HOURS
    # -------------------------------
    hourly_files = sorted(
        split_dir.glob("*/aurora_sfc_*.grib"),
        key=parse_datetime_from_name
    )

    print(f"[INFO] Found {len(hourly_files)} hourly SFC files")

    for prev, curr in zip(hourly_files[:-1], hourly_files[1:]):
        out = merged_dir / curr.name
        if out.exists():
            continue

        print(f"[MERGE] {prev.name} + {curr.name}")
        run(["cdo", "mergetime", str(prev), str(curr), str(out)])

    print(f"[DONE] SFC pipeline finished in {time.time() - t0:.1f}s")


# =====================================================
# PL PIPELINE
# =====================================================
def process_pl():
    print("\n=== PL SPLIT + MERGE PIPELINE ===")

    base_dir = Path(
        "/home/project/77010001/weather_department/nwp/intern_sharing/aurora_inputs/pl" # change path as needed
    )
    input_pattern = base_dir / "aurora_pl_week_*_highres.grib"

    out_dir = Path(
        "/home/project/77010001/weather_department/nwp/zach/aurora_folder" # change path as needed
    )

    split_dir = out_dir / "aurora_pl_split_individual"
    merged_dir = out_dir / "aurora_pl_splits_6hr"

    split_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    MIN_WEEK_SIZE = int((21.98 - 0.05) * 1024**3)

    t0 = time.time()

    # -------------------------------
    # PHASE 1: WEEK → DAY → 6H STEP
    # -------------------------------
    for week_file in sorted(glob.glob(str(input_pattern))):
        week_path = Path(week_file)
        size = week_path.stat().st_size
        week_num = int(week_path.stem.split("_")[3])

        if week_num != 53 and size < MIN_WEEK_SIZE:
            print(f"[SKIP] {week_path.name} too small")
            continue

        print(f"\n[INFO] Processing {week_path.name}")

        daily_dir = split_dir / f"temp_daily_{week_path.stem}"
        daily_dir.mkdir(parents=True, exist_ok=True)

        if not list(daily_dir.glob("date_*.grb*")):
            print("[RUN] Weekly → daily")
            run(["cdo", "splitdate", str(week_path), str(daily_dir / "date_")])

        for daily_file in sorted(daily_dir.glob("date_*.grb*")):
            step_dir = daily_dir / daily_file.stem
            step_dir.mkdir(parents=True, exist_ok=True)

            if list(step_dir.glob("aurora_pl_*.grib")):
                continue

            print(f"[RUN] {daily_file.name} → 6h steps")
            run(["cdo", "splitdatetime", str(daily_file), str(step_dir / "step_")])

            for step in step_dir.glob("step_*.grb*"):
                ts = step.stem.replace("step_", "")
                date, time_ = ts.split("T")
                hour = time_[:2]
                step.rename(step_dir / f"aurora_pl_{date}_{hour}.grib")

    print(f"[INFO] PL splitting done in {time.time() - t0:.1f}s")

    # -------------------------------
    # PHASE 2: MERGE CONSECUTIVE STEPS
    # -------------------------------
    steps = sorted(
        split_dir.glob("*/**/aurora_pl_*.grib"),
        key=parse_datetime_from_name
    )

    print(f"[INFO] Found {len(steps)} PL step files")

    for prev, curr in zip(steps[:-1], steps[1:]):
        out = merged_dir / curr.name
        if out.exists():
            continue

        print(f"[MERGE] {prev.name} + {curr.name}")
        run(["cdo", "mergetime", str(prev), str(curr), str(out)])

    print(f"[DONE] PL pipeline finished in {time.time() - t0:.1f}s")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    overall_start = time.time()

    process_sfc()
    process_pl()

    print(f"\nALL PIPELINES COMPLETED IN {time.time() - overall_start:.1f}s")
