#!/usr/bin/env python3

import time
from pathlib import Path
import os
import cfgrib
import numpy as np
from collections import defaultdict
import datetime
import re
import earthkit.regrid as ekr

# Define directories please change them to your directory
base_dir = Path("/home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder")
sfc_splits = base_dir / "aifs_sfc_splits_6hr"
pl_splits = base_dir / "aifs_pl_splits_6hr"
soil_splits = base_dir / "aifs_soil_splits_6hr"
merged_dir = base_dir / "aifs_merged_splits"
processed_inputs_dir = base_dir / "aifs_processed_inputs"

# testing output 0 lead time
output_nc_dir = base_dir / "aifs_output"
output_nc_truth_dir = base_dir / "aifs_output_truth"

# Create output directory if missing
merged_dir.mkdir(parents=True, exist_ok=True)
processed_inputs_dir.mkdir(parents=True, exist_ok=True)   

# convert to npz
start_time = time.time()
print("now beginning the conversion to npz...")

def convert_gribs_to_npz(data_dir=merged_dir, output_dir=processed_inputs_dir, output_truth_dir=output_nc_truth_dir):
    """
    Iterates through GRIB files in `data_dir` in pairs (t-6h, t0),
    regrids, stacks, and saves them as .npz files in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    grib_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.grib')])

    for i in range(1, len(grib_files)):
        fname_prev = os.path.join(data_dir, grib_files[i-1])
        fname_curr = os.path.join(data_dir, grib_files[i])

        # Build forecast date from current file
        forecast_date_str = re.search(r'aifs_merged_(\d{4}-\d{2}-\d{2}_\d{2})\.grib', grib_files[i]).group(1)
        forecast_date = datetime.datetime.strptime(forecast_date_str, "%Y-%m-%d_%H")
        output_npz = os.path.join(output_dir, f"input_state_{forecast_date.strftime('%Y-%m-%d_%H%M')}.npz")
        output_npz_truth = os.path.join(output_truth_dir, f"input_state_truth_{forecast_date.strftime('%Y-%m-%d_%H%M')}.npz")

        fields = defaultdict(list)
        lat = lon = None

        for fname in [fname_prev, fname_curr]:
            try:
                datasets = cfgrib.open_datasets(fname, decode_timedelta=True)
                for ds in datasets:


                    # extract lat lon
                    if lat is None and "latitude" in ds and "longitude" in ds:
                        lat = ds["latitude"].values
                        lon = ds["longitude"].values


                    # Detect vertical dimension
                    for lev_dim in ["level", "isobaricInhPa", "depthBelowLand", "hybrid"]:
                        if lev_dim in ds.dims:
                            level_name = lev_dim
                            break
                    else:
                        level_name = None

                    for var in ds.data_vars:
                        data = ds[var].values

                        # Vertical levels
                        if data.ndim == 3 and level_name:
                            for idx, lev in enumerate(ds[level_name].values):
                                slice2d = data[idx]
                                assert slice2d.shape == (721, 1440)
                                # Roll longitude
                                slice2d = np.roll(slice2d, -slice2d.shape[1]//2, axis=1)
                                slice2d = ekr.interpolate(slice2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                                fields[f"{var}_{int(lev)}"].append(slice2d)
                        else:
                            # 2D surface variable
                            assert data.shape == (721, 1440)
                            slice2d = np.roll(data, -data.shape[1]//2, axis=1)
                            slice2d = ekr.interpolate(slice2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                            fields[var].append(slice2d)
            except Exception as e:
                print(f"Failed to process file '{fname}': {e}")

        # Stack last two time slices
        for param_name, vals in fields.items():
            fields[param_name] = np.stack(vals, axis=0)  # shape (2, N)

        # Save as .npz
        input_state = {
            "date": forecast_date,
            "fields": dict(fields)
        }

        # save a new npz with the coords
        input_state_truth = {
            "date": forecast_date,
            "fields": dict(fields),
            "latitudes": lat,
            "longitudes": lon
        }


        np.savez(output_npz, **input_state)
        np.savez(output_npz_truth, **input_state_truth)
        print(f"Saved {output_npz}")

        # now convert to .nc
        

    print("All GRIB pairs converted to .npz successfully.")

# run func
convert_gribs_to_npz()

end_time = time.time()
print(f"Merging completed in {end_time - start_time:.2f} seconds.")
