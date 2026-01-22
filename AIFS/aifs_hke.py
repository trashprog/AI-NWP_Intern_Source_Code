import os
from datetime import datetime
import numpy as np
import xarray as xr
from pathlib import Path
from numpy.fft import fftn, fftshift, fftfreq
from tqdm import tqdm

# ---------------------------------------------------
# Config
# ---------------------------------------------------
folder = Path("/home/project/17001770/ccrs_dwr/nwp/zach/aifs_folder/aifs_output_regridded/")
out_dir = Path("/home/project/17001770/ccrs_dwr/nwp/zach/research_paper/spectra_npzs/aifs/")
out_dir.mkdir(exist_ok=True)

LOG = out_dir / "missing_batches.log"
if LOG.exists():
    LOG.unlink()

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
PRESSURES = [850, 700, 200]

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def log_missing(init_str, missing_file):
    with LOG.open("a") as f:
        f.write(f"{init_str}: missing {missing_file}\n")


def parse_init(fname):
    core = fname.replace("_regridded.nc", "")
    if core.startswith("aifs_gridded_"):
        core = core.replace("aifs_gridded_", "")
    elif core.startswith("aifs_output_"):
        core = core.replace("aifs_output_", "")
    init_part, _ = core.split("-out-")
    dt = datetime.strptime(init_part, "%Y-%m-%d_%H")
    return dt, init_part


def compute_hke(ds, level):
    # squeeze to remove any singleton dims
    u = ds["u"].sel(level=level).squeeze().values
    v = ds["v"].sel(level=level).squeeze().values
    return 0.5 * (u**2 + v**2)


def hke_spectrum_3d(hke, ds):
    ds = ds.sortby('time')
    time_vals = ds.time.squeeze().values
    dt_seconds = (time_vals[1] - time_vals[0]) / np.timedelta64(1, 's')

    lat = ds.latitude.values
    lon = ds.longitude.values

    dy_deg = abs(lat[1] - lat[0])
    dx_deg = abs(lon[1] - lon[0])

    earth_r = 6371220.0
    meters_deg = 2 * np.pi * earth_r / 360.0
    dx = dx_deg * meters_deg
    dy = dy_deg * meters_deg

    H = fftn(hke)
    H_power = fftshift(np.abs(H)**2, axes=(0, 1, 2))

    nt, ny, nx = hke.shape
    freqs = fftshift(fftfreq(nt, dt_seconds))
    kx = fftshift(fftfreq(nx, dx))
    ky = fftshift(fftfreq(ny, dy))

    return H_power, freqs, kx, ky


def radial_average_spectrum(H_power, kx, ky):
    if max(kx) > max(ky):
        return H_power.mean(axis=2), ky
    return H_power.mean(axis=1), kx

# ---------------------------------------------------
# Collect files and group by init
# ---------------------------------------------------
all_files = [f for f in os.listdir(folder) if f.endswith(".nc")]

groups = {}
for f in all_files:
    dt, init_str = parse_init(f)
    groups.setdefault(init_str, []).append(f)

required_outs = [6, 12, 18, 24, 30, 36, 42, 48]

# ---------------------------------------------------
# Process batches
# ---------------------------------------------------
for init_str, files in sorted(groups.items()):
    # select one file per forecast hour
    matched = []
    for hr in required_outs:
        g = f"aifs_gridded_{init_str}-out-{hr}_regridded.nc"
        if g in files:
            matched.append(g)
        else:
            log_missing(init_str, f"missing out-{hr}")
            matched = None
            break

    if matched is None:
        continue

    print(f"VALID 48-hour AIFS forecast set for init {init_str}")
    for f in matched:
        print(f"    {f}")

    files_48h = [folder / f for f in matched]

    try:
        with xr.open_mfdataset(
            files_48h,
            combine="nested",
            concat_dim="time",
            parallel=True
        ) as ds:

            ds = (
                ds.sortby("latitude").sortby("longitude")
                .sel(latitude=slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
                    longitude=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]))
            )

            # Continue with HKE computation...

    except Exception as e:
        print(f"Skipping batch {init_str} due to error: {e}")
        log_missing(init_str, "corrupted or unreadable files")
        continue

    # compute batch start and end for naming
    # batch_start = init_str
    # batch_end_dt = datetime.strptime(init_str, "%Y-%m-%d_%H") + np.timedelta64(48, 'h')
    # batch_end = batch_end_dt.astype('M8[h]').astype(str).replace("T", "_")
    from datetime import timedelta
    batch_start = init_str
    batch_end_dt = datetime.strptime(init_str, "%Y-%m-%d_%H") + timedelta(hours=48)
    batch_end = batch_end_dt.strftime("%Y-%m-%d_%H")

    # Check if this batch is already fully processed
    existing = []
    for p in PRESSURES:
        outname = out_dir / f"HKE_aifs_{batch_start}_to_{batch_end}_{p}.npz"
        if outname.exists():
            existing.append(p)

    # If all levels already done, skip batch
    if len(existing) == len(PRESSURES):
        print(f"Skipping {init_str} — all pressure levels already processed.")
        continue

    # Pressure-level loop
    for p in PRESSURES:
        outname = out_dir / f"HKE_aifs_{batch_start}_to_{batch_end}_{p}.npz"
        if outname.exists():
            print(f"  Skipping pressure level {p}: output already exists.")
            continue

        print(f"  Computing pressure {p}...")
        hke = compute_hke(ds, p)
        H_power, freqs, kx, ky = hke_spectrum_3d(hke, ds)
        H_rad, k_mag = radial_average_spectrum(H_power, kx, ky)

        outname = out_dir / f"HKE_aifs_{batch_start}_to_{batch_end}_{p}.npz"
        np.savez_compressed(
            outname,
            H_power=H_power,
            H_rad=H_rad,
            freqs=freqs,
            kx=kx,
            ky=ky,
            k_mag=k_mag
        )