import os
from datetime import datetime
import numpy as np
import xarray as xr
from pathlib import Path
from numpy.fft import fftn, fftshift, fftfreq
from tqdm import tqdm


# ---------------------------------------------------
# Config (GC model)
# ---------------------------------------------------
folder = Path("/home/project/17001770/weather_department/nwp/zach/aurora_folder/aurora_0p25deg/")

out_dir = Path("/home/project/17001770/weather_department/nwp/zach/research_folder/spectra_npzs/aurora/")
out_dir.mkdir(exist_ok=True)

LOG = out_dir / "missing_batches.log"
if LOG.exists():
    LOG.unlink()

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
PRESSURES = [850, 700, 200]

required_outs = [6, 12, 18, 24, 30, 36, 42, 48]


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def log_missing(init_str, missing_list):
    with LOG.open("a") as f:
        f.write(f"{init_str}: missing {missing_list}\n")


def parse_init(fname):
    # aurora_forecast_2024-03-13_12-out-12.nc
    core = fname.replace("aurora_forecast_", "").replace(".nc", "")
    init_part, _ = core.split("-out-")
    return datetime.strptime(init_part, "%Y-%m-%d_%H"), init_part


def compute_hke(ds, level):
    u = ds["u"].sel(isobaricInhPa=level).values
    v = ds["v"].sel(isobaricInhPa=level).values
    return 0.5 * (u**2 + v**2)


def hke_spectrum_3d(hke, ds):
    ds = ds.sortby("valid_time")
    time_vals = ds.valid_time.values
    dt_seconds = (time_vals[1] - time_vals[0]) / np.timedelta64(1, "s")

    lat = ds.latitude.values
    lon = ds.longitude.values

    dy_deg = abs(lat[1] - lat[0])
    dx_deg = abs(lon[1] - lon[0])

    earth_r = 6371220.0
    meters_deg = 2 * np.pi * earth_r / 360.0
    dx = dx_deg * meters_deg
    dy = dy_deg * meters_deg

    H = fftn(hke)
    H_power = fftshift(np.abs(H) ** 2, axes=(0, 1, 2))

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
# Build GC batches (48h per init)
# ---------------------------------------------------
all_files = [
    f for f in os.listdir(folder)
    if f.endswith(".nc")          # require ONLY .grib, not .grib.idx
    and f.startswith("aurora_forecast_")
    and "-out-" in f
]

groups = {}
for f in all_files:
    dt, init_str = parse_init(f)
    groups.setdefault(init_str, []).append(f)

groups_sorted = sorted(groups.items(), key=lambda x: datetime.strptime(x[0], "%Y-%m-%d_%H"))

batches = []

for init_str, files in groups_sorted:
    expected = [f"aurora_forecast_{init_str}-out-{hr}.nc" for hr in required_outs]
    missing = [f for f in expected if f not in files]

    if missing:
        log_missing(init_str, missing)
        continue

    batches.append((init_str, expected))


print(f"Found {len(batches)} complete 48-hour GC batches")


# ---------------------------------------------------
# Process batches
# ---------------------------------------------------
for init_str, filelist in tqdm(batches, desc="Processing AURORA 48h batches"):
    batch_start = init_str
    batch_end = f"{init_str}_+48h"

    files_48h = [folder / f for f in filelist]
    print(batch_start)
    for i in files_48h:
        print(f"    {i}")

    ds = xr.open_mfdataset(
        files_48h,
        combine="nested",
        concat_dim="time",
        parallel=False,
    )

    ds = (
        ds.sortby("latitude").sortby("longitude")
          .sel(latitude=slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
               longitude=slice(DOMAIN["lon_min"], DOMAIN["lon_max"]))
    )

    for p in tqdm(PRESSURES, leave=False, desc=f"{batch_start}"):
        hke = compute_hke(ds, p)
        H_power, freqs, kx, ky = hke_spectrum_3d(hke, ds)
        H_rad, k_mag = radial_average_spectrum(H_power, kx, ky)

        outname = out_dir / f"HKE_aurora_{batch_start}_to_{batch_end}_{p}.npz"

        np.savez_compressed(
            outname,
            H_power=H_power,
            H_rad=H_rad,
            freqs=freqs,
            kx=kx,
            ky=ky,
            k_mag=k_mag
        )
