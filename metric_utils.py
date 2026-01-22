import numpy as np
import xarray as xr
import cfgrib
import pandas as pd
from datetime import datetime
import os
import re

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140} # this is the sea domain

def slice_domain(ds, domain=DOMAIN):
    """Apply SEA domain slice with consistent lat/lon ordering."""

    if 'latitude' in ds.coords:
        return (
            ds.sortby('latitude')
            .sortby('longitude')
            .sel(latitude=slice(domain['lat_min'], domain['lat_max']),
                longitude=slice(domain['lon_min'], domain['lon_max']))
        )

    else:
        return (
            ds.sortby('lat')
            .sortby('lon')
            .sel(lat=slice(domain['lat_min'], domain['lat_max']),
                lon=slice(domain['lon_min'], domain['lon_max']))
        )


def load_dataset(path, engine="nc", var=""):
    """Unified loader for GRIB or NetCDF with domain slicing. Specify var for the data you want to extract"""
    if engine == "nc":
        ds = xr.open_dataset(path)
    else:
        if var == "isobaricInhPa":
            ds = xr.open_dataset(path, engine=engine, filter_by_keys={"typeOfLevel": var})
        elif var == "heightAboveGround":
            ds = xr.open_dataset(path, engine=engine, filter_by_keys={"shortName": ["10u", "10v"], "typeOfLevel": var})
        elif var == 'surface':
            ds = xr.open_dataset(path, engine=engine, filter_by_keys={"typeOfLevel": var})
        elif var == "heightAboveGround_t2m":
            ds = xr.open_dataset(path, engine=engine, filter_by_keys={"shortName": "2t", "typeOfLevel": "heightAboveGround"})
        else:
            ds = xr.open_dataset(path, engine='cfgrib')
            
    return slice_domain(ds)


def compute_mse(t, z, q, cp = 1004, g = 9.81, Lv = 2.5e6):
    # temperature, geopotential z, specific humidity q
    g_z = z/g
    return cp*t + g*g_z + Lv*q # mse J/kg

def safe_gradient(arr, axis, spacing):
    """
    Compute gradient while ignoring NaNs:
    - Replaces NaNs with nearest valid value along the axis.
    """
    arr_copy = arr.copy()
    arr_copy = np.where(np.isnan(arr_copy), np.nan_to_num(arr_copy, nan=0.0), arr_copy)
    grad = np.gradient(arr_copy, axis=axis) / spacing
    return grad

def compute_mse_vorticity_convergence(u, v, lats, lons, mse):
    """
    Compute horizontal MSE convergence:
    - u, v: wind components (2D: lat x lon or 3D: lev x lat x lon)
    - lats, lons: 1D arrays
    - mse: MSE array of same shape as u/v
    Returns: horizontal convergence array (same shape)
    """
    Re = 6371000  # Earth radius in meters

    # Convert lat/lon spacing to radians
    dlat = np.deg2rad(np.gradient(lats))          # 1D array
    dlon = np.deg2rad(np.gradient(lons))          # 1D array
    coslat = np.cos(np.deg2rad(lats))
    coslat = np.clip(coslat, 1e-3, 1.0)
    # print("cos(lat) min/max:", coslat.min(), coslat.max())

    # vorticity calculation
    zeta = np.gradient(v, axis=-1) / (dlon[None, :] * Re * coslat[:, None]) - np.gradient(u, axis=-2) / (dlat[:, None] * Re)
    
    # MSE fluxes
    Fx = u
    Fy = v
    
    # J kg^-1 s^-1
    mse_dudx = np.gradient(Fx*mse, axis=-1) / (dlon[None, :] * Re * coslat[:, None])
    mse_dvdy = np.gradient(Fy*mse, axis=-2) / (dlat[:, None] * Re) 

    # J kg^-1 s^-2
    zeta_dudx = np.gradient(Fx*mse*zeta, axis=-1) / (dlon[None, :] * Re * coslat[:, None])
    zeta_dvdy = safe_gradient(Fy*zeta*mse, axis=-2, spacing=(dlat[:, None] * Re))
    nan_mask = np.isnan(v*zeta*mse)

    # convert vorticity 0.0 to nans
    clean_array = np.where(nan_mask, 0.0, v*zeta*mse) # Replace NaNs with 0 for gradient computation
    zeta_dvdy = np.gradient(clean_array, axis=-2) / (dlat[:, None] * Re) # Compute gradient safely
    zeta_dvdy[nan_mask] = np.nan  # Restore NaNs

    # Horizontal divergence
    mse_div_F = mse_dudx + mse_dvdy
    zeta_div_F = zeta_dudx + zeta_dvdy

    # Horizontal convergence = - divergence
    mse_convergence = np.where(mse_div_F < 0, mse_div_F, np.nan)
    mse_divergence = np.where(mse_div_F > 0, mse_div_F, np.nan)
    zeta_convergence = np.where(zeta_div_F < 0, zeta_div_F, np.nan)
    zeta_divergence = np.where(zeta_div_F > 0, zeta_div_F, np.nan)
    
    return mse_convergence, mse_divergence, zeta_convergence, zeta_divergence

def compute_t2m_energy(u10, v10, lats, lons, t2m, cp=1004, density_sfc=1.2):
    """
    Computes the surface temperature energy convergence
    """
    cp_t2m = cp * t2m * density_sfc
    Re = 6371000
    dlat = np.deg2rad(np.gradient(lats))
    dlon = np.deg2rad(np.gradient(lons))
    coslat = np.clip(np.cos(np.deg2rad(lats)), 1e-3, 1.0)

    dudx = np.gradient(u10*cp_t2m, axis=-1) / (dlon[None, :] * Re * coslat[:, None])
    dvdy = np.gradient(v10*cp_t2m, axis=-2) / (dlat[:, None] * Re)
    return dudx + dvdy

def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f.ravel() - t.ravel())**2))


def compute_drwb_pl(ds, model=''):
    """
    Computes the dynamic weatherbench pressure metrics
    note: the pressure level name might be different for some model outputs, 
    if its different its better to rename it to isobaricInhPa, itll make your life easier
    """
    z_850 = ds['z'].sel(isobaricInhPa=850).values
    t_850 = ds['t'].sel(isobaricInhPa=850).values
    z_200 = ds['z'].sel(isobaricInhPa=200).values
    t_200 = ds['t'].sel(isobaricInhPa=200).values
    q_850 = ds['q'].sel(isobaricInhPa=850).values
    q_200 = ds['q'].sel(isobaricInhPa=200).values

    u_850 = ds['u'].sel(isobaricInhPa=850).values
    v_850 = ds['v'].sel(isobaricInhPa=850).values
    u_200 = ds['u'].sel(isobaricInhPa=200).values
    v_200 = ds['v'].sel(isobaricInhPa=200).values

    mse_850 = compute_mse(t_850, z_850, q_850)
    mse_200 = compute_mse(t_200, z_200, q_200)
    mse_convergence_850, _, zeta_convergence_850, _ = compute_mse_vorticity_convergence(
        u_850, v_850, ds.latitude.values, ds.longitude.values, mse_850)
    _, mse_divergence_200, _, _ = compute_mse_vorticity_convergence(
        u_200, v_200, ds.latitude.values, ds.longitude.values, mse_200)
    
    return mse_convergence_850, mse_divergence_200, zeta_convergence_850


def compute_drwb_sfc(ds_sfc, model='', ds_t2m=None):
    """
    Computes the surfact temperature energy convergence, 
    some models have different variables names therefore you can either change the name here
    or just rename them before computing their metrics
    """
    if model == 'au':
        t2m = ds_t2m['2t'].values
        u10 = ds_sfc['10u'].values
        v10 = ds_sfc['10v'].values
    else:
        t2m = ds_t2m['t2m'].values
        u10 = ds_sfc['u10'].values
        v10 = ds_sfc['v10'].values
    

    return compute_t2m_energy(u10, v10, ds_sfc.latitude.values, ds_sfc.longitude.values, t2m)


def rh_to_specific_humidity(rh, t, p):
    """
    Convert relative humidity (%) to specific humidity (kg/kg)
    applicable to models that do not have specific umidity and only have relative humidity
    """
    t_c = t - 273.15
    e_s = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e = rh / 100.0 * e_s
    q = 0.622 * e / (p - 0.378 * e)
    return q

def compute_wind_rmse(u_truth, v_truth, u_forecast, v_forecast):
    """
    computes the wind rmse in the traditional weatherbench
    """
    rmse_u = np.sqrt(np.nanmean((u_truth.ravel() - u_forecast.ravel())**2))
    rmse_v = np.sqrt(np.nanmean((v_truth.ravel() - v_forecast.ravel())**2))
    rmse_vec = np.sqrt(rmse_u**2 + rmse_v**2)
    return rmse_vec, rmse_u, rmse_v


def extract_trwb_vars(ds, model=''):

    """
    extracts the tradiitonal weatherbench variables from the dataset
    """
    if model == 'aifs':
        z = ds['z'].sel(level=500).squeeze().values
        t = ds['t'].sel(level=850).squeeze().values
        q = ds['q'].sel(level=700).squeeze().values
        u = ds['u'].sel(level=850).squeeze().values
        v = ds['v'].sel(level=850).squeeze().values
    else:  # default for other models
        z = ds['z'].sel(isobaricInhPa=500).values
        t = ds['t'].sel(isobaricInhPa=850).values
        if model == 'fv2':
            q = rh_to_specific_humidity(ds['r'].sel(isobaricInhPa=700), ds['t'].sel(isobaricInhPa=700), 700).values
        else:
            q = ds['q'].sel(isobaricInhPa=700).values
        u = ds['u'].sel(isobaricInhPa=850).values
        v = ds['v'].sel(isobaricInhPa=850).values

    return z, t, q, u, v

# function to compute all weatherbench metrics, traditional and dynamic
def compute_weatherbenches_json(truth_pl, truth_sfc, truth_t2m, forecast_pl, forecast_sfc, forecast_t2m, model='au', gpm=None, forecast_tp=None):
    # --- traditional metrics ---
    fc_z, fc_t, fc_q, fc_u, fc_v = extract_trwb_vars(forecast_pl, model='')
    t_z, t_t, t_q, t_u, t_v = extract_trwb_vars(truth_pl, model='')

    wind_vector_rmse, _, _ = compute_wind_rmse(t_u, t_v, fc_u, fc_v)

    traditional_metrics = {
        'z_500_rmse': compute_scalar_rmse(fc_z, t_z),
        't_850_rmse': compute_scalar_rmse(fc_t, t_t),
        'q_700_rmse': compute_scalar_rmse(fc_q, t_q),
        'u_850_rmse': compute_scalar_rmse(fc_u, t_u),
        'v_850_rmse': compute_scalar_rmse(fc_v, t_v),
        'wind_rmse_850': wind_vector_rmse
    }

    # --- dynamic metrics ---
    fc_mse_c850, fc_mse_d200, fc_zeta850 = compute_drwb_pl(forecast_pl)
    t_mse_c850, t_mse_d200, t_zeta850 = compute_drwb_pl(truth_pl)
    fc_ste = compute_drwb_sfc(forecast_sfc, ds_t2m=forecast_t2m, model=model)
    t_ste  = compute_drwb_sfc(truth_sfc, ds_t2m=truth_t2m, model=model)

    dynamic_metrics = {
        'mse_conv_850': compute_scalar_rmse(fc_mse_c850, t_mse_c850),
        'mse_div_200': compute_scalar_rmse(fc_mse_d200, t_mse_d200),
        'vorticity_conv_850': compute_scalar_rmse(fc_zeta850, t_zeta850),
        'surf_temp_energy_conv': compute_scalar_rmse(fc_ste, t_ste)
    }

    if gpm is not None and forecast_tp is not None:
        dynamic_metrics['total_precipitation'] = compute_scalar_rmse(
            forecast_tp['tp'].squeeze().values, gpm['precipitation'].squeeze().values
        )

    # merge everything into a single flat JSON
    all_metrics = {**traditional_metrics, **dynamic_metrics}
    return all_metrics
