"""
Reusable data loading functions for Aurora rollouts, CAMS forecasts,
FINN fire detections, and backbone embeddings.
"""

import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from aurora import Batch

from mirora.config import LATENT_LEVELS, MMR_TO_PPB, SPATIAL_DIM, SURFACE_LEVEL_IDX
from mirora.utils import get_init_batch_path, get_rollout_dir

logger = logging.getLogger(__name__)


def fetch_rollout(rollout_path: Path) -> list[Batch]:
    """Load a pickled list of Aurora Batch predictions."""
    with open(rollout_path, "rb") as f:
        return pickle.load(f)


def surface(pred: Batch, var: str, ppb: bool = False) -> np.ndarray:
    """[H, W] field of atmospheric variable `var` at the 1000 hPa level of one prediction,
    as mass mixing ratio or (ppb=True) ppb by volume."""
    field = pred.atmos_vars[var][0, 0, SURFACE_LEVEL_IDX].numpy()
    return field * MMR_TO_PPB[var] if ppb else field


def load_batch(batch_dir: Path, init_time: datetime) -> Batch:
    """Load the pickled Aurora init Batch for `init_time`."""
    with open(get_init_batch_path(batch_dir, init_time), "rb") as f:
        return pickle.load(f)


def available_init_times(batch_dir: Path, start: datetime, end: datetime) -> list[datetime]:
    """Every 12-hourly init time in [start, end] that has a saved init batch."""
    times = []
    cur = start
    while cur <= end:
        if get_init_batch_path(batch_dir, cur).exists():
            times.append(cur)
        cur += timedelta(hours=12)
    return times


def load_cams_step(
    cams_data_path: Path,
    init_date_str: str,
    step: int,
    var_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CAMS forecast variable at a given step.

    Args:
        cams_data_path: Root directory containing cams_YYYYMMDD_HHMM/ subdirs.
        init_date_str: Init time string like '20220901_0000'.
        step: Rollout step (0-indexed). Maps to forecast_period step+1
              since period 0 is the analysis.
        var_key: Variable name (e.g. 'tcco', 'pm2p5').

    Returns:
        (data, lats, lons) arrays.
    """
    sfc_path = cams_data_path / f"cams_{init_date_str}" / "data_sfc.nc"
    ds = xr.open_dataset(sfc_path)
    data = ds[var_key].isel(forecast_period=step + 1, forecast_reference_time=0).values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.close()
    return data, lats, lons


def load_embedding(
    rollout_dir: Path,
    stage: str,
    num_levels: int = LATENT_LEVELS,
    spatial_dim: int = SPATIAL_DIM,
) -> np.ndarray | None:
    """Load the first rollout step's backbone embedding, reshaped to (levels, spatial, D).

    Args:
        rollout_dir: Directory containing *_embeddings.pickle files.
        stage: 'pre_backbone' or 'post_backbone'.

    Returns:
        Array of shape (num_levels, spatial_dim, D), or None if not found.
    """
    path = rollout_dir / f"{stage}_embeddings.pickle"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    emb = data[0].squeeze().cpu().numpy()
    D = emb.shape[-1]
    return emb.reshape(num_levels, spatial_dim, D)


def get_rollout_dirs(
    base_dir: Path,
    start: datetime,
    end: datetime,
) -> list[Path]:
    """Get all rollout directories between start and end (12-hourly)."""
    dirs = []
    cur = start
    while cur <= end:
        d = get_rollout_dir(base_dir, cur)
        if d.exists():
            dirs.append(d)
        cur += timedelta(hours=12)
    return dirs


def load_finn_fires(
    finn_dir: Path,
    event_name: str,
    valid_time: datetime,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load FINN fire locations for a specific date.

    Tries gridded NetCDF files first, then falls back to text files.

    Returns:
        (lats, lons) of active fire grid cells, or None if not found.
    """
    import pandas as pd

    finn_event_dir = finn_dir / event_name
    if not finn_event_dir.exists():
        return None

    nc_files = sorted(finn_event_dir.glob("*.nc")) + sorted(finn_event_dir.glob("*.nc4"))
    for nc_path in nc_files:
        ds = xr.open_dataset(nc_path)
        time_dim = None
        for dim in ["time", "date", "day"]:
            if dim in ds.dims or dim in ds.coords:
                time_dim = dim
                break

        if time_dim is not None:
            times = pd.to_datetime(ds[time_dim].values)
            target_date = valid_time.date()
            time_mask = [t.date() == target_date for t in times]
            if not any(time_mask):
                continue
            idx = time_mask.index(True)
            co_data = ds["fire_modisviirs_CO"].isel({time_dim: idx})
        else:
            doy = valid_time.timetuple().tm_yday
            if (f"{valid_time.year}{doy:03d}" not in nc_path.name
                    and valid_time.strftime("%Y%m%d") not in nc_path.name):
                continue
            co_data = ds["fire_modisviirs_CO"]

        co_vals = co_data.squeeze().values
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lats = ds[lat_name].values
        lons = ds[lon_name].values

        fire_mask = co_vals > 0
        if not fire_mask.any():
            ds.close()
            continue

        fire_lat_idx, fire_lon_idx = np.where(fire_mask)
        fire_lats = lats[fire_lat_idx]
        fire_lons = lons[fire_lon_idx]
        # Convert 0-360 to -180-180 if needed
        fire_lons = np.where(fire_lons > 180, fire_lons - 360, fire_lons)
        ds.close()
        return fire_lats, fire_lons
