"""
Methods for initializing and saving Aurora Batch objects.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import torch
import xarray as xr
from aurora import Batch, Metadata
from torch.utils.data import Dataset

from mirora.config import CAMS_TO_AAP_SURF_VARS, CAMS_TO_AAP_ATMOS_VARS
from mirora.utils import get_init_batch_path, get_cams_forecast_path

logger = logging.getLogger(__name__)

def verify_init_batch(batch_path: Path) -> bool:
    try:
        with open(batch_path, 'rb') as batch_file:
            pickle.load(batch_file)
    except Exception:
        logger.warning(f"Aurora batch at path {str(batch_path)} not found.")
        return False
    else:
        return True

def verify_all_init_batches(dir_path: Path, start: datetime, end: datetime) -> None:
    """
    Checks that all batches between `start` and `end` (inclusive) exist in the `dir_path` directory.
    """
    cur_time = start
    while cur_time <= end:
        batch_path = get_init_batch_path(dir_path, cur_time)
        verify_init_batch(batch_path)
        cur_time += timedelta(hours=12)

def load_saved_cams_forecast(dir_path: Path, init_time: datetime, forecast_period: int=0) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Loads the CAMS forecast initialized at `init_time` saved in the `dir_path` directory. 
    Returns two xarray datasets, 
    the first of surface CAMS variables and 
    the second of atmospheric CAMS variables.
    """
    batch_path = get_cams_forecast_path(dir_path, init_time)

    surf_vars_ds = xr.open_dataset(
        batch_path / "data_sfc.nc", 
        engine="netcdf4", 
        decode_timedelta=True
    ).isel(forecast_period=forecast_period)

    atmos_vars_ds = xr.open_dataset(
        batch_path / "data_plev.nc", 
        engine="netcdf4", 
        decode_timedelta=True
    ).isel(forecast_period=forecast_period)

    return surf_vars_ds, atmos_vars_ds

def init_aap_batch(init_time: datetime, cams_save_path: Path, static_vars_path: Path) -> Batch:
    """
    Initialize an AuroraAirPollution Batch with 
    CAMS data at time `init_time` and 12 hours before.
    """
    if init_time.hour not in (0, 12) or init_time.minute != 0:
        raise ValueError(f"Initialization datetime {init_time} must be 00:00Z or 12:00Z.")
    
    with open(static_vars_path, "rb") as static_file:
        static_vars = pickle.load(static_file)

    t_minus1 = init_time - timedelta(hours=12)
    surf_t_minus1, atmos_t_minus1 = load_saved_cams_forecast(cams_save_path, t_minus1)
    surf_init, atmos_init = load_saved_cams_forecast(cams_save_path, init_time)
    
    surf_vars, atmos_vars = {}, {}
    for cams_surf_var, aap_surf_var in CAMS_TO_AAP_SURF_VARS.items():
        surf_vars[aap_surf_var] = torch.stack(
            (
                torch.from_numpy(surf_t_minus1[cams_surf_var][0].values), 
                torch.from_numpy(surf_init[cams_surf_var][0].values),
            )
        ).unsqueeze(0)  # dimension (B, T, L, W)
    for cams_atmos_var, aap_atmos_var in CAMS_TO_AAP_ATMOS_VARS.items():
        atmos_vars[aap_atmos_var] = torch.stack(
            (
                torch.from_numpy(atmos_t_minus1[cams_atmos_var][0].values), 
                torch.from_numpy(atmos_init[cams_atmos_var][0].values),
            )
        ).unsqueeze(0)  # dimension (B, T, H, L, W)
    
    batch = Batch(
        surf_vars=surf_vars,
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=torch.from_numpy(atmos_init.latitude.values),
            lon=torch.from_numpy(atmos_init.longitude.values),
            time=(init_time,),
            atmos_levels=tuple(int(level) for level in atmos_init.pressure_level.values),
        ),
    )
    return batch

def save_init_batch(init_time: datetime, save_dir: Path, cams_save_path: Path, static_vars_path: Path):
    """
    Generate and store an AuroraAirPollution initialization Batch
    at time `init_time`.
    """
    batch_path = get_init_batch_path(save_dir, init_time)
    if verify_init_batch(batch_path):
        logger.info(f"Aurora init batch at time {init_time} already exists.")
        return
    logger.info(f"Creating AuroraAirPollution initialization batch at time {init_time}.")
    init_batch = init_aap_batch(init_time, cams_save_path, static_vars_path)
    with open(batch_path, 'wb') as batch_file:
        pickle.dump(init_batch, batch_file)

class AuroraInitBatchesDataset(Dataset):
    """
    PyTorch Dataset for loading Aurora batches.
    """
    def __init__(self, save_dir: Path, start: datetime, end: datetime, skip_batch_check: bool=False):
        self.inits = []
        cur_time = start
        
        while cur_time <= end:
            batch_path = get_init_batch_path(save_dir, cur_time)
            if skip_batch_check or verify_init_batch(batch_path):
                self.inits.append(batch_path)
            else:
                logger.warning(f"Skipping missing file: {batch_path}")
            cur_time += timedelta(hours=12)

        if not self.inits:
            logger.warning("No valid batches found in the specified time range.")

    def __len__(self):
        return len(self.inits)

    def __getitem__(self, idx: int) -> Batch:
        batch_path = self.inits[idx]
        with open(batch_path, 'rb') as init_file:
            batch = pickle.load(init_file)
        return batch
