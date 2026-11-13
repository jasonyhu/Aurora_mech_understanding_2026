"""
Defines the CAMSDataloader class.
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from mirora.utils import get_cams_forecast_path

logger = logging.getLogger(__name__)

class CAMSDataloader:
    def __init__(self, dir_path: Path):
        self.dir_path = dir_path

    def load_atmos_var(
        self, dt: datetime, key: str, level: int, 
        forecast_period: int=0
    ) -> np.ndarray:
        """
        Loads a 2d grid of the atmospheric variable `key` at time `dt` and level `level`, 
        with a forecast offset of `forecast_period`. Valid keys and levels are defined in aurora_consts,
        and valid forecast periods depend on the number of rollout steps executed.
        """
        atmos_vars_path = get_cams_forecast_path(self.dir_path, dt) / "data_plev.nc"
        atmos_vars = xr.open_dataset(atmos_vars_path, engine="netcdf4", decode_timedelta=True)
        data = atmos_vars.isel(forecast_period=forecast_period)[key][0, level].to_numpy()
        return data
