"""Path conventions for CAMS forecasts, init batches, and rollouts.

Variable name mappings between CAMS and Aurora live in :mod:`mirora.config`.
"""

from pathlib import Path
from datetime import datetime


def get_init_batch_path(dir_path: Path, dt: datetime):
    return dir_path / dt.strftime(r"init_%Y%m%d_%H%M")

def get_cams_forecast_path(dir_path: Path, dt: datetime):
    return dir_path / dt.strftime(r"cams_%Y%m%d_%H%M")

def get_rollout_dir(dir_path: Path, dt: datetime):
    return dir_path / dt.strftime(r"%Y%m%d_%H%M")
