"""Path conventions for CAMS forecasts, init batches, and rollouts, plus script logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_init_batch_path(dir_path: Path, dt: datetime):
    return dir_path / dt.strftime(r"init_%Y%m%d_%H%M")

def get_cams_forecast_path(dir_path: Path, dt: datetime):
    return dir_path / dt.strftime(r"cams_%Y%m%d_%H%M")

def get_rollout_dir(dir_path: Path, dt: datetime):
    return dir_path / dt.strftime(r"%Y%m%d_%H%M")


def setup_logging(level: int = logging.INFO) -> None:
    """Log to stdout with timestamps (for entry-point scripts; the library never calls this)."""
    logging.basicConfig(level=level, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
