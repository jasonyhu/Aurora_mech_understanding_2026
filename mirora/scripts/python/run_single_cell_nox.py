"""
Single-grid-cell NOx perturbation to demonstrate that a localized input
disturbance produces non-local (global) effects through Aurora's operator.

Spikes NO and NO2 in ONE grid cell over a megacity/emissions hotspot
(Beijing, ~39.9N 116.4E), across all pressure levels and both input
timesteps, then runs a 10-step rollout. Baseline is the Sep 1 2022 00Z
base rollout. Compare with plot_single_cell_nox.py.
"""

import pickle
import logging
import sys
from datetime import datetime

import numpy as np
import torch

from mirora.inference.inference import aap_rollout
from mirora.utils import get_init_batch_path
import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

INIT_TIME = datetime(2022, 9, 1, 0)
ROLLOUT_STEPS = 10
MAG_PPB = 100.0            # spike magnitude at the single cell
TARGET_LAT, TARGET_LON = 39.9, 116.4   # Beijing
EXP_NAME = "single_cell_nox_beijing"


def nearest_idx(coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coords - value)))


def main():
    with open(get_init_batch_path(consts.BATCH_INITS_PATH, INIT_TIME), "rb") as f:
        batch = pickle.load(f)

    lat = np.asarray(batch.metadata.lat)
    lon = np.asarray(batch.metadata.lon)          # 0..359.6
    lat_i = nearest_idx(lat, TARGET_LAT)
    lon_i = nearest_idx(lon, TARGET_LON % 360)
    logger.info(f"Target cell: lat[{lat_i}]={lat[lat_i]:.2f}, lon[{lon_i}]={lon[lon_i]:.2f}")

    offset = MAG_PPB * 1e-9
    for var in ("no", "no2"):
        # shape [B=1, T, L, H, W] — spike the single (H,W) cell at all levels/times
        batch.atmos_vars[var][..., lat_i, lon_i] = (
            batch.atmos_vars[var][..., lat_i, lon_i] + offset
        )
    logger.info(f"Applied +{MAG_PPB} ppb NO/NO2 spike at single cell.")

    save_dir = consts.EXPERIMENTS_PATH / EXP_NAME
    aap_rollout(
        batch=batch,
        batch_date=INIT_TIME,
        aap_checkpoint_path=consts.CHECKPOINT_PATH,
        save_dir=save_dir,
        rollout_steps=ROLLOUT_STEPS,
        save_backbone_activations=False,
    )
    logger.info(f"Saved single-cell rollout under {save_dir}")


if __name__ == "__main__":
    main()
