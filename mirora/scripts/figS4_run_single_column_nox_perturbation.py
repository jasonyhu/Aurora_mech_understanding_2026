"""
Text S1 / Figure S4 rollout: single-grid-cell NOx perturbation to demonstrate that a localized input
disturbance produces non-local (global) effects through Aurora's operator.

Spikes NO and NO2 in ONE grid cell over a megacity/emissions hotspot
(Beijing, ~39.9N 116.4E), across all pressure levels and both input
timesteps, then runs a 10-step rollout. Baseline is the Sep 1 2022 00Z
base rollout. Compare with figS4_plot_single_column_nox_response.py.
"""

import logging

import numpy as np

from mirora.data.loading import load_batch
from mirora.inference.inference import aap_rollout
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

INIT_TIME = consts.MAIN_INIT_TIME
ROLLOUT_STEPS = 10
MAG_PPB = 100.0            # spike magnitude at the single cell
TARGET_LAT, TARGET_LON = 39.9, 116.4   # Beijing
EXP_NAME = "single_cell_nox_beijing"


def nearest_idx(coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coords - value)))


def main():
    setup_logging()
    consts.ensure_checkpoint()
    batch = load_batch(consts.BATCH_INITS_PATH, INIT_TIME)

    lat = np.asarray(batch.metadata.lat)
    lon = np.asarray(batch.metadata.lon)          # 0..359.6
    lat_i = nearest_idx(lat, TARGET_LAT)
    lon_i = nearest_idx(lon, TARGET_LON % 360)
    logger.info(f"Target cell: lat[{lat_i}]={lat[lat_i]:.2f}, lon[{lon_i}]={lon[lon_i]:.2f}")

    offsets = consts.nox_offsets(MAG_PPB)
    for var, offset in offsets.items():
        # shape [B=1, T, L, H, W] — spike the single (H,W) cell at all levels/times
        batch.atmos_vars[var][..., lat_i, lon_i] = (
            batch.atmos_vars[var][..., lat_i, lon_i] + offset
        )
    logger.info(f"Applied +{MAG_PPB} {consts.NOX_UNIT_LABEL} NO/NO2 spike at single cell.")

    save_dir = consts.NOX_EXPERIMENTS_PATH / EXP_NAME
    aap_rollout(
        batch=batch,
        batch_date=INIT_TIME,
        aap_checkpoint_path=consts.CHECKPOINT_PATH,
        save_dir=save_dir,
        rollout_steps=ROLLOUT_STEPS,
    )
    logger.info(f"Saved single-cell rollout under {save_dir}")


if __name__ == "__main__":
    main()
