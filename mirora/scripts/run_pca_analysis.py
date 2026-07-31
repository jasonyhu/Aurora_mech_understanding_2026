"""
PCA embedding analysis at three temporal scales:
  1. Single day: Sep 1, 2022 (day=1200, night=0000)
  2. Monthly average: all of September 2022
  3. Yearly average: full validation year

For each scale, fits 8 PCAs (4 latent levels × pre/post backbone)
and produces 16 figures (8 PCAs × day/night).
Total: 48 figures.

Directory structure:
  $FIGURES/pca_embeddings/
  ├── single_day_20220901/
  │   ├── pre_backbone_level0_day.png
  │   ├── pre_backbone_level0_night.png
  │   ├── pre_backbone_level1_day.png
  │   ├── ...
  │   ├── post_backbone_level3_day.png
  │   └── post_backbone_level3_night.png
  ├── monthly_202209/
  │   └── (same 16 files)
  └── yearly/
      └── (same 16 files)
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
import sklearn

os.environ["OMP_NUM_THREADS"] = str(len(os.sched_getaffinity(0)))
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

from mirora.viz.plot_rgb import plot_rgb_grid
from mirora.config import LATENT_LEVELS, SPATIAL_DIM, PRE_BACKBONE_DIM, POST_BACKBONE_DIM
from mirora.data.loading import load_embedding, get_rollout_dirs

import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

NUM_LEVELS = LATENT_LEVELS
PRE_DIM = PRE_BACKBONE_DIM
POST_DIM = POST_BACKBONE_DIM
BASE_DIR = consts.EXPERIMENTS_PATH / "base"
SAVE_DIR = consts.FIGURES_PATH / "pca_embeddings"


def is_day(rollout_dir: Path) -> bool:
    return rollout_dir.name.endswith("_1200")


def run_pca_experiment(
    rollout_dirs: list[Path],
    output_dir: Path,
    experiment_label: str,
):
    """Run PCA analysis for a set of rollout directories."""
    sklearn.set_config(assume_finite=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage, emb_dim in [("pre_backbone", PRE_DIM), ("post_backbone", POST_DIM)]:
        logger.info(f"  [{experiment_label}] Processing {stage} (dim={emb_dim})...")

        # One IncrementalPCA per latent level
        ipcas = [IncrementalPCA(n_components=3) for _ in range(NUM_LEVELS)]

        day_sum = np.zeros((NUM_LEVELS, SPATIAL_DIM, emb_dim), dtype=np.float64)
        night_sum = np.zeros((NUM_LEVELS, SPATIAL_DIM, emb_dim), dtype=np.float64)
        day_count = 0
        night_count = 0

        # Pass 1: fit PCAs and accumulate sums
        for d in rollout_dirs:
            emb = load_embedding(d, stage)
            if emb is None:
                continue

            for level in range(NUM_LEVELS):
                ipcas[level].partial_fit(emb[level])

            if is_day(d):
                day_sum += emb
                day_count += 1
            else:
                night_sum += emb
                night_count += 1

        logger.info(f"    Fitted on {day_count} day + {night_count} night samples")
        for level in range(NUM_LEVELS):
            evr = ipcas[level].explained_variance_ratio_
            logger.info(f"    {stage} level {level}: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, PC3={evr[2]:.4f}, total={evr.sum():.4f}")

        if day_count == 0 or night_count == 0:
            logger.warning(f"    Skipping {stage}: missing day or night data")
            continue

        avg_day = (day_sum / day_count).astype(np.float32)
        avg_night = (night_sum / night_count).astype(np.float32)

        # Pass 2: transform and plot
        for level in range(NUM_LEVELS):
            day_pcs = ipcas[level].transform(avg_day[level])
            night_pcs = ipcas[level].transform(avg_night[level])

            # Normalize RGB jointly so day/night are comparable
            joint_min = np.minimum(day_pcs.min(axis=0), night_pcs.min(axis=0))
            joint_max = np.maximum(day_pcs.max(axis=0), night_pcs.max(axis=0))

            day_rgb = ((day_pcs - joint_min) / (joint_max - joint_min + 1e-8)).reshape(150, 300, 3)
            night_rgb = ((night_pcs - joint_min) / (joint_max - joint_min + 1e-8)).reshape(150, 300, 3)

            day_rgb = np.clip(day_rgb, 0, 1)
            night_rgb = np.clip(night_rgb, 0, 1)

            plot_rgb_grid(
                day_rgb,
                f"{stage.replace('_', '-')} level {level} — day ({experiment_label})",
                output_dir / f"{stage}_level{level}_day.png",
            )
            plot_rgb_grid(
                night_rgb,
                f"{stage.replace('_', '-')} level {level} — night ({experiment_label})",
                output_dir / f"{stage}_level{level}_night.png",
            )

    logger.info(f"  [{experiment_label}] Done. Figures saved to {output_dir}")


def main():
    # Experiment 1: Single day — Sep 1, 2022
    logger.info("=== Experiment 1: Single day (Sep 1, 2022) ===")
    single_day_dirs = get_rollout_dirs(BASE_DIR, datetime(2022, 9, 1, 0), datetime(2022, 9, 1, 12))
    run_pca_experiment(
        single_day_dirs,
        SAVE_DIR / "single_day_20220901",
        "Sep 1 2022",
    )

    # Experiment 2: Monthly average — September 2022
    logger.info("=== Experiment 2: Monthly average (Sep 2022) ===")
    monthly_dirs = get_rollout_dirs(BASE_DIR, datetime(2022, 9, 1, 0), datetime(2022, 9, 30, 12))
    run_pca_experiment(
        monthly_dirs,
        SAVE_DIR / "monthly_202209",
        "Sep 2022 avg",
    )

    # Experiment 3: Yearly average — full validation year
    logger.info("=== Experiment 3: Yearly average ===")
    yearly_dirs = get_rollout_dirs(BASE_DIR, consts.val_start_time, consts.val_end_time)
    run_pca_experiment(
        yearly_dirs,
        SAVE_DIR / "yearly",
        "yearly avg",
    )

    logger.info(f"All 48 figures saved under {SAVE_DIR}")


if __name__ == "__main__":
    main()
