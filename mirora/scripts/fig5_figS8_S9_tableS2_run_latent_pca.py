"""
PCA embedding analysis (Section 2.3) at three temporal scales:
  1. Single day: Sep 1, 2022 (day=1200, night=0000)     -> Figure 5
  2. Monthly average: all of September 2022              -> Figure S8
  3. Yearly average: full validation year                -> Figure S9
plus the top-3 subspace overlap between the three fits (Table S2).

For each scale, fits 8 PCAs (4 latent levels x pre/post backbone)
and produces 16 figures (8 PCAs x day/night).
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

--no-title writes the same figures without titles to <scale>_notitle/;
--n-components k (k != 3) writes to pca_embeddings/n{k}/.
"""

import csv
import os
import logging
from argparse import ArgumentParser
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import IncrementalPCA
import sklearn

os.environ["OMP_NUM_THREADS"] = str(len(os.sched_getaffinity(0)))
os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

from mirora.viz.maps import setup_matplotlib_style
from mirora.viz.plot_rgb import plot_rgb_grid
from mirora.config import LATENT_LEVELS, PATCH_H, PATCH_W, SPATIAL_DIM, PRE_BACKBONE_DIM, POST_BACKBONE_DIM
from mirora.data.loading import load_embedding, get_rollout_dirs
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

BASE_DIR = consts.EXPERIMENTS_PATH / "base"
SAVE_DIR = consts.FIGURES_PATH / "pca_embeddings"


def is_day(rollout_dir: Path) -> bool:
    return rollout_dir.name.endswith("_1200")


def run_pca_experiment(
    rollout_dirs: list[Path],
    output_dir: Path,
    experiment_label: str,
    n_components: int = 3,
    titles: bool = True,
):
    """Run PCA analysis for a set of rollout directories."""
    sklearn.set_config(assume_finite=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    loadings = {}

    for stage, emb_dim in [("pre_backbone", PRE_BACKBONE_DIM), ("post_backbone", POST_BACKBONE_DIM)]:
        logger.info(f"  [{experiment_label}] Processing {stage} (dim={emb_dim})...")

        # One IncrementalPCA per latent level
        ipcas = [IncrementalPCA(n_components=n_components) for _ in range(LATENT_LEVELS)]

        day_sum = np.zeros((LATENT_LEVELS, SPATIAL_DIM, emb_dim), dtype=np.float64)
        night_sum = np.zeros((LATENT_LEVELS, SPATIAL_DIM, emb_dim), dtype=np.float64)
        day_count = 0
        night_count = 0

        # Pass 1: fit PCAs and accumulate sums
        for d in rollout_dirs:
            emb = load_embedding(d, stage)
            if emb is None:
                continue

            for level in range(LATENT_LEVELS):
                ipcas[level].partial_fit(emb[level])

            if is_day(d):
                day_sum += emb
                day_count += 1
            else:
                night_sum += emb
                night_count += 1

        logger.info(f"    Fitted on {day_count} day + {night_count} night samples")
        for level in range(LATENT_LEVELS):
            evr = ipcas[level].explained_variance_ratio_
            logger.info(f"    {stage} level {level}: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, PC3={evr[2]:.4f}, total={evr.sum():.4f}")
            if n_components > 3:
                # The top-3 subspace is only well defined if PC3 is separated from PC4
                ev = ipcas[level].explained_variance_
                logger.info(f"    {stage} level {level}: eigengap lambda3/lambda4 = {ev[2] / ev[3]:.3f}")

        # Persist the loadings so subspace comparisons don't need a refit
        loadings[stage] = [p.components_ for p in ipcas]
        np.savez(
            output_dir / f"pca_loadings_{stage}.npz",
            components=np.stack(loadings[stage]),
            evr=np.stack([p.explained_variance_ratio_ for p in ipcas]),
            explained_variance=np.stack([p.explained_variance_ for p in ipcas]),
            mean=np.stack([p.mean_ for p in ipcas]),
        )

        if day_count == 0 or night_count == 0:
            logger.warning(f"    Skipping {stage}: missing day or night data")
            continue

        avg_day = (day_sum / day_count).astype(np.float32)
        avg_night = (night_sum / night_count).astype(np.float32)

        # Pass 2: transform and plot
        for level in range(LATENT_LEVELS):
            day_pcs = ipcas[level].transform(avg_day[level])[:, :3]  # RGB = top 3 PCs
            night_pcs = ipcas[level].transform(avg_night[level])[:, :3]

            # Normalize RGB jointly so day/night are comparable
            joint_min = np.minimum(day_pcs.min(axis=0), night_pcs.min(axis=0))
            joint_max = np.maximum(day_pcs.max(axis=0), night_pcs.max(axis=0))

            for tod, pcs in [("day", day_pcs), ("night", night_pcs)]:
                rgb = np.clip(((pcs - joint_min) / (joint_max - joint_min + 1e-8)).reshape(PATCH_H, PATCH_W, 3), 0, 1)
                title = f"{stage.replace('_', '-')} level {level} — {tod} ({experiment_label})" if titles else ""
                plot_rgb_grid(rgb, title, output_dir / f"{stage}_level{level}_{tod}.png")

    logger.info(f"  [{experiment_label}] Done. Figures saved to {output_dir}")
    return loadings


def report_subspace_overlap(fits: dict[str, dict[str, list[np.ndarray]]], out_dir: Path):
    """Principal angles between the top-3 PC subspaces of each pair of fitted scales.

    Logs the three angles, their mean in degrees, and the mean squared cosine
    (1 = identical subspaces, 0 = orthogonal), and writes them to a CSV.
    """
    names = list(fits)
    pairs = list(combinations(names, 2))
    if not pairs:
        # Only one scale: compare it with itself to exercise the code path (expect 0 deg)
        pairs = [(names[0], names[0])]

    rows = []
    for a, b in pairs:
        for stage in fits[a]:
            for level, (comp_a, comp_b) in enumerate(zip(fits[a][stage], fits[b][stage])):
                # Table S2 compares top-3 subspaces regardless of how many PCs were fit
                theta = subspace_angles(comp_a[:3].T, comp_b[:3].T)
                rows.append({
                    "pair": f"{a}_vs_{b}", "stage": stage, "level": level,
                    "angles_deg": " ".join(f"{t:.2f}" for t in np.degrees(theta)),
                    "mean_angle_deg": float(np.degrees(theta).mean()),
                    "mean_cos2": float((np.cos(theta) ** 2).mean()),
                })
                r = rows[-1]
                logger.info(f"  {r['pair']} {stage} level {level}: angles=[{r['angles_deg']}] deg, "
                            f"mean={r['mean_angle_deg']:.2f} deg, mean cos^2={r['mean_cos2']:.4f}")

    csv_path = out_dir / "subspace_overlap.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Subspace overlap saved to {csv_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--scales", nargs="+", choices=["single", "monthly", "yearly"],
                        default=["single", "monthly", "yearly"])
    parser.add_argument("--n-components", type=int, default=3,
                        help="PCs per latent level; values other than 3 write to pca_embeddings/n{k}/")
    parser.add_argument("--no-title", action="store_true",
                        help="omit figure titles; writes to <scale>_notitle/ folders")
    args = parser.parse_args()
    setup_logging()
    setup_matplotlib_style(12)
    save_dir = SAVE_DIR if args.n_components == 3 else SAVE_DIR / f"n{args.n_components}"

    experiments = {
        # Single day: Sep 1, 2022
        "single": (consts.MAIN_INIT_TIME, datetime(2022, 9, 1, 12), "single_day_20220901", "Sep 1 2022"),
        # Monthly average: September 2022
        "monthly": (datetime(2022, 9, 1, 0), datetime(2022, 9, 30, 12), "monthly_202209", "Sep 2022 avg"),
        # Yearly average: full validation year
        "yearly": (consts.val_start_time, consts.val_end_time, "yearly", "yearly avg"),
    }

    fits = {}
    for scale in args.scales:
        start, end, subdir, label = experiments[scale]
        logger.info(f"=== {scale}: {label} ===")
        if args.no_title:
            subdir += "_notitle"
        fits[scale] = run_pca_experiment(get_rollout_dirs(BASE_DIR, start, end), save_dir / subdir, label,
                                         n_components=args.n_components, titles=not args.no_title)

    logger.info(f"Figures saved under {save_dir}")
    report_subspace_overlap(fits, save_dir)


if __name__ == "__main__":
    main()
