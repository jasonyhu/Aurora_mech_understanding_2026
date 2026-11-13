"""
Multi-initialization robustness sweep for the +5 ppb NOx perturbation: do the
single-init Sep 1 2022 statistics hold across initializations and seasons?

For every available init batch on OAK, this runs a base rollout and a
+5 ppb NOx rollout IN-MEMORY (no giant predictions.pickle written — only a
small per-task CSV of surface-O3 statistics is saved), then reports, per lead
time, both raw and cosine-of-latitude area-weighted versions of:

  - median(base), median(pert), diff-of-medians, median(pert - base)
  - mean(pert - base), mean|pert - base|
  - fraction of cells with O3 suppressed (p < b)
  - fraction of cells with perturbed O3 < 0
  - tropical-band (|lat| < 23.5 deg) median ΔO3 and fraction increased

The model checkpoint is loaded ONCE per task and reused across all
assigned inits. Aggregate partial CSVs with aggregate_nox_robustness.py.

USAGE: python3 run_nox_robustness_sweep.py <task_id> <task_ct>
"""

import copy
import csv
import logging
import sys

import numpy as np
import torch
from aurora import rollout

from mirora.config import MMR_TO_PPB
from mirora.data.loading import load_batch, surface
from mirora.experiments.perturbation import perturb_batch
from mirora.inference.inference import load_aurora
from mirora.stats import lat_weights, wmean
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)


PPB = 5.0                 # perturbation magnitude
ROLLOUT_STEPS = 10        # 12h .. 120h
TROPICS_DEG = 23.5

OUT_DIR = consts.NOX_ANALYSIS_PATH / "nox_robustness"

FIELDS = [
    "init_time", "step", "lead_hours",
    "median_base", "median_pert", "diff_of_medians", "median_of_diff",
    "mean_diff", "mean_absdiff",
    "wmean_diff", "wmean_absdiff",
    "frac_suppressed_raw", "frac_suppressed_areaw",
    "frac_neg_raw", "frac_neg_areaw", "frac_neg_base_raw",
    "min_pert", "n_cells",
    "trop_median_of_diff", "trop_frac_increased_areaw", "trop_frac_suppressed_areaw",
]


def run_rollout(model, batch):
    with torch.inference_mode():
        return [p.to("cpu") for p in rollout(model, batch, steps=ROLLOUT_STEPS)]


def o3_field(preds, step):
    return surface(preds[step], "go3").astype(np.float64) * MMR_TO_PPB["go3"]


def stats_row(base_preds, pert_preds, w_row, lat, step, init_str):
    b = o3_field(base_preds, step)
    p = o3_field(pert_preds, step)
    d = p - b
    trop = np.abs(lat) < TROPICS_DEG          # [H]
    trop2d = np.broadcast_to(trop[:, None], d.shape)
    w_trop = np.where(trop, w_row, 0.0)
    return {
        "init_time": init_str,
        "step": step,
        "lead_hours": (step + 1) * 12,
        "median_base": float(np.median(b)),
        "median_pert": float(np.median(p)),
        "diff_of_medians": float(np.median(p) - np.median(b)),
        "median_of_diff": float(np.median(d)),
        "mean_diff": float(np.mean(d)),
        "mean_absdiff": float(np.mean(np.abs(d))),
        "wmean_diff": wmean(d, w_row),
        "wmean_absdiff": wmean(np.abs(d), w_row),
        "frac_suppressed_raw": float(np.mean(p < b)),
        "frac_suppressed_areaw": wmean((p < b).astype(np.float64), w_row),
        "frac_neg_raw": float(np.mean(p < 0)),
        "frac_neg_areaw": wmean((p < 0).astype(np.float64), w_row),
        "frac_neg_base_raw": float(np.mean(b < 0)),
        "min_pert": float(p.min()),
        "n_cells": int(b.size),
        "trop_median_of_diff": float(np.median(d[trop2d])) if trop.any() else float("nan"),
        "trop_frac_increased_areaw": wmean((p > b).astype(np.float64), w_trop) if w_trop.sum() > 0 else float("nan"),
        "trop_frac_suppressed_areaw": wmean((p < b).astype(np.float64), w_trop) if w_trop.sum() > 0 else float("nan"),
    }


def main():
    setup_logging()
    assert len(sys.argv) == 3, "USAGE: python3 run_nox_robustness_sweep.py <task_id> <task_ct>"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])
    init_times = consts.val_init_times()
    mine = init_times[task_id::task_ct]
    logger.info(f"Task {task_id}/{task_ct}: {len(mine)} inits (of {len(init_times)} total), "
                f"convention={consts.NOX_CONVENTION}")

    consts.ensure_checkpoint()
    model, device = load_aurora(consts.CHECKPOINT_PATH)
    logger.info(f"Loaded model onto {device}")

    rows = []
    for init_time in mine:
        init_str = init_time.strftime(r"%Y%m%d_%H%M")
        try:
            base_batch = load_batch(consts.BATCH_INITS_PATH, init_time)
        except Exception as e:
            logger.warning(f"[skip] {init_str}: cannot load batch ({e})")
            continue

        logger.info(f"Rolling out {init_str} (base + {PPB}ppb) ...")
        pert_batch = copy.deepcopy(base_batch)
        perturb_batch(pert_batch, atmos_vars=consts.nox_offsets(PPB))

        base_preds = run_rollout(model, base_batch)
        pert_preds = run_rollout(model, pert_batch)

        w_row = lat_weights(base_preds)
        lat = base_preds[0].metadata.lat
        if isinstance(lat, torch.Tensor):
            lat = lat.cpu().numpy()
        lat = np.asarray(lat, dtype=np.float64)

        for step in range(ROLLOUT_STEPS):
            rows.append(stats_row(base_preds, pert_preds, w_row, lat, step, init_str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"nox_robustness_partial_{task_id:03d}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
