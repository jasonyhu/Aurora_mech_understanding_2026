"""
Compute fraction of initial NO perturbation remaining at 1000 hPa,
cosine-of-latitude-weighted global mean, for the +0.5/+5/+20 ppb NOx
perturbation experiments. Sampled every 10 days across the validation year.

Writes one partial CSV per parallel task; aggregate with
aggregate_perturbation_decay.py.

USAGE: python3 compute_perturbation_decay.py <task_id> <task_ct>
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np

from mirora.data.loading import fetch_rollout, surface
from mirora.stats import lat_weights, wmean
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_DAYS = 10
OUT_DIR = consts.NOX_ANALYSIS_PATH / "perturbation_decay"


def load_no_l0(pickle_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (surface NO with shape [n_steps, H, W], cos-lat row weights [H])."""
    rollout = fetch_rollout(pickle_path)
    no = np.stack([surface(s, "no") for s in rollout], axis=0)
    return no, lat_weights(rollout)


def main() -> None:
    setup_logging()
    assert len(sys.argv) == 3, "USAGE: python3 compute_perturbation_decay.py <task_id> <task_ct>"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])
    init_times = consts.val_init_times()[::2 * SAMPLE_INTERVAL_DAYS]  # inits are 12-hourly
    my_inits = init_times[task_id::task_ct]
    logger.info(f"Task {task_id}/{task_ct}: {len(my_inits)} init times "
                f"(of {len(init_times)} total)")

    rows = []
    for init_time in my_inits:
        time_str = init_time.strftime(r"%Y%m%d_%H%M")
        base_path = consts.base_rollout_path(init_time)
        if not base_path.exists():
            logger.warning(f"Missing base rollout at {base_path}, skipping")
            continue

        logger.info(f"Loading base rollout {time_str}")
        base_no, w_row = load_no_l0(base_path)

        for ppb in consts.NOX_MAGNITUDES_PPB:
            pert_path = consts.rollout_path(consts.nox_experiment_dir(ppb), init_time)
            if not pert_path.exists():
                logger.warning(f"Missing perturbed rollout at {pert_path}, skipping")
                continue

            logger.info(f"Loading +{ppb}ppb rollout {time_str}")
            pert_no, _ = load_no_l0(pert_path)

            diff = pert_no - base_no                         # [n_steps, H, W]
            fraction = [wmean(d, w_row) / consts.nox_offsets(ppb)["no"] for d in diff]

            for step_idx, frac in enumerate(fraction):
                rows.append({
                    "init_time": time_str,
                    "ppb": ppb,
                    "step": step_idx,
                    "lead_hours": (step_idx + 1) * 12,
                    "fraction_remaining": float(frac),
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"decay_partial_{task_id:02d}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["init_time", "ppb", "step", "lead_hours", "fraction_remaining"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
