"""
Table 1: fraction of variance explained (FVE = 1 - MSE/Var(x)) and L0 of the 48
AuroraScope SAEs. Hooks all 48 blocks at once, runs the first NUM_BATCHES
validation inits, and writes FIGURES_PATH/fve_l1_results.csv (one row per block;
Table 1 reports the per-stage mean and range).

USAGE: python3 table1_compute_sae_fve.py
"""

import csv
import logging
import sys

import torch

from mirora.config import STAGES
from mirora.data.loading import load_batch
from mirora.inference.inference import load_aurora
from mirora.interp import capture_blocks, load_sae, sae_path
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

NUM_BATCHES = 20


def load_saes(device):
    """{STAGES index: sae} for every block with a trained SAE."""
    saes = {}
    for idx, (component, stage_idx, block_idx, dim) in enumerate(STAGES):
        path = sae_path(consts.SAE_DIR, component, stage_idx, block_idx)
        if not path.exists():
            logger.warning(f"SAE not found: {path}")
            continue
        saes[idx] = load_sae(consts.SAE_DIR, component, stage_idx, block_idx, dim, device)
    return saes


def main():
    setup_logging()
    logger.info("Loading Aurora model...")
    model, device = load_aurora(consts.CHECKPOINT_PATH)

    logger.info("Loading SAEs...")
    saes = load_saes(device)
    if not saes:
        logger.error("No SAEs found. Run training (or download_aurorascope_saes.py) first.")
        sys.exit(1)

    stats = {idx: {"sum_var": 0.0, "mse_sum": 0.0, "l0_sum": 0.0, "n": 0} for idx in saes}
    keys = {idx: STAGES[idx][:3] for idx in saes}

    logger.info(f"Running {NUM_BATCHES} batches...")
    with capture_blocks(model, list(keys.values())) as captured, torch.no_grad():
        for batch_idx, init_time in enumerate(consts.val_init_times()[:NUM_BATCHES]):
            model(load_batch(consts.BATCH_INITS_PATH, init_time).to(device))

            for idx, sae in saes.items():
                x = captured[keys[idx]]
                x = x.reshape(-1, x.shape[-1])
                x_hat, h = sae(x)

                stats[idx]["sum_var"] += x.var().item()
                stats[idx]["mse_sum"] += torch.nn.functional.mse_loss(x_hat, x).item()
                stats[idx]["l0_sum"] += (h > 0).float().sum(dim=-1).mean().item()
                stats[idx]["n"] += 1

            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}/{NUM_BATCHES}")

    width = 95
    print("\n" + "=" * width)
    print(f"{'Layer':<32} {'Var(x)':>10} {'MSE':>10} {'FVE':>10} {'FVE %':>10} {'L0':>8}")
    print("=" * width)

    rows = []
    for idx, (component, stage_idx, block_idx, dim) in enumerate(STAGES):
        if idx not in saes or stats[idx]["n"] == 0:
            continue
        n = stats[idx]["n"]
        avg_var = stats[idx]["sum_var"] / n
        avg_mse = stats[idx]["mse_sum"] / n
        fve = 1.0 - avg_mse / avg_var if avg_var > 0 else float("nan")
        avg_l0 = stats[idx]["l0_sum"] / n

        print(f"{component} s{stage_idx} b{block_idx} (dim={dim})".ljust(32)
              + f" {avg_var:>10.4f} {avg_mse:>10.6f} {fve:>10.6f} {fve*100:>9.2f}% {avg_l0:>8.1f}")

        rows.append({"component": component, "stage": stage_idx, "block": block_idx, "dim": dim,
                     "method": "l1", "var": avg_var, "mse": avg_mse, "fve": fve, "l0": avg_l0})

    print("=" * width)

    csv_path = consts.FIGURES_PATH / "fve_l1_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
