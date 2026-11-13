"""
Fraction of variance explained (FVE = 1 - MSE/Var(x)) and L0 of all trained SAEs.
Hooks all 48 blocks at once, runs the first NUM_BATCHES validation inits, and
writes FIGURES_PATH/fve_{method}_results.csv.

USAGE: python3 compute_fve.py [--method {l1,topk}]
"""

import argparse
import csv
import logging
import sys

import torch

from mirora.config import DEFAULT_EXPANSION_FACTOR, STAGES
from mirora.data.loading import load_batch
from mirora.inference.inference import load_aurora
from mirora.interp import BatchTopKSAE, capture_blocks, find_topk_checkpoint, load_sae, sae_path
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

NUM_BATCHES = 20


def load_saes(method, device):
    """{STAGES index: (sae, k)} for every block with a trained SAE (k is None for L1)."""
    saes = {}
    for idx, (component, stage_idx, block_idx, dim) in enumerate(STAGES):
        if method == "topk":
            found = find_topk_checkpoint(consts.MODEL_PATH / "sae_topk", component, stage_idx, block_idx)
            if found is None:
                continue
            path, k = found
            sae = BatchTopKSAE(d_model=dim, expansion_factor=DEFAULT_EXPANSION_FACTOR, k=k)
            sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            saes[idx] = (sae.to(device).eval(), k)
        else:
            path = sae_path(consts.SAE_DIR, component, stage_idx, block_idx)
            if not path.exists():
                logger.warning(f"SAE not found: {path}")
                continue
            saes[idx] = (load_sae(consts.SAE_DIR, component, stage_idx, block_idx, dim, device), None)
    return saes


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["l1", "topk"], default="l1")
    args = parser.parse_args()

    logger.info("Loading Aurora model...")
    model, device = load_aurora(consts.CHECKPOINT_PATH)

    logger.info(f"Loading {args.method} SAEs...")
    saes = load_saes(args.method, device)
    if not saes:
        logger.error(f"No {args.method} SAEs found. Run training first.")
        sys.exit(1)

    stats = {idx: {"sum_var": 0.0, "mse_sum": 0.0, "l0_sum": 0.0, "n": 0} for idx in saes}
    keys = {idx: STAGES[idx][:3] for idx in saes}

    logger.info(f"Running {NUM_BATCHES} batches...")
    with capture_blocks(model, list(keys.values())) as captured, torch.no_grad():
        for batch_idx, init_time in enumerate(consts.val_init_times()[:NUM_BATCHES]):
            model(load_batch(consts.BATCH_INITS_PATH, init_time).to(device))

            for idx, (sae, _) in saes.items():
                x = captured[keys[idx]]
                x = x.reshape(-1, x.shape[-1])
                if args.method == "topk":
                    x_hat, h, _ = sae(x, training=False)
                else:
                    x_hat, h = sae(x)

                stats[idx]["sum_var"] += x.var().item()
                stats[idx]["mse_sum"] += torch.nn.functional.mse_loss(x_hat, x).item()
                stats[idx]["l0_sum"] += (h > 0).float().sum(dim=-1).mean().item()
                stats[idx]["n"] += 1

            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}/{NUM_BATCHES}")

    width = 100 if args.method == "topk" else 95
    k_col = f" {'k':>6}" if args.method == "topk" else ""
    print("\n" + "=" * width)
    print(f"{'Layer':<32}{k_col} {'Var(x)':>10} {'MSE':>10} {'FVE':>10} {'FVE %':>10} {'L0':>8}")
    print("=" * width)

    rows = []
    for idx, (component, stage_idx, block_idx, dim) in enumerate(STAGES):
        if idx not in saes or stats[idx]["n"] == 0:
            continue
        k = saes[idx][1]
        n = stats[idx]["n"]
        avg_var = stats[idx]["sum_var"] / n
        avg_mse = stats[idx]["mse_sum"] / n
        fve = 1.0 - avg_mse / avg_var if avg_var > 0 else float("nan")
        avg_l0 = stats[idx]["l0_sum"] / n

        k_val = f" {k:>6}" if k is not None else ""
        print(f"{component} s{stage_idx} b{block_idx} (dim={dim})".ljust(32)
              + f"{k_val} {avg_var:>10.4f} {avg_mse:>10.6f} {fve:>10.6f} {fve*100:>9.2f}% {avg_l0:>8.1f}")

        row = {"component": component, "stage": stage_idx, "block": block_idx, "dim": dim}
        if k is not None:
            row["k"] = k
        rows.append({**row, "method": args.method, "var": avg_var, "mse": avg_mse, "fve": fve, "l0": avg_l0})

    print("=" * width)

    csv_path = consts.FIGURES_PATH / f"fve_{args.method}_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
