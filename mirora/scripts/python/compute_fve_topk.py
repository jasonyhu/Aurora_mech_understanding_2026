"""
Compute FVE and L0 sparsity for all trained BatchTopK SAEs.
Saves results to CSV for comparison figure generation.
"""

import csv
import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.interp.batch_topk_sae import BatchTopKSAE
from mirora.config import STAGES
from mirora.interp.train_sae import aurora_collate_fn
from aurora import AuroraAirPollution

import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

NUM_BATCHES = 20
EXPANSION_FACTOR = 8


def find_topk_sae(sae_dir: Path, component: str, stage_idx: int, block_idx: int) -> tuple[Path, int] | None:
    """Find a BatchTopK SAE checkpoint and extract k from the filename."""
    pattern = f"sae_topk_{component}_s{stage_idx}_b{block_idx}_x{EXPANSION_FACTOR}_k*.pt"
    matches = list(sae_dir.glob(pattern))
    # Exclude training checkpoints
    matches = [m for m in matches if "epoch" not in m.name]
    if not matches:
        return None
    path = matches[0]
    # Extract k from filename: ...k{k}.pt
    k = int(path.stem.split("_k")[-1])
    return path, k


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_dir = consts.MODEL_PATH / "sae_topk"

    logger.info("Loading Aurora model...")
    model = AuroraAirPollution()
    model.load_checkpoint_local(str(consts.CHECKPOINT_PATH))
    model = model.to(device).eval()

    logger.info("Loading BatchTopK SAEs...")
    saes: dict[int, tuple[BatchTopKSAE, int]] = {}
    for idx, (target_decoder, stage_idx, block_idx, dim) in enumerate(STAGES):
        component = "decoder" if target_decoder else "encoder"
        result = find_topk_sae(sae_dir, component, stage_idx, block_idx)
        if result is None:
            continue
        path, k = result
        sae = BatchTopKSAE(d_model=dim, expansion_factor=EXPANSION_FACTOR, k=k)
        sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        sae = sae.to(device).eval()
        saes[idx] = (sae, k)

    if not saes:
        logger.error("No BatchTopK SAEs found. Run training first.")
        sys.exit(1)

    logger.info(f"Loaded {len(saes)} SAEs")

    stats: dict[int, dict] = {idx: {"sum_var": 0.0, "mse_sum": 0.0, "l0_sum": 0.0, "n": 0} for idx in saes}

    handles = []
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook_fn(_module, _input, output):
            x = output[0]
            captured[idx] = x.reshape(-1, x.shape[-1]).detach()
        return hook_fn

    for idx, (target_decoder, stage_idx, block_idx, _dim) in enumerate(STAGES):
        if idx not in saes:
            continue
        if target_decoder:
            layer = model.backbone.decoder_layers[stage_idx].blocks[block_idx]  # type: ignore[index]
        else:
            layer = model.backbone.encoder_layers[stage_idx].blocks[block_idx]  # type: ignore[index]
        handles.append(layer.register_forward_hook(make_hook(idx)))

    dataset = AuroraInitBatchesDataset(
        save_dir=consts.BATCH_INITS_PATH,
        start=consts.val_start_time, end=consts.val_end_time,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=aurora_collate_fn, num_workers=4)

    logger.info(f"Running {NUM_BATCHES} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= NUM_BATCHES:
                break
            captured.clear()
            batch = batch.to(device)
            _ = model(batch)

            for idx, (sae, _k) in saes.items():
                if idx not in captured:
                    continue
                x = captured[idx]
                x_hat, h, aux = sae(x, training=False)

                stats[idx]["sum_var"] += x.var().item()
                stats[idx]["mse_sum"] += torch.nn.functional.mse_loss(x_hat, x).item()
                stats[idx]["l0_sum"] += (h > 0).float().sum(dim=-1).mean().item()
                stats[idx]["n"] += 1

            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}/{NUM_BATCHES}")

    for h in handles:
        h.remove()

    # Print and save results
    print("\n" + "=" * 100)
    print(f"{'Layer':<32} {'k':>6} {'Var(x)':>10} {'MSE':>10} {'FVE':>10} {'FVE %':>10} {'L0':>8}")
    print("=" * 100)

    csv_path = consts.FIGURES_PATH / "fve_topk_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, (target_decoder, stage_idx, block_idx, dim) in enumerate(STAGES):
        if idx not in saes or stats[idx]["n"] == 0:
            continue
        component = "decoder" if target_decoder else "encoder"
        _, k = saes[idx]
        n = stats[idx]["n"]
        avg_var = stats[idx]["sum_var"] / n
        avg_mse = stats[idx]["mse_sum"] / n
        avg_l0 = stats[idx]["l0_sum"] / n
        fve = 1.0 - avg_mse / avg_var if avg_var > 0 else float("nan")

        label = f"{component} s{stage_idx} b{block_idx} (dim={dim})"
        print(f"{label:<32} {k:>6} {avg_var:>10.4f} {avg_mse:>10.6f} {fve:>10.6f} {fve*100:>9.2f}% {avg_l0:>8.1f}")
        rows.append({
            "component": component, "stage": stage_idx, "block": block_idx,
            "dim": dim, "k": k, "method": "topk",
            "var": avg_var, "mse": avg_mse, "fve": fve, "l0": avg_l0,
        })

    print("=" * 100)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
