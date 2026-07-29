"""
Compute fraction of variance explained (FVE) for all trained SAEs.
Hooks all 48 layers simultaneously, runs N batches, and reports FVE = 1 - MSE/Var(x).
"""

import csv
import sys
import logging

import torch
from torch.utils.data import DataLoader

from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.interp.aurora_sae import AuroraSAE
from mirora.config import STAGES
from mirora.interp.train_sae import aurora_collate_fn
from aurora import AuroraAirPollution

import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

NUM_BATCHES = 20

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_dir = consts.MODEL_PATH / "sae"

    # Load Aurora
    logger.info("Loading Aurora model...")
    model = AuroraAirPollution()
    model.load_checkpoint_local(str(consts.CHECKPOINT_PATH))
    model = model.to(device).eval()

    # Load all SAEs
    logger.info("Loading all SAEs...")
    saes = {}
    for idx, (target_decoder, stage_idx, block_idx, dim) in enumerate(STAGES):
        component = "decoder" if target_decoder else "encoder"
        path = sae_dir / f"sae_{component}_s{stage_idx}_b{block_idx}.pt"
        if not path.exists():
            logger.warning(f"SAE not found: {path}")
            continue
        sae = AuroraSAE(d_model=dim, expansion_factor=8)
        sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        sae = sae.to(device).eval()
        saes[idx] = sae

    # Accumulators: running sum, sum of squares, MSE, and count per layer
    stats = {idx: {"sum_var": 0.0, "mse_sum": 0.0, "l0_sum": 0.0, "n": 0} for idx in saes}

    # Register hooks on all layers
    handles = []
    captured = {}

    def make_hook(idx):
        def hook_fn(_module, _input, output):
            x = output[0]
            captured[idx] = x.reshape(-1, x.shape[-1]).detach()
        return hook_fn

    for idx, (target_decoder, stage_idx, block_idx, _dim) in enumerate(STAGES):
        if idx not in saes:
            continue
        if target_decoder:
            target_layer = model.backbone.decoder_layers[stage_idx].blocks[block_idx]  # type: ignore[index]
        else:
            target_layer = model.backbone.encoder_layers[stage_idx].blocks[block_idx]  # type: ignore[index]
        handles.append(target_layer.register_forward_hook(make_hook(idx)))

    # DataLoader
    dataset = AuroraInitBatchesDataset(
        save_dir=consts.BATCH_INITS_PATH,
        start=consts.val_start_time,
        end=consts.val_end_time,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=aurora_collate_fn, num_workers=4,
    )

    # Run inference
    logger.info(f"Running {NUM_BATCHES} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= NUM_BATCHES:
                break

            captured.clear()
            batch = batch.to(device)
            _ = model(batch)

            for idx, sae in saes.items():
                if idx not in captured:
                    continue
                x = captured[idx]
                x_hat, h = sae(x)

                # Variance of x (across all tokens and dimensions)
                stats[idx]["sum_var"] += x.var().item()
                stats[idx]["mse_sum"] += torch.nn.functional.mse_loss(x_hat, x).item()
                stats[idx]["l0_sum"] += (h > 0).float().sum(dim=-1).mean().item()
                stats[idx]["n"] += 1

            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}/{NUM_BATCHES}")

    # Remove hooks
    for h in handles:
        h.remove()

    # Report results
    print("\n" + "=" * 95)
    print(f"{'Layer':<30} {'Var(x)':>10} {'MSE':>10} {'FVE':>10} {'FVE %':>10} {'L0':>8}")
    print("=" * 95)

    csv_path = consts.FIGURES_PATH / "fve_l1_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, (target_decoder, stage_idx, block_idx, dim) in enumerate(STAGES):
        if idx not in saes or stats[idx]["n"] == 0:
            continue
        component = "decoder" if target_decoder else "encoder"
        n = stats[idx]["n"]
        avg_var = stats[idx]["sum_var"] / n
        avg_mse = stats[idx]["mse_sum"] / n
        fve = 1.0 - avg_mse / avg_var if avg_var > 0 else float("nan")
        avg_l0 = stats[idx]["l0_sum"] / n

        print(f"{component} s{stage_idx} b{block_idx} (dim={dim})".ljust(30)
              + f"{avg_var:>10.4f} {avg_mse:>10.6f} {fve:>10.6f} {fve*100:>9.2f}% {avg_l0:>8.1f}")

        rows.append({
            "component": component, "stage": stage_idx, "block": block_idx,
            "dim": dim, "method": "l1", "var": avg_var, "mse": avg_mse,
            "fve": fve, "l0": avg_l0,
        })

    print("=" * 95)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
