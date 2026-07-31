"""
Generate a 6×2 figure of L1 SAE activation examples on the Aurora backbone.

Rows: first block of each backbone stage (encoder s0/s1/s2, decoder s0/s1/s2)
Columns:
  Left:  top-activating feature of that block
  Right: arbitrary feature (index 1)

No titles, font sizes ≥20.
"""

import pickle
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from torch.utils.data import DataLoader

from aurora import AuroraAirPollution
from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.interp.aurora_sae import AuroraSAE

import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 20,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

SAE_DIR = consts.MODEL_PATH / "sae"
EXPANSION = 8
RANDOM_FEATURE_IDX = 1  # arbitrary "random" feature

# (target_decoder, stage_idx, block_idx, dim, label)
ROWS = [
    (False, 0, 0, 512,  "Encoder S0 B0"),
    (False, 1, 0, 1024, "Encoder S1 B0"),
    (False, 2, 0, 2048, "Encoder S2 B0"),
    (True,  0, 0, 2048, "Decoder S0 B0"),
    (True,  1, 0, 1024, "Decoder S1 B0"),
    (True,  2, 0, 512,  "Decoder S2 B0"),
]


def get_spatial_resolution(model, batch, target_decoder: bool, stage_idx: int) -> tuple[int, int, int]:
    H_img = batch.metadata.lat.shape[0]
    W_img = batch.metadata.lon.shape[0]
    patch_res = (model.encoder.latent_levels, H_img // model.patch_size, W_img // model.patch_size)
    all_res, _ = model.backbone.get_encoder_specs(patch_res)
    num_stages = len(all_res)
    res_idx = (num_stages - stage_idx - 1) if target_decoder else stage_idx
    return all_res[res_idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Aurora model...")
    model = AuroraAirPollution()
    model.load_checkpoint_local(str(consts.CHECKPOINT_PATH))
    model = model.to(device).eval()

    logger.info("Loading batch...")
    batch_date = datetime(2022, 11, 1, 0)
    dataset = AuroraInitBatchesDataset(consts.BATCH_INITS_PATH, start=batch_date, end=batch_date)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batch = next(iter(dataloader)).to(device)

    # Hook all 6 target blocks at once
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook_fn(_m, _i, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[idx] = x.detach()
        return hook_fn

    for idx, (target_decoder, stage_idx, block_idx, _dim, _lbl) in enumerate(ROWS):
        if target_decoder:
            tgt = model.backbone.decoder_layers[stage_idx].blocks[block_idx]
        else:
            tgt = model.backbone.encoder_layers[stage_idx].blocks[block_idx]
        handles.append(tgt.register_forward_hook(make_hook(idx)))

    logger.info("Running forward pass...")
    with torch.no_grad():
        model(batch)

    for h in handles:
        h.remove()

    # Process each row: load SAE, extract features
    spatial_results = []
    for idx, (target_decoder, stage_idx, block_idx, dim, label) in enumerate(ROWS):
        component = "decoder" if target_decoder else "encoder"
        sae_path = SAE_DIR / f"sae_{component}_s{stage_idx}_b{block_idx}.pt"
        logger.info(f"Loading {sae_path.name}...")
        sae = AuroraSAE(d_model=dim, expansion_factor=EXPANSION)
        sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
        sae = sae.to(device).eval()

        x = captured[idx]  # (B, L, D)
        D = x.shape[-1]
        flat = x.reshape(-1, D)
        with torch.no_grad():
            _, h = sae(flat)
        # h: (tokens, d_hidden)

        # Spatial resolution
        C, H, W = get_spatial_resolution(model, batch, target_decoder, stage_idx)
        spatial = h.reshape(C, H, W, -1).mean(dim=0).cpu().numpy()  # (H, W, d_hidden)

        # Top feature by per-feature variance contribution
        # contrib_k = ||h[:, k] · W_dec[k, :]||² / ||x − b_dec||²
        # Since W_dec rows are unit-normalized: ||h[:,k] · W_dec[k,:]||² = ||h[:,k]||²
        # So contrib_k = sum_t h[t,k]² / ||x − b_dec||²
        with torch.no_grad():
            x_centered = flat - sae.b_dec
            total_var = (x_centered ** 2).sum().item()
            energy_per_feat = (h ** 2).sum(dim=0)  # (d_hidden,)
            contrib_per_feat = (energy_per_feat / total_var).cpu()
        top_idx = int(torch.argmax(contrib_per_feat).item())
        top_contrib = float(contrib_per_feat[top_idx].item())
        rand_contrib = float(contrib_per_feat[RANDOM_FEATURE_IDX].item())
        logger.info(
            f"  Top feature: #{top_idx} (contrib={top_contrib:.4f}); "
            f"random feature #{RANDOM_FEATURE_IDX} contrib={rand_contrib:.4f}"
        )

        d_hidden = dim * EXPANSION
        spatial_results.append({
            "label": label,
            "d_hidden": d_hidden,
            "top_idx": top_idx,
            "rand_idx": RANDOM_FEATURE_IDX,
            "top_map": spatial[:, :, top_idx],
            "rand_map": spatial[:, :, RANDOM_FEATURE_IDX],
        })

        del sae, h
        torch.cuda.empty_cache()

    # Plot 6×2 grid
    logger.info("Building figure...")
    fig, axes = plt.subplots(
        6, 2, figsize=(16, 18),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)},
    )

    for row, info in enumerate(spatial_results):
        d_hidden = info["d_hidden"]
        # Left column: random feature (#1). Right column: top-activating feature.
        for col, (feat_map, feat_idx) in enumerate([
            (info["rand_map"], info["rand_idx"]),
            (info["top_map"], info["top_idx"]),
        ]):
            ax = axes[row, col]
            rolled = np.roll(feat_map, feat_map.shape[1] // 2, axis=1)
            im = ax.imshow(
                rolled, extent=[-180, 180, -90, 90], origin="upper",
                transform=ccrs.PlateCarree(), cmap="inferno", interpolation="nearest",
            )
            ax.coastlines(color="white", linewidth=0.5)
            ax.set_global()
            ax.set_title(f"Feature #{feat_idx} of {d_hidden}", fontsize=20, pad=8)
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=15)
            cbar.ax.tick_params(labelsize=20)
            cbar.set_label("SAE activation", fontsize=20, labelpad=8)

    plt.subplots_adjust(left=0.02, right=0.96, top=0.96, bottom=0.02, wspace=0.025, hspace=0.18)

    save_dir = consts.FIGURES_PATH / "sae_examples"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path_pdf = save_dir / "sae_examples_6x2.pdf"
    save_path_png = save_dir / "sae_examples_6x2.png"
    plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"Saved to {save_path_pdf}")
    logger.info(f"Saved to {save_path_png}")
    for r in spatial_results:
        logger.info(f"  {r['label']}: top feature = #{r['top_idx']}, random feature = #{r['rand_idx']}")


if __name__ == "__main__":
    main()
