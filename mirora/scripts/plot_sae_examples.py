"""
Generate a 6×2 figure of L1 SAE activation examples on the Aurora backbone
(1 Nov 2022 00Z init).

Rows: first block of each backbone stage (encoder s0/s1/s2, decoder s0/s1/s2)
Columns:
  Left:  arbitrary feature (index 1)
  Right: top-activating feature of that block

Font sizes ≥20.
"""

import logging
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from mirora.config import DEFAULT_EXPANSION_FACTOR, STAGES
from mirora.data.loading import load_batch
from mirora.inference.inference import load_aurora
from mirora.interp import capture_blocks, get_spatial_resolution, load_sae
from mirora.utils import setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style

import consts

logger = logging.getLogger(__name__)

RANDOM_FEATURE_IDX = 1  # arbitrary "random" feature

# First block of each stage: encoder s0/s1/s2, then decoder s0/s1/s2.
ROWS = [(component, stage, block, dim) for component, stage, block, dim in STAGES if block == 0]


def main():
    setup_logging()
    setup_matplotlib_style(20)
    model, device = load_aurora(consts.CHECKPOINT_PATH)
    batch = load_batch(consts.BATCH_INITS_PATH, datetime(2022, 11, 1, 0)).to(device)

    logger.info("Running forward pass...")
    with torch.no_grad(), capture_blocks(model, [r[:3] for r in ROWS]) as captured:
        model(batch)

    spatial_results = []
    for component, stage_idx, block_idx, dim in ROWS:
        sae = load_sae(consts.SAE_DIR, component, stage_idx, block_idx, dim, device)
        x = captured[(component, stage_idx, block_idx)]
        flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            h = sae.encode(flat)  # (tokens, d_hidden)

        C, H, W = get_spatial_resolution(model, batch, component, stage_idx)
        spatial = h.reshape(C, H, W, -1).mean(dim=0).cpu().numpy()  # (H, W, d_hidden)

        # Block-level ranking (not per-token var_contribution): a feature's total energy
        # over all tokens / the block's total centred variance, sum_t h[t,k]² / sum_t ||x[t] − b_dec||².
        with torch.no_grad():
            total_var = ((flat - sae.b_dec) ** 2).sum().item()
            contrib_per_feat = ((h ** 2).sum(dim=0) / total_var).cpu()
        top_idx = int(torch.argmax(contrib_per_feat).item())
        logger.info(
            f"  {component} s{stage_idx} b{block_idx}: top feature #{top_idx} "
            f"(contrib={float(contrib_per_feat[top_idx]):.4f}); random feature #{RANDOM_FEATURE_IDX} "
            f"contrib={float(contrib_per_feat[RANDOM_FEATURE_IDX]):.4f}"
        )

        spatial_results.append({
            "d_hidden": dim * DEFAULT_EXPANSION_FACTOR,
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
            im = draw_global_field(ax, feat_map, "inferno", coast_color="white", coast_width=0.5)
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

    logger.info(f"Saved to {save_path_pdf} and {save_path_png}")


if __name__ == "__main__":
    main()
