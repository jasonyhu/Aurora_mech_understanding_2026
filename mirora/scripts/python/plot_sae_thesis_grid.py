"""
Generate 9 individual SAE feature visualization PDFs for thesis composite figure.

Each PDF shows one specific feature from a specific backbone block. The user
will tile them in a 3x3 grid in Illustrator manually.

Activations are normalized per-feature to [0, 1] using the feature's max
activation on the running example (Sep 1 2022 00Z).

Also saves `plot_data.pickle` (list of dicts) so the spatial maps can be
re-plotted on CPU without rerunning the model on GPU.
"""

import math
import pickle
import sys
import logging
from datetime import datetime

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
    "font.family": "sans-serif",
    "font.size": 22,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

SAE_DIR = consts.MODEL_PATH / "sae"
EXPANSION = 8
OUT_DIR = consts.FIGURES_PATH / "sae_thesis_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot specs: (plot_num, component, stage_0idx, block_0idx, dim, feature_idx)
# 2x2 layout (plot_num = row*2 + col + 1). plot_num reflects reading order
# (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right). The trailing comment
# notes pipeline order (Enc S0 → Enc S1 → Dec S0 → Dec S2).
GRID_NCOLS = 2
PLOTS = [
    (1, "encoder", 1, 0, 1024, 6873),    # top-left  — transport       (pipeline 2)
    (2, "encoder", 0, 3, 512,  1145),    # top-right — activates everywhere (pipeline 1)
    (3, "decoder", 2, 4, 512,  1421),    # bot-left  — day/night + transport (pipeline 4)
    (4, "decoder", 0, 6, 2048, 14566),   # bot-right — latitude stripe (pipeline 3)
]

# Legacy 3x3 layout (kept for reference; uncomment above and swap to use):
# GRID_NCOLS = 3
# PLOTS = [
#     (1, "encoder", 0, 0, 512,  3191),   # strongest activator
#     (2, "encoder", 1, 0, 1024, 6873),   # transport
#     (3, "encoder", 1, 1, 1024, 6883),   # South-America plume
#     (4, "decoder", 2, 4, 512,  1421),   # day/night + transport
#     (5, "encoder", 0, 3, 512,  1145),   # activates everywhere
#     (6, "decoder", 1, 0, 1024, 6137),   # day/night only
#     (7, "encoder", 2, 4, 2048, 10870),  # Antarctic
#     (8, "decoder", 0, 0, 2048, 1),      # normal / pixelated random
#     (9, "decoder", 0, 6, 2048, 14566),  # latitude stripe
# ]


def nice_ticks_4(peak: float) -> np.ndarray:
    """Return 4 nice tick values [0, s, 2s, 3s] with 3s >= peak and s in {1,2,2.5,5}×10^k."""
    if peak <= 0:
        return np.array([0.0, 1 / 3, 2 / 3, 1.0])
    target = peak / 3.0
    exp = math.floor(math.log10(target))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * 10 ** exp
        if 3 * step >= peak - 1e-12:
            return np.array([0.0, step, 2 * step, 3 * step])
    return np.array([0.0, peak / 3, 2 * peak / 3, peak])


def get_spatial_resolution(model, batch, target_decoder: bool, stage_idx: int):
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
    batch_date = datetime(2022, 9, 1, 0)
    dataset = AuroraInitBatchesDataset(consts.BATCH_INITS_PATH, start=batch_date, end=batch_date)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    batch = next(iter(dataloader)).to(device)

    # Hook all unique (component, stage, block) blocks at once
    unique_blocks = list({(p[1], p[2], p[3]) for p in PLOTS})
    captured: dict[tuple, torch.Tensor] = {}
    handles = []

    def make_hook(key):
        def hook_fn(_m, _i, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[key] = x.detach()
        return hook_fn

    for component, stage, block in unique_blocks:
        if component == "decoder":
            tgt = model.backbone.decoder_layers[stage].blocks[block]
        else:
            tgt = model.backbone.encoder_layers[stage].blocks[block]
        handles.append(tgt.register_forward_hook(make_hook((component, stage, block))))

    logger.info(f"Running forward pass for {len(unique_blocks)} unique blocks...")
    with torch.no_grad():
        model(batch)

    for h in handles:
        h.remove()

    # === Pass 1: compute per-token variance-contribution maps for all 9 plots ===
    # contrib_per_token[t] = h[t, k]² / ||x[t] - b_dec||²
    # (W_dec rows are unit-normalized, so this equals the fraction of token t's
    # input variance reconstructable from feature k alone.)
    plot_data = []
    logger.info("Computing per-token variance-contribution maps...")
    for plot_num, component, stage, block, dim, feat_idx in PLOTS:
        sae_path = SAE_DIR / f"sae_{component}_s{stage}_b{block}.pt"
        sae = AuroraSAE(d_model=dim, expansion_factor=EXPANSION)
        sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
        sae = sae.to(device).eval()

        x = captured[(component, stage, block)]
        flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            _, h = sae(flat)
            x_centered = flat - sae.b_dec
            token_var = (x_centered ** 2).sum(dim=-1)  # (tokens,)
            contrib_per_token = (h[:, feat_idx] ** 2) / (token_var + 1e-12)  # (tokens,)

        target_decoder = (component == "decoder")
        C, H, W = get_spatial_resolution(model, batch, target_decoder, stage)
        # Average contribution over latent levels for spatial visualization
        spatial_contrib = contrib_per_token.reshape(C, H, W).mean(dim=0).cpu().numpy()

        plot_data.append({
            "plot_num": plot_num,
            "component": component,
            "stage": stage,
            "block": block,
            "feat_idx": feat_idx,
            "spatial": spatial_contrib,
            "peak": float(np.nanmax(spatial_contrib)),
        })
        logger.info(
            f"  Plot {plot_num} ({component} s{stage} b{block} feat #{feat_idx}): "
            f"peak contrib = {plot_data[-1]['peak']:.4f}"
        )

        del sae
        torch.cuda.empty_cache()

    # === Per-plot colorbars (each plot has its own [0, peak] scale) ===
    SUB_DIR = OUT_DIR / "per_plot_colorbar"
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Save plot_data (list of dicts with spatial maps + metadata) so plots
    # can be regenerated without rerunning the model on GPU.
    plot_data_path = OUT_DIR / "plot_data.pickle"
    with open(plot_data_path, "wb") as f:
        pickle.dump(plot_data, f)
    logger.info(f"Saved plot_data ({len(plot_data)} features) → {plot_data_path}")

    def render_panel(ax, d, cbar_target_ax, show_cbar_label: bool = True):
        """Render one panel into `ax` and a colorbar into `cbar_target_ax`."""
        component = d["component"]
        stage = d["stage"]
        block = d["block"]
        feat_idx = d["feat_idx"]
        peak = d["peak"]

        comp_label = component.capitalize()
        title = (
            f"{comp_label} stage {stage}, "
            f"block {block}, "
            f"feature #{feat_idx}"
        )

        ticks = nice_ticks_4(peak)
        vmax = float(ticks[-1]) if peak > 0 else 1.0

        rolled = np.roll(d["spatial"], d["spatial"].shape[1] // 2, axis=1)
        im = ax.imshow(
            rolled, extent=[-180, 180, -90, 90], origin="upper",
            transform=ccrs.PlateCarree(), cmap="inferno",
            vmin=0, vmax=vmax, interpolation="nearest",
        )
        ax.coastlines(color="white", linewidth=0.5)
        ax.set_global()
        ax.set_title(title, fontsize=22, fontweight="normal", pad=8)

        cbar = ax.figure.colorbar(im, cax=cbar_target_ax, orientation="horizontal",
                                  ticks=list(ticks))
        if show_cbar_label:
            cbar.set_label("Variance contribution per token", fontsize=20, labelpad=8)
        cbar.ax.tick_params(labelsize=20)

    # === Composite with per-panel colorbars (rows × GRID_NCOLS) ===
    # Each panel: 2-row inner gridspec (map on top, horizontal colorbar below).
    n_plots = len(PLOTS)
    n_rows = (n_plots + GRID_NCOLS - 1) // GRID_NCOLS
    # Sensible default sizes: 6"/col wide, 4"/row tall (room for horizontal cbar).
    fig = plt.figure(figsize=(6 * GRID_NCOLS + 1, 4 * n_rows + 1))
    outer_gs = fig.add_gridspec(
        n_rows, GRID_NCOLS,
        wspace=0.18, hspace=0.30,
        left=0.03, right=0.97, top=0.96, bottom=0.04,
    )

    for d in plot_data:
        plot_num = d["plot_num"]
        row = (plot_num - 1) // GRID_NCOLS
        col = (plot_num - 1) % GRID_NCOLS
        inner_gs = outer_gs[row, col].subgridspec(
            2, 1, height_ratios=[1, 0.05], hspace=0.30,
        )
        ax = fig.add_subplot(inner_gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
        cbar_ax = fig.add_subplot(inner_gs[1, 0])
        # Only the bottom row shows the colorbar axis label.
        render_panel(ax, d, cbar_ax, show_cbar_label=(row == n_rows - 1))

    composite_pdf = SUB_DIR / "thesis_3x3_grid.pdf"
    composite_png = composite_pdf.with_suffix(".png")
    plt.savefig(composite_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(composite_png, dpi=300, bbox_inches="tight", facecolor="white")
    # Also save to OUT_DIR (top-level) for convenient direct access.
    out_pdf = OUT_DIR / "thesis_3x3_grid.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"\nComposite saved to {composite_pdf} and {out_pdf}")

    # === Individual plot{1-9} figures ===
    for d in plot_data:
        plot_num = d["plot_num"]
        fig = plt.figure(figsize=(13, 6))
        gs_single = fig.add_gridspec(
            1, 2, width_ratios=[1, 0.035], wspace=0.04,
            left=0.04, right=0.93, top=0.90, bottom=0.05,
        )
        ax = fig.add_subplot(gs_single[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
        cbar_ax = fig.add_subplot(gs_single[0, 1])
        render_panel(ax, d, cbar_ax)

        component = d["component"]
        stage = d["stage"]
        block = d["block"]
        feat_idx = d["feat_idx"]
        save_pdf = SUB_DIR / f"plot{plot_num}_{component}_s{stage}_b{block}_feat{feat_idx}.pdf"
        plt.savefig(save_pdf, dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(save_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved {save_pdf.name}")

    logger.info(f"\nAll outputs in {SUB_DIR}")


if __name__ == "__main__":
    main()
