"""
Per-token variance-contribution maps of chosen L1 SAE features on the running
example (Sep 1 2022 00Z), for the paper's SAE feature-grid figure.

Usage:
    python plot_sae_feature_grid.py                       # 2x2 case-study grid
    python plot_sae_feature_grid.py --top10               # top-10 features, 5x2
    python plot_sae_feature_grid.py --feature COMPONENT STAGE BLOCK FEAT [PLOT_NUM]

Grid modes save the composite, one single-feature figure per panel (in
`per_plot_colorbar/`), and `plot_data.pickle` (list of dicts) so the maps can be
re-plotted on CPU without rerunning the model. `--feature` saves one
single-feature figure plus a pickle of its map.
"""

import math
import pickle
import sys
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from mirora.config import DECODER_CONFIG, ENCODER_CONFIG
from mirora.data.loading import load_batch
from mirora.inference.inference import load_aurora
from mirora.interp import capture_blocks, get_spatial_resolution, load_sae
from mirora.utils import setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style

import consts

logger = logging.getLogger(__name__)

# Plot specs: (plot_num, component, stage, block, feature_idx), all 0-indexed.
# plot_num is reading order (1=top-left, ..., row-major over GRID_NCOLS columns).
GRID_NCOLS = 2
PLOTS = [
    (1, "encoder", 1, 0, 6873),    # transport
    (2, "encoder", 0, 3, 1145),    # activates everywhere
    (3, "decoder", 2, 4, 1421),    # day/night + transport
    (4, "decoder", 0, 6, 14566),   # latitude stripe
]

# --top10: the 1 Sep 2022 top-10 features by max variance contribution
# (find_top_activating_features.py), in rank order, 5 x 2.
TOP10_PLOTS = [
    (1, "encoder", 0, 5, 3813),
    (2, "encoder", 0, 4, 3711),
    (3, "encoder", 0, 5, 972),
    (4, "decoder", 1, 9, 2712),
    (5, "encoder", 1, 5, 2866),
    (6, "encoder", 0, 5, 2870),
    (7, "decoder", 0, 3, 12363),
    (8, "encoder", 2, 2, 10282),
    (9, "encoder", 2, 2, 4710),
    (10, "decoder", 0, 0, 5449),
]


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


def compute_plot_data(plots) -> list[dict]:
    """Per-token variance contribution h[t, k]² / ||x[t] - b_dec||² of each plotted
    feature, averaged over latent levels."""
    model, device = load_aurora(consts.CHECKPOINT_PATH)
    batch = load_batch(consts.BATCH_INITS_PATH, consts.MAIN_INIT_TIME).to(device)

    keys = list({(c, s, b) for _, c, s, b, _ in plots})
    logger.info(f"Running forward pass for {len(keys)} unique blocks...")
    with torch.no_grad(), capture_blocks(model, keys) as acts:
        model(batch)

    plot_data = []
    for plot_num, component, stage, block, feat_idx in plots:
        dim = (ENCODER_CONFIG if component == "encoder" else DECODER_CONFIG)[stage][1]
        sae = load_sae(consts.SAE_DIR, component, stage, block, dim, device)
        x = acts[(component, stage, block)]
        flat = x.reshape(-1, x.shape[-1])
        # Only feature k's contribution is needed; var_contribution would
        # materialise h**2 for every feature (tens of GB at decoder S0).
        with torch.no_grad():
            h = sae.encode(flat)
            token_var = ((flat - sae.b_dec) ** 2).sum(dim=-1)
            contrib_per_token = (h[:, feat_idx] ** 2) / (token_var + 1e-12)

        C, H, W = get_spatial_resolution(model, batch, component, stage)
        spatial = contrib_per_token.reshape(C, H, W).mean(dim=0).cpu().numpy()
        plot_data.append({
            "plot_num": plot_num,
            "component": component,
            "stage": stage,
            "block": block,
            "feat_idx": feat_idx,
            "spatial": spatial,
            "peak": float(np.nanmax(spatial)),
        })
        logger.info(f"  Plot {plot_num} ({component} s{stage} b{block} feat #{feat_idx}): "
                    f"peak contrib = {plot_data[-1]['peak']:.4f}")
        del sae, h
        torch.cuda.empty_cache()
    return plot_data


def render_panel(ax, cbar_ax, d, orientation: str, cbar_label: str | None):
    ticks = nice_ticks_4(d["peak"])
    vmax = float(ticks[-1]) if d["peak"] > 0 else 1.0
    im = draw_global_field(ax, d["spatial"], "inferno", 0, vmax, coast_color="white", coast_width=0.5)
    ax.set_title(f"{d['component'].capitalize()} stage {d['stage']}, block {d['block']}, "
                 f"feature #{d['feat_idx']}", fontsize=22, fontweight="normal", pad=8)
    cbar = ax.figure.colorbar(im, cax=cbar_ax, orientation=orientation, ticks=list(ticks))
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=20, labelpad=8)
    cbar.ax.tick_params(labelsize=20)


def save_fig(fig, save_pdf):
    fig.savefig(save_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {save_pdf}")


def plot_single(d, out_dir):
    """One feature map with a vertical colorbar; returns the output path stem."""
    fig = plt.figure(figsize=(8, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.035], wspace=0.04,
                          left=0.04, right=0.93, top=0.90, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    render_panel(ax, fig.add_subplot(gs[0, 1]), d, "vertical", "Variance contribution\nper token")
    stem = out_dir / f"plot{d['plot_num']}_{d['component']}_s{d['stage']}_b{d['block']}_feat{d['feat_idx']}"
    save_fig(fig, stem.with_suffix(".pdf"))
    return stem


def plot_grid(plot_data, save_pdf):
    """Rows × GRID_NCOLS panels, each with its own horizontal colorbar below."""
    n_rows = (len(plot_data) + GRID_NCOLS - 1) // GRID_NCOLS
    fig = plt.figure(figsize=(6 * GRID_NCOLS + 1, 4 * n_rows + 1))
    outer_gs = fig.add_gridspec(n_rows, GRID_NCOLS, wspace=0.18, hspace=0.30,
                                left=0.03, right=0.97, top=0.96, bottom=0.04)
    for d in plot_data:
        row, col = divmod(d["plot_num"] - 1, GRID_NCOLS)
        inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[1, 0.05], hspace=0.30)
        ax = fig.add_subplot(inner_gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
        # Only the bottom row shows the colorbar label.
        render_panel(ax, fig.add_subplot(inner_gs[1, 0]), d, "horizontal",
                     "Variance contribution per token" if row == n_rows - 1 else None)
    save_fig(fig, save_pdf)


def main():
    setup_logging()
    setup_matplotlib_style(22, family="sans-serif")
    args = sys.argv[1:]

    if args[:1] == ["--feature"]:
        if len(args) not in (5, 6):
            sys.exit(__doc__)
        component, stage, block, feat_idx = args[1], int(args[2]), int(args[3]), int(args[4])
        assert component in ("encoder", "decoder")
        plot_num = int(args[5]) if len(args) > 5 else 1
        out_dir = consts.FIGURES_PATH / "sae_feature_grid" / "per_plot_colorbar"
        out_dir.mkdir(parents=True, exist_ok=True)
        [d] = compute_plot_data([(plot_num, component, stage, block, feat_idx)])
        stem = plot_single(d, out_dir)
        with open(stem.with_suffix(".pickle"), "wb") as f:
            pickle.dump(d, f)
        logger.info(f"Saved {stem.with_suffix('.pickle')}")
        return

    if args == ["--top10"]:
        plots, out_dir, composite_name = TOP10_PLOTS, consts.FIGURES_PATH / "sae_top10_sep01", "top10_grid"
    elif not args:
        plots, out_dir, composite_name = PLOTS, consts.FIGURES_PATH / "sae_feature_grid", "feature_grid"
    else:
        sys.exit(__doc__)
    sub_dir = out_dir / "per_plot_colorbar"
    sub_dir.mkdir(parents=True, exist_ok=True)

    plot_data = compute_plot_data(plots)
    with open(out_dir / "plot_data.pickle", "wb") as f:
        pickle.dump(plot_data, f)
    logger.info(f"Saved plot_data ({len(plot_data)} features) → {out_dir / 'plot_data.pickle'}")

    plot_grid(plot_data, out_dir / f"{composite_name}.pdf")
    for d in plot_data:
        plot_single(d, sub_dir)


if __name__ == "__main__":
    main()
