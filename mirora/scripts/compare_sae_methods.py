"""
Comparison figures: L1 SAE vs BatchTopK SAE.

Figures:
  1. FVE per backbone stage (grouped bars)
  2. L0 sparsity per backbone stage (grouped bars)
  3. Training loss curves (one representative block per stage)
  4. Dead feature fraction over BatchTopK training
  5. Top feature spatial maps side-by-side (needs the GPU)

Requires trained L1 and BatchTopK SAEs and the FVE CSVs from
`compute_fve.py --method l1` and `compute_fve.py --method topk`. Training-curve
panels are left empty for SAEs without a training log.
"""

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from mirora.config import DECODER_CONFIG, DEFAULT_EXPANSION_FACTOR, ENCODER_CONFIG
from mirora.data.loading import load_batch
from mirora.inference.inference import load_aurora
from mirora.interp import (BatchTopKSAE, capture_blocks, find_topk_checkpoint, get_spatial_resolution,
                           load_sae, sae_path)
from mirora.utils import setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style

import consts

logger = logging.getLogger(__name__)

FIGURES_DIR = consts.FIGURES_PATH / "sae_comparison"
SAE_TOPK_DIR = consts.MODEL_PATH / "sae_topk"

# (component, stage, n_blocks, dim) per backbone stage, in pipeline order
STAGE_INFO = [("encoder", s, n, d) for s, (n, d) in enumerate(ENCODER_CONFIG)] + \
             [("decoder", s, n, d) for s, (n, d) in enumerate(DECODER_CONFIG)]
STAGE_LABELS = [f"{c[:3].title()} S{s}\n(dim={d})" for c, s, _, d in STAGE_INFO]
# One representative (middle) block per stage
REPRESENTATIVE_BLOCKS = [(c, s, n // 2) for c, s, n, _ in STAGE_INFO]

L1_COLOR, TOPK_COLOR = "#4878CF", "#D65F5F"


def load_fve_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def aggregate_by_stage(rows: list[dict], metric: str) -> tuple[list[float], list[float]]:
    """Mean and std of `metric` across blocks within each stage, in STAGE_INFO order (0 if missing)."""
    accum = defaultdict(list)
    for r in rows:
        accum[(r["component"], int(r["stage"]))].append(float(r.get(metric, 0)))
    means = [np.mean(accum[(c, s)]) if accum[(c, s)] else 0 for c, s, _, _ in STAGE_INFO]
    stds = [np.std(accum[(c, s)]) if accum[(c, s)] else 0 for c, s, _, _ in STAGE_INFO]
    return means, stds


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {name}")


def fig_stage_bars(metric, scale, ylabel, title, name, ylim=None):
    """Grouped bar chart of a per-stage FVE-CSV metric, L1 vs BatchTopK."""
    logger.info(f"{name}: {metric} by stage")
    l1_rows = load_fve_csv(consts.FIGURES_PATH / "fve_l1_results.csv")
    topk_rows = load_fve_csv(consts.FIGURES_PATH / "fve_topk_results.csv")
    if not l1_rows and not topk_rows:
        logger.warning("No FVE CSVs found. Run compute_fve.py --method l1 / --method topk first.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(STAGE_LABELS))
    width = 0.35
    for rows, offset, label, color in [(l1_rows, -width / 2, "L1 SAE", L1_COLOR),
                                       (topk_rows, width / 2, "BatchTopK SAE", TOPK_COLOR)]:
        if rows:
            means, stds = aggregate_by_stage(rows, metric)
            ax.bar(x + offset, np.multiply(means, scale), width, yerr=np.multiply(stds, scale), label=label,
                   color=color, edgecolor="white", linewidth=0.5, capsize=3)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, name)


def load_training_log(component, stage, block, method):
    """Per-batch training log written next to the SAE checkpoint, or None."""
    if method == "topk":
        found = find_topk_checkpoint(SAE_TOPK_DIR, component, stage, block)
        if found is None:
            return None
        ckpt = found[0]
    else:
        ckpt = sae_path(consts.SAE_DIR, component, stage, block)
    path = ckpt.with_name(f"training_log_{ckpt.stem}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig3_training_curves():
    """Reconstruction-loss curves from the training logs: one panel per backbone stage."""
    logger.info("Figure 3: Training loss curves")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    fig.suptitle("Training Loss Curves: L1 vs BatchTopK SAE", fontsize=14, fontweight="bold")

    for ax, key, label in zip(axes.flatten(), REPRESENTATIVE_BLOCKS, STAGE_LABELS):
        for method, color, name in [("l1", L1_COLOR, "L1"), ("topk", TOPK_COLOR, "BatchTopK")]:
            log = load_training_log(*key, method)
            if log is not None:
                ax.plot([e["recon_loss"] for e in log], color=color, alpha=0.7, linewidth=0.8, label=name)

        ax.set_title(label.replace("\n", " "), fontsize=10)
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("Reconstruction MSE", fontsize=9)
        if ax.get_lines():
            ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    save(fig, "fig3_training_curves")


def fig4_dead_features():
    """Dead feature fraction over training from the BatchTopK logs."""
    logger.info("Figure 4: Dead feature fraction")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    fig.suptitle("Dead Feature Fraction During Training (BatchTopK SAE)", fontsize=14, fontweight="bold")

    for ax, key, label in zip(axes.flatten(), REPRESENTATIVE_BLOCKS, STAGE_LABELS):
        log = load_training_log(*key, "topk")
        if log is not None:
            dead_fracs = [e["dead_frac"] for e in log]
            ax.plot(dead_fracs, color=TOPK_COLOR, linewidth=0.8)
            ax.fill_between(range(len(dead_fracs)), dead_fracs, alpha=0.15, color=TOPK_COLOR)

        ax.set_title(label.replace("\n", " "), fontsize=10)
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("Dead feature fraction", fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    save(fig, "fig4_dead_features")


def fig5_feature_maps():
    """Side-by-side top feature spatial maps (encoder S1 B5): L1 vs BatchTopK."""
    logger.info("Figure 5: Top feature spatial maps (requires GPU)")
    component, stage_idx, block_idx, dim = "encoder", 1, 5, 1024
    top_k = 4

    found = find_topk_checkpoint(SAE_TOPK_DIR, component, stage_idx, block_idx)
    if found is None:
        logger.warning("No BatchTopK SAE found for feature map comparison.")
        return
    topk_path, k = found

    model, device = load_aurora(consts.CHECKPOINT_PATH)
    batch = load_batch(consts.BATCH_INITS_PATH, datetime(2022, 11, 1, 0)).to(device)
    key = (component, stage_idx, block_idx)
    with capture_blocks(model, [key]) as captured, torch.no_grad():
        model(batch)
    x = captured[key]
    x = x.reshape(-1, x.shape[-1])
    C, H, W = get_spatial_resolution(model, batch, component, stage_idx)

    l1_sae = load_sae(consts.SAE_DIR, component, stage_idx, block_idx, dim, device)
    topk_sae = BatchTopKSAE(d_model=dim, expansion_factor=DEFAULT_EXPANSION_FACTOR, k=k)
    topk_sae.load_state_dict(torch.load(topk_path, map_location=device, weights_only=True))
    topk_sae = topk_sae.to(device).eval()

    maps = []
    with torch.no_grad():
        for h in (l1_sae(x)[1], topk_sae(x, training=False)[1]):
            spatial = h.reshape(C, H, W, -1).mean(dim=0).cpu().numpy()
            maps.append((spatial, torch.topk(h.mean(dim=0).cpu(), top_k).indices.tolist()))

    fig, axes = plt.subplots(
        top_k, 2, figsize=(12, 3 * top_k),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)},
    )
    fig.suptitle(
        f"Top {top_k} Features — {component} stage {stage_idx} block {block_idx} (dim={dim})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    axes[0, 0].set_title("L1 SAE", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].set_title("BatchTopK SAE", fontsize=13, fontweight="bold", pad=8)

    for row in range(top_k):
        for col, (spatial, top_idx) in enumerate(maps):
            ax = axes[row, col]
            im = draw_global_field(ax, spatial[:, :, top_idx[row]], "inferno", coast_color="white", coast_width=0.5)
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=15)
            cbar.ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"Feature #{top_idx[row]}", fontsize=10, labelpad=10)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.02, wspace=0.15, hspace=0.12)
    save(fig, "fig5_feature_maps")


def main():
    setup_logging()
    setup_matplotlib_style(11, axes_linewidth=0.6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_stage_bars("fve", 100, "Fraction of Variance Explained (%)",
                   "Reconstruction Quality: L1 vs BatchTopK SAE", "fig1_fve_comparison", ylim=(60, 102))
    fig_stage_bars("l0", 1, "Mean L0 (active features per token)",
                   "Sparsity: L1 vs BatchTopK SAE", "fig2_l0_comparison")
    fig3_training_curves()
    fig4_dead_features()
    fig5_feature_maps()

    logger.info(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
