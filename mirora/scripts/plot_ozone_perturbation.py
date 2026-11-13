"""
Panels showing how NOx perturbations affect ozone, at +12, +72 and +120 h:
  per lead: baseline absolute O3, % change for +0.5/+5/+20 ppb, and the O3 KDE
  composites (one column per lead): o3_stacked_fig3 (+0.5/+5 ppb, Figure 3) and
  o3_stacked_si (+0.5/+5/+20 ppb, SI)
--negative-map: SI map of cells with negative O3 under +5 ppb at +12 h only
--negative-species: one figure per species, negative (red) vs nonnegative (white) cells, +5 ppb, +12 h

All at 1000 hPa, initialized Sep 1 2022 00:00Z. No titles on map plots.
"""

import functools
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from scipy.stats import gaussian_kde

from mirora.config import CHEM_ATMOS_VARS, MMR_TO_PPB
from mirora.data.loading import fetch_rollout, surface
from mirora.stats import lat_weights, wmean
from mirora.utils import setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style

import consts

logger = logging.getLogger(__name__)

setup_matplotlib_style(11, family="sans-serif", axes_linewidth=0.6)

FIGURES_DIR = consts.NOX_FIGURES_PATH / "ozone_perturbation"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INIT_STR = consts.MAIN_INIT_STR

LABELS = {
    "base": "Baseline",
    "NOx_0.5ppb_increase": "+0.5 ppb",
    "NOx_5ppb_increase": "+5 ppb",
    "NOx_20ppb_increase": "+20 ppb",
}
PERTURBED = [e for e in LABELS if e != "base"]

COLORS = {
    "base": "#4878CF",
    "NOx_0.5ppb_increase": "#6AB572",
    "NOx_5ppb_increase": "#E5AE38",
    "NOx_20ppb_increase": "#C44E52",
}

KG_KG_TO_PPB = MMR_TO_PPB["go3"]
LEADS = {12: 0, 72: 5, 120: 9}  # lead hours -> rollout step


@functools.lru_cache(maxsize=None)
def load_o3_steps(exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(O3 at 1000 hPa for every rollout step [steps, H, W], cos-lat row weights [H]); each pickle read once."""
    exp_dir = consts.EXPERIMENTS_PATH / "base" if exp_name == "base" else consts.NOX_EXPERIMENTS_PATH / exp_name
    preds = fetch_rollout(consts.rollout_path(exp_dir, INIT_STR))
    return np.stack([surface(p, "go3") for p in preds]), lat_weights(preds)


def load_o3(exp_name: str, step: int) -> np.ndarray:
    """Load O3 at 1000 hPa for a given experiment and step."""
    return load_o3_steps(exp_name)[0][step]


def draw_o3_map(ax, data: np.ndarray, cmap, vmin: float, vmax: float):
    """O3 map style shared by every Fig 3 panel."""
    return draw_global_field(ax, data, cmap, vmin, vmax, borders=True, border_width=0.6, border_color="#888888")


def plot_single_map(data: np.ndarray, save_path: Path, cmap, vmin: float, vmax: float, cbar_label: str):
    """One O3 panel with its own vertical colorbar, no title."""
    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.03], wspace=0.02,
                          left=0.02, right=0.88, top=0.97, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    cbar_ax = fig.add_subplot(gs[0, 1])

    im = draw_o3_map(ax, data, cmap, vmin, vmax)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def plot_absolute_map(data: np.ndarray, save_path: Path, vmin: float, vmax: float):
    """Absolute O3 concentration (input in kg/kg) in ppb."""
    plot_single_map(data * KG_KG_TO_PPB, save_path, "viridis", vmin, vmax, r"O$_3$ (ppb)")


def plot_pct_change_map(data: np.ndarray, save_path: Path, vmin: float, vmax: float):
    """Percent change in O3; masked (NaN) cells are drawn white."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    plot_single_map(data, save_path, cmap, vmin, vmax, r"$\Delta$O$_3$ (% from baseline)")


def draw_kde_panel(ax, o3_fields: dict[str, np.ndarray], label_size: int, ylabel: bool = True) -> float:
    """Overlaid O3 KDEs (one per experiment, fixed -40..60 ppb x-range); returns the peak density."""
    x_range = np.linspace(-40, 60, 800)
    max_density = 0.0
    for exp_name, field in o3_fields.items():
        density = kde_curve(field, x_range)
        max_density = max(max_density, float(density.max()))
        ax.fill_between(x_range, density, alpha=0.2, color=COLORS[exp_name], edgecolor="none")
        ax.plot(x_range, density, color=COLORS[exp_name], linewidth=2.5, label=LABELS[exp_name])
    ax.set_xlim(-40, 60)
    ax.set_xlabel(r"O$_3$ concentration (ppb)", fontsize=label_size, labelpad=8)
    if ylabel:
        ax.set_ylabel("Probability density", fontsize=label_size, labelpad=8)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)  # y-axis is shared
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=20, width=0.8, length=5)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
    return max_density


def plot_kde_distribution(o3_fields: dict[str, np.ndarray], save_path: Path):
    """Plot KDE distributions of O3 for all 4 scenarios."""
    fig, ax = plt.subplots(figsize=(16, 10))
    draw_kde_panel(ax, o3_fields, label_size=20)
    ax.set_ylim(bottom=0)
    legend = ax.legend(
        fontsize=20, framealpha=0.95, edgecolor="#dddddd", fancybox=True,
        loc="upper right", borderpad=1.0, handlelength=2.5,
    )
    legend.get_frame().set_linewidth(0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def kde_curve(field: np.ndarray, x_range: np.ndarray) -> np.ndarray:
    """Gaussian KDE (in ppb) of a 50k-cell random subsample of an O3 field (kg/kg)."""
    flat = field.ravel() * KG_KG_TO_PPB
    subsample = np.random.default_rng(42).choice(flat, size=min(50000, len(flat)), replace=False)
    return gaussian_kde(subsample, bw_method=0.15)(x_range)


def plot_stacked_composite(base, pct, perturbed, pct_keys, abs_vmin, abs_vmax, pct_limit, save_path):
    """Composite with one column per lead time and shared colorbars.

    base, perturbed[exp], pct[exp]: dicts {lead_hours: field}. Rows:
      0: baseline absolute O3 (ppb)         -> absolute colorbar
      1..n: % change for each of pct_keys   -> shared % colorbar
      last: O3 distributions (KDE) at each lead
    """
    leads = list(base)
    n_lead, n_pct = len(leads), len(pct_keys)
    kde_row = 1 + n_pct
    proj = ccrs.PlateCarree(central_longitude=0)
    pct_cmap = plt.get_cmap("RdBu_r").copy()
    pct_cmap.set_bad("white")

    fig = plt.figure(figsize=(10 * n_lead, 5 * (1 + n_pct) + 7))
    gs = fig.add_gridspec(
        kde_row + 1, n_lead + 1,
        width_ratios=[1] * n_lead + [0.04],
        height_ratios=[1] * kde_row + [1.4],
        wspace=0.06, hspace=0.10,
        left=0.04, right=0.90, top=0.97, bottom=0.05,
    )

    for col, lead in enumerate(leads):
        ax = fig.add_subplot(gs[0, col], projection=proj)
        im_abs = draw_o3_map(ax, base[lead] * KG_KG_TO_PPB, "viridis", abs_vmin, abs_vmax)
        ax.set_title(f"+{lead} h", fontsize=24, pad=10)
        for row, key in enumerate(pct_keys, start=1):
            ax = fig.add_subplot(gs[row, col], projection=proj)
            im_pct = draw_o3_map(ax, pct[key][lead], pct_cmap, -pct_limit, pct_limit)

    cbar_abs = fig.colorbar(im_abs, cax=fig.add_subplot(gs[0, n_lead]), orientation="vertical", extend="both")
    cbar_abs.set_label(r"O$_3$ (ppb)", fontsize=22, labelpad=10)
    cbar_abs.ax.tick_params(labelsize=20)
    cbar_pct = fig.colorbar(im_pct, cax=fig.add_subplot(gs[1:kde_row, n_lead]), orientation="vertical", extend="both")
    cbar_pct.set_label(r"$\Delta$O$_3$ (% deviation from baseline)", fontsize=22, labelpad=10)
    cbar_pct.ax.tick_params(labelsize=20)

    # Final row: KDE distributions, extra wspace so x-tick labels don't overlap
    sub_gs = gs[kde_row, :n_lead].subgridspec(1, n_lead, wspace=0.10)
    dist_axes = []
    max_density = 0.0
    for col, lead in enumerate(leads):
        ax = fig.add_subplot(sub_gs[0, col], sharey=dist_axes[0] if dist_axes else None)
        dist_axes.append(ax)
        fields = {"base": base[lead], **{k: perturbed[k][lead] for k in pct_keys}}
        max_density = max(max_density, draw_kde_panel(ax, fields, label_size=22, ylabel=col == 0))
    dist_axes[0].set_ylim(0, max_density * 1.10)
    legend = dist_axes[0].legend(fontsize=20, framealpha=0.95, edgecolor="#dddddd", fancybox=True,
                                 loc="upper left", borderpad=0.8, handlelength=2.0)
    legend.get_frame().set_linewidth(0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def pct_change(perturbed, baseline):
    threshold = np.nanmean(np.abs(baseline)) * 0.001
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(baseline) > threshold, 100 * (perturbed - baseline) / baseline, np.nan)


def plot_negative_o3_map(exp_name: str = "NOx_5ppb_increase", lead: int = 12):
    """Single-day SI map of where a NOx perturbation drives surface O3 negative.

    Cells with O3 >= 0 are light grey; negative cells are coloured by concentration.
    """
    o3 = load_o3(exp_name, LEADS[lead]) * KG_KG_TO_PPB
    neg = np.where(o3 < 0, o3, np.nan)
    w_row = load_o3_steps(exp_name)[1]
    logger.info(f"{exp_name} +{lead}h: {100 * np.mean(o3 < 0):.2f}% of cells negative "
                f"({100 * wmean(o3 < 0, w_row):.2f}% area-weighted), min {np.nanmin(o3):.1f} ppb")

    cmap = plt.get_cmap("Reds_r").copy()
    cmap.set_bad("#e6e6e6")
    vmin = float(np.floor(np.nanmin(neg)))
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    im = draw_o3_map(ax, neg, cmap, vmin, 0)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", aspect=40, fraction=0.05, pad=0.04)
    cbar.set_label(r"Negative surface O$_3$ (ppb)", fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    save_path = FIGURES_DIR / f"o3_negative_{exp_name}_{lead}h.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def plot_negative_species_maps(exp_name: str = "NOx_5ppb_increase", lead: int = 12):
    """SI figures, one per chemistry species: cells <= 0 in red, cells > 0 in white."""
    preds = fetch_rollout(consts.rollout_path(consts.NOX_EXPERIMENTS_PATH / exp_name, INIT_STR))
    w_row = lat_weights(preds)
    cmap = ListedColormap(["#d62728", "white"])  # 0 = negative, 1 = nonnegative
    for sp, label in CHEM_ATMOS_VARS.items():
        handles = [Patch(facecolor="#d62728", label=rf"Surface [{label}] $\leq$ 0"),
                   Patch(facecolor="white", edgecolor="black", label=f"Surface [{label}] > 0")]
        neg = surface(preds[LEADS[lead]], sp) <= 0
        logger.info(f"{exp_name} +{lead}h {sp}: {100 * neg.mean():.2f}% of cells nonpositive "
                    f"({100 * wmean(neg, w_row):.2f}% area-weighted)")
        fig, ax = plt.subplots(figsize=(14, 7.5), subplot_kw={"projection": ccrs.PlateCarree()})
        # interpolation="none": the PDF embeds the 0.4 deg grid unresampled, so every cell has the same size
        draw_global_field(ax, (~neg).astype(float), cmap, 0, 1, borders=True, border_width=0.6,
                          border_color="#888888", interpolation="none")
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                  fontsize=22, frameon=False)
        save_path = FIGURES_DIR / f"negative_{sp}_{exp_name}_{lead}h.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(save_path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved {save_path.name}")


def main():
    setup_logging()
    if sys.argv[1:] == ["--negative-map"]:
        return plot_negative_o3_map()
    if sys.argv[1:] == ["--negative-species"]:
        return plot_negative_species_maps()
    logger.info("Loading O3 fields...")
    base = {lead: load_o3("base", step) for lead, step in LEADS.items()}
    perturbed = {exp: {lead: load_o3(exp, step) for lead, step in LEADS.items()} for exp in PERTURBED}
    pct = {exp: {lead: pct_change(f, base[lead]) for lead, f in fields.items()} for exp, fields in perturbed.items()}

    # Shared symmetric % colorbar (95th pct of |change| over every experiment and lead)
    pct_limit = np.nanpercentile(np.abs(np.concatenate(
        [p[np.isfinite(p)] for fields in pct.values() for p in fields.values()])), 95)
    logger.info(f"Shared pct colorbar: ±{pct_limit:.2f}%")
    abs_vmin = 0
    abs_vmax = max(np.nanpercentile(f * KG_KG_TO_PPB, 98) for f in base.values())

    for lead in LEADS:
        logger.info(f"Generating +{lead}h panels...")
        plot_absolute_map(base[lead], FIGURES_DIR / f"o3_baseline_{lead}h.pdf", vmin=abs_vmin, vmax=abs_vmax)
        for exp_name in PERTURBED:
            plot_pct_change_map(pct[exp_name][lead], FIGURES_DIR / f"o3_pct_{exp_name}_{lead}h.pdf",
                                vmin=-pct_limit, vmax=pct_limit)
        plot_kde_distribution({"base": base[lead], **{e: perturbed[e][lead] for e in perturbed}},
                              FIGURES_DIR / f"o3_distribution_{lead}h.pdf")

    logger.info("Generating stacked composites...")
    for keys, name in [(["NOx_0.5ppb_increase", "NOx_5ppb_increase"], "o3_stacked_fig3"),
                       (PERTURBED, "o3_stacked_si")]:
        plot_stacked_composite(base, pct, perturbed, keys, abs_vmin, abs_vmax, pct_limit, FIGURES_DIR / f"{name}.pdf")

    logger.info(f"\nAll panels saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
