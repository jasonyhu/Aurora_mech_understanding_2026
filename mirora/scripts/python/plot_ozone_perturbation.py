"""
Generate 9 panels showing how NOx perturbations affect ozone:
  Plots 1-4: +12h rollout (baseline absolute O3, then % change for +0.5/+5/+20 ppb)
  Plots 5-8: +72h rollout (same layout)
  Plot 9: KDE distribution of O3 concentrations for all 4 scenarios at +12h

All at 1000 hPa, initialized Sep 1 2022 00:00Z. No titles on map plots.
"""

import pickle
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

FIGURES_DIR = consts.FIGURES_PATH / "ozone_perturbation"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INIT_STR = "20220901_0000"
LEVEL = 0  # 1000 hPa

EXPERIMENTS = [
    ("base", "Baseline", None),
    ("NOx_0.5ppb_increase", "+0.5 ppb", 0.5),
    ("NOx_5ppb_increase", "+5 ppb", 5),
    ("NOx_20ppb_increase", "+20 ppb", 20),
]

COLORS = {
    "base": "#4878CF",
    "NOx_0.5ppb_increase": "#6AB572",
    "NOx_5ppb_increase": "#E5AE38",
    "NOx_20ppb_increase": "#C44E52",
}

# Conversion factor: kg/kg → ppb for O3
# ppb = mmr * (M_air / M_O3) * 1e9
M_AIR = 28.97  # g/mol
M_O3 = 48.0    # g/mol
KG_KG_TO_PPB = (M_AIR / M_O3) * 1e9


def load_o3(exp_name: str, step: int) -> np.ndarray:
    """Load O3 at 1000 hPa for a given experiment and step."""
    if exp_name == "base":
        path = consts.EXPERIMENTS_PATH / "base" / INIT_STR / "predictions.pickle"
    else:
        path = consts.EXPERIMENTS_PATH / exp_name / INIT_STR / "predictions.pickle"
    with open(path, "rb") as f:
        preds = pickle.load(f)
    return preds[step].atmos_vars["go3"][0, 0, LEVEL].numpy()


def plot_absolute_map(data: np.ndarray, save_path: Path, vmin: float, vmax: float):
    """Plot absolute O3 concentration in ppb with its own colorbar."""
    data = data * KG_KG_TO_PPB
    data_rolled = np.roll(data, data.shape[1] // 2, axis=1)

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.03], wspace=0.02,
                          left=0.02, right=0.88, top=0.97, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    cbar_ax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(data_rolled, extent=[-180, 180, -90, 90], origin="upper",
                   transform=ccrs.PlateCarree(), cmap="viridis",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_global()
    ax.coastlines(linewidth=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#888888")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(r"O$_3$ (ppb)", fontsize=14)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def plot_pct_change_map(data: np.ndarray, save_path: Path, vmin: float, vmax: float):
    """Plot percent change in O3, no title, shared colorbar range."""
    data_rolled = np.roll(data, data.shape[1] // 2, axis=1)

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.03], wspace=0.02,
                          left=0.02, right=0.88, top=0.97, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    cbar_ax = fig.add_subplot(gs[0, 1])

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    im = ax.imshow(data_rolled, extent=[-180, 180, -90, 90], origin="upper",
                   transform=ccrs.PlateCarree(), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_global()
    ax.coastlines(linewidth=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#888888")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(r"$\Delta$O$_3$ (% from baseline)", fontsize=14)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def plot_kde_distribution(o3_fields: dict[str, np.ndarray], save_path: Path):
    """Plot KDE distributions of O3 for all 4 scenarios."""
    fig, ax = plt.subplots(figsize=(16, 10))

    labels = {
        "base": "Baseline",
        "NOx_0.5ppb_increase": r"+0.5 ppb NO$_x$",
        "NOx_5ppb_increase": r"+5 ppb NO$_x$",
        "NOx_20ppb_increase": r"+20 ppb NO$_x$",
    }

    # Fixed x-axis range for consistency across +12h and +72h
    x_range = np.linspace(-40, 60, 800)

    for exp_name, field in o3_fields.items():
        flat = field.ravel() * KG_KG_TO_PPB
        rng = np.random.default_rng(42)
        subsample = rng.choice(flat, size=min(50000, len(flat)), replace=False)

        kde = gaussian_kde(subsample, bw_method=0.15)
        density = kde(x_range)

        ax.fill_between(x_range, density, alpha=0.2, color=COLORS[exp_name],
                         edgecolor="none")
        ax.plot(x_range, density, color=COLORS[exp_name], linewidth=2.5,
                label=labels[exp_name])

    ax.set_xlim(-40, 60)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"O$_3$ concentration (ppb)", fontsize=20, labelpad=8)
    ax.set_ylabel("Probability density", fontsize=20, labelpad=8)

    legend = ax.legend(
        fontsize=20, framealpha=0.95, edgecolor="#dddddd", fancybox=True,
        loc="upper right", borderpad=1.0, handlelength=2.5,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=20, width=0.8, length=5)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    # Subtle vertical line at x=0 to mark zero
    ax.axvline(0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def plot_stacked_8panel(
    base_12h, base_72h, pct_12h, pct_72h,
    o3_fields_12h, o3_fields_72h,
    abs_vmin, abs_vmax, pct_limit, save_path,
):
    """Create a 4-row × 2-column composite with shared colorbars + distributions.

    Layout (rows × cols):
      Row 0: Baseline +12h  |  Baseline +72h     → shared colorbar (absolute, ppb)
      Row 1: +0.5ppb  +12h  |  +0.5ppb  +72h     → shared colorbar (% change)
      Row 2: +5ppb    +12h  |  +5ppb    +72h     →        "
      Row 3: KDE distribution +12h  |  KDE distribution +72h
    """
    proj = ccrs.PlateCarree(central_longitude=0)
    pct_cmap = plt.get_cmap("RdBu_r").copy()
    pct_cmap.set_bad("white")

    fig = plt.figure(figsize=(20, 25))

    # Main grid: 5 rows × 3 cols (left, right, colorbar):
    # row0 baseline abs; rows1-3 pct change (+0.5/+5/+20 ppb); row4 KDE.
    n_pct = 3
    kde_row = 1 + n_pct
    gs = fig.add_gridspec(
        1 + n_pct + 1, 3,
        width_ratios=[1, 1, 0.04],
        height_ratios=[1] * (1 + n_pct) + [1.4],
        wspace=0.06, hspace=0.10,
        left=0.04, right=0.90, top=0.99, bottom=0.05,
    )

    pct_keys = ["NOx_0.5ppb_increase", "NOx_5ppb_increase", "NOx_20ppb_increase"]

    panels = []
    # Row 0: baseline absolute (ppb)
    panels.append((base_12h * KG_KG_TO_PPB, "viridis", abs_vmin, abs_vmax))
    panels.append((base_72h * KG_KG_TO_PPB, "viridis", abs_vmin, abs_vmax))
    # Rows 1-2: pct change for +0.5 and +5 ppb
    for key in pct_keys:
        panels.append((pct_12h[key], pct_cmap, -pct_limit, pct_limit))
        panels.append((pct_72h[key], pct_cmap, -pct_limit, pct_limit))

    im_abs = None
    im_pct = None

    for i, (data, cmap, vmin, vmax) in enumerate(panels):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=proj)

        data_rolled = np.roll(data, data.shape[1] // 2, axis=1)
        im = ax.imshow(
            data_rolled, extent=(-180, 180, -90, 90), origin="upper",
            transform=ccrs.PlateCarree(), cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="#333333")
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#888888")
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            im_abs = im
        else:
            im_pct = im

    # Absolute colorbar — spans row 0 only
    assert im_abs is not None
    cbar_abs_ax = fig.add_subplot(gs[0, 2])
    cbar_abs = fig.colorbar(im_abs, cax=cbar_abs_ax, orientation="vertical", extend="both")
    cbar_abs.set_label(r"O$_3$ (ppb)", fontsize=22, labelpad=10)
    cbar_abs.ax.tick_params(labelsize=20)

    # Pct colorbar — spans the pct rows only
    assert im_pct is not None
    cbar_pct_ax = fig.add_subplot(gs[1:kde_row, 2])
    cbar_pct = fig.colorbar(im_pct, cax=cbar_pct_ax, orientation="vertical", extend="both")
    cbar_pct.set_label(r"$\Delta$O$_3$ (% deviation from baseline)", fontsize=22, labelpad=10)
    cbar_pct.ax.tick_params(labelsize=20)

    # Final row: KDE distributions
    labels = {
        "base": "Baseline",
        "NOx_0.5ppb_increase": r"+0.5 ppb NO$_x$",
        "NOx_5ppb_increase": r"+5 ppb NO$_x$",
        "NOx_20ppb_increase": r"+20 ppb NO$_x$",
    }
    x_range = np.linspace(-40, 60, 800)

    # Sub-gridspec for KDE row with extra wspace so x-tick labels don't overlap
    sub_gs = gs[kde_row, :2].subgridspec(1, 2, wspace=0.10)
    dist_axes = []
    max_density = 0.0
    for col, o3_fields in enumerate([o3_fields_12h, o3_fields_72h]):
        if col == 0:
            ax = fig.add_subplot(sub_gs[0, 0])
        else:
            ax = fig.add_subplot(sub_gs[0, 1], sharey=dist_axes[0])
        dist_axes.append(ax)

        for exp_name, field in o3_fields.items():
            flat = field.ravel() * KG_KG_TO_PPB
            rng = np.random.default_rng(42)
            subsample = rng.choice(flat, size=min(50000, len(flat)), replace=False)
            kde = gaussian_kde(subsample, bw_method=0.15)
            density = kde(x_range)
            max_density = max(max_density, float(density.max()))
            ax.fill_between(x_range, density, alpha=0.2,
                            color=COLORS[exp_name], edgecolor="none")
            ax.plot(x_range, density, color=COLORS[exp_name], linewidth=2.5,
                    label=labels[exp_name])

        ax.set_xlim(-40, 60)
        ax.set_xlabel(r"O$_3$ concentration (ppb)", fontsize=22, labelpad=8)
        if col == 0:
            ax.set_ylabel("Probability density", fontsize=22, labelpad=8)
        else:
            # Hide y-tick labels on right panel since y-axis is shared
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(labelsize=20, width=0.8, length=5)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.axvline(0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)

    # Set shared y-limit to fit the largest density across both panels
    dist_axes[0].set_ylim(0, max_density * 1.10)

    # Single shared legend in the top-left of the left panel
    handles, leg_labels = dist_axes[0].get_legend_handles_labels()
    legend = dist_axes[0].legend(
        handles, leg_labels,
        fontsize=20, framealpha=0.95, edgecolor="#dddddd", fancybox=True,
        loc="upper left", borderpad=0.8, handlelength=2.0,
    )
    legend.get_frame().set_linewidth(0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def main():
    # Load all O3 fields
    logger.info("Loading O3 fields...")
    step_12h = 0   # +12h
    step_72h = 5   # +72h

    base_12h = load_o3("base", step_12h)
    base_72h = load_o3("base", step_72h)

    perturbed_12h = {}
    perturbed_72h = {}
    for exp_name, label, ppb in EXPERIMENTS[1:]:
        perturbed_12h[exp_name] = load_o3(exp_name, step_12h)
        perturbed_72h[exp_name] = load_o3(exp_name, step_72h)

    # Compute percent changes
    def pct_change(perturbed, baseline):
        threshold = np.nanmean(np.abs(baseline)) * 0.001
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(np.abs(baseline) > threshold,
                           100 * (perturbed - baseline) / baseline, np.nan)
        return pct

    pct_12h = {k: pct_change(v, base_12h) for k, v in perturbed_12h.items()}
    pct_72h = {k: pct_change(v, base_72h) for k, v in perturbed_72h.items()}

    # Compute shared percent-change colorbar range (symmetric, same for both groups)
    all_pcts = np.concatenate([
        *[p[np.isfinite(p)] for p in pct_12h.values()],
        *[p[np.isfinite(p)] for p in pct_72h.values()],
    ])
    pct_limit = np.nanpercentile(np.abs(all_pcts), 95)
    logger.info(f"Shared pct colorbar: ±{pct_limit:.2f}%")

    # Baseline absolute colorbar range (in ppb)
    abs_vmin = 0
    abs_vmax_12h = np.nanpercentile(base_12h * KG_KG_TO_PPB, 98)
    abs_vmax_72h = np.nanpercentile(base_72h * KG_KG_TO_PPB, 98)
    abs_vmax = max(abs_vmax_12h, abs_vmax_72h)

    # === Plots 1-4: +12h rollout ===
    logger.info("Generating +12h plots...")
    plot_absolute_map(base_12h, FIGURES_DIR / "o3_baseline_12h.pdf",
                      vmin=abs_vmin, vmax=abs_vmax)
    for exp_name, label, ppb in EXPERIMENTS[1:]:
        plot_pct_change_map(pct_12h[exp_name],
                            FIGURES_DIR / f"o3_pct_{exp_name}_12h.pdf",
                            vmin=-pct_limit, vmax=pct_limit)

    # === Plots 5-8: +72h rollout ===
    logger.info("Generating +72h plots...")
    plot_absolute_map(base_72h, FIGURES_DIR / "o3_baseline_72h.pdf",
                      vmin=abs_vmin, vmax=abs_vmax)
    for exp_name, label, ppb in EXPERIMENTS[1:]:
        plot_pct_change_map(pct_72h[exp_name],
                            FIGURES_DIR / f"o3_pct_{exp_name}_72h.pdf",
                            vmin=-pct_limit, vmax=pct_limit)

    # === KDE distributions ===
    logger.info("Generating +12h distribution plot...")
    o3_fields_12h = {"base": base_12h}
    o3_fields_12h.update(perturbed_12h)
    plot_kde_distribution(o3_fields_12h, FIGURES_DIR / "o3_distribution_12h.pdf")

    logger.info("Generating +72h distribution plot...")
    o3_fields_72h = {"base": base_72h}
    o3_fields_72h.update(perturbed_72h)
    plot_kde_distribution(o3_fields_72h, FIGURES_DIR / "o3_distribution_72h.pdf")

    # === Stacked composite (3 map rows + distribution row) ===
    logger.info("Generating stacked composite with distributions...")
    plot_stacked_8panel(
        base_12h, base_72h, pct_12h, pct_72h,
        o3_fields_12h, o3_fields_72h,
        abs_vmin, abs_vmax, pct_limit,
        FIGURES_DIR / "o3_stacked_8panel.pdf",
    )

    logger.info(f"\nAll panels saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
