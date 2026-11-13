"""
Distribution figures for the multi-initialization +5 ppb NOx robustness sweep
(do the single-init Sep 1 2022 statistics generalize across initializations
and seasons?).

Reads the per-init partial CSVs written by run_nox_robustness_sweep.py and
produces two figures in NOX_FIGURES_PATH/nox_robustness/:

  1. nox_robustness_violins.{pdf,png} — 2x2 violin panels: median ΔO3 and
     area-weighted % cells suppressed, at 12h and 72h, split by season
     (All / JJA / SON / DJF / MAM). The Sep 1 2022 00Z single-init value the
     paper reports is overplotted as a star, to show it sits within the spread.

  2. nox_robustness_leadtime.{pdf,png} — 2 panels vs lead time, mean with
     inter-init IQR band: (left) global median ΔO3 vs mean|ΔO3| (the median
     collapses toward zero while the magnitude persists — the "cancellation
     artifact"); (right) global vs tropical median ΔO3 (the tropics reverse
     from suppression at 12h to a large increase by 72h).

USAGE: python3 plot_nox_robustness.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mirora.viz.maps import setup_matplotlib_style

import consts
from aggregate_nox_robustness import load_partials

setup_matplotlib_style(20, axes_linewidth=0.8)

OUT_DIR = consts.NOX_FIGURES_PATH / "nox_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP1 = consts.MAIN_INIT_STR          # the paper's single init
GROUPS = ["All", "JJA", "SON", "DJF", "MAM"]
SEASON_COLORS = {
    "All": "#4C4C4C",
    "JJA": "#C44E52",   # summer  — warm
    "SON": "#E5AE38",   # autumn  — amber
    "DJF": "#4878CF",   # winter  — cool
    "MAM": "#6AB572",   # spring  — green
}


def _group_values(df: pd.DataFrame, step: int, col: str):
    """Return list of value-arrays, one per GROUP, for a violinplot."""
    d = df[df["step"] == step]
    out = []
    for g in GROUPS:
        sub = d if g == "All" else d[d["season"] == g]
        v = sub[col].to_numpy()
        out.append(v[np.isfinite(v)])
    return out


def _violin_panel(ax, df, step, col, scale, ylabel, ref=None, ref_label=None):
    data = _group_values(df, step, col)
    positions = np.arange(len(GROUPS))
    parts = ax.violinplot([d * scale for d in data], positions=positions,
                          showmeans=False, showmedians=True, showextrema=False,
                          widths=0.8)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(SEASON_COLORS[GROUPS[i]])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.6)

    if ref is not None:
        ax.axhline(ref, color="0.4", ls="--", lw=1.4, zorder=0,
                   label=ref_label)

    # Overplot the Sep 1 2022 single-init value as a star (at 'All' and its season).
    sep = df[(df["step"] == step) & (df["init_time"] == SEP1)]
    if not sep.empty:
        y = float(sep[col].iloc[0]) * scale
        for xpos in (0, GROUPS.index("SON")):
            ax.scatter([xpos], [y], marker="*", s=420, color="white",
                       edgecolor="black", linewidth=1.3, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    if ref_label is not None:
        ax.legend(loc="best", fontsize=15, framealpha=0.9)


def figure_violins(df):
    fig, axs = plt.subplots(2, 2, figsize=(17, 13), layout="constrained")

    _violin_panel(axs[0, 0], df, 0, "diff_of_medians", 1.0,
                  r"median $\Delta$O$_3$ (ppb)", ref=0.0)
    axs[0, 0].set_title("12 h lead", fontsize=22)

    _violin_panel(axs[0, 1], df, 5, "diff_of_medians", 1.0,
                  r"median $\Delta$O$_3$ (ppb)", ref=0.0)
    axs[0, 1].set_title("72 h lead", fontsize=22)

    _violin_panel(axs[1, 0], df, 0, "frac_suppressed_areaw", 100.0,
                  r"O$_3$ suppressed (area-wtd %)", ref=89.3,
                  ref_label="Sep 1 (89.3%)")
    _violin_panel(axs[1, 1], df, 5, "frac_suppressed_areaw", 100.0,
                  r"O$_3$ suppressed (area-wtd %)", ref=50.0,
                  ref_label="50%")

    fig.suptitle(r"+5 ppb NO$_x$ surface-O$_3$ response across 730 initializations",
                 fontsize=24)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"nox_robustness_violins.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def _band(ax, df, col, color, label):
    """Plot inter-init mean with IQR band vs lead time for a metric column."""
    steps = sorted(df["step"].unique())
    leads = [(s + 1) * 12 for s in steps]
    mean, q25, q75 = [], [], []
    for s in steps:
        v = df[df["step"] == s][col].to_numpy()
        v = v[np.isfinite(v)]
        mean.append(np.mean(v))
        q25.append(np.percentile(v, 25))
        q75.append(np.percentile(v, 75))
    ax.plot(leads, mean, "-o", color=color, lw=2.2, ms=7, label=label)
    ax.fill_between(leads, q25, q75, color=color, alpha=0.2)


def figure_leadtime(df):
    fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), layout="constrained")

    # Left: magnitude persists while median collapses.
    _band(axs[0], df, "median_of_diff", "#4878CF", r"median $\Delta$O$_3$")
    _band(axs[0], df, "mean_absdiff", "#C44E52", r"mean $|\Delta$O$_3|$")
    axs[0].axhline(0, color="0.4", ls="--", lw=1.2)
    axs[0].set_xlabel("lead time (h)")
    axs[0].set_ylabel(r"$\Delta$O$_3$ (ppb)")
    axs[0].set_title("Magnitude persists; median collapses", fontsize=21)
    axs[0].legend(loc="best", fontsize=16)
    axs[0].grid(ls=":", lw=0.6, alpha=0.6)

    # Right: tropical reversal vs global.
    _band(axs[1], df, "median_of_diff", "#4C4C4C", "global median")
    _band(axs[1], df, "trop_median_of_diff", "#E5AE38",
          r"tropical median (|lat|<23.5$\degree$)")
    axs[1].axhline(0, color="0.4", ls="--", lw=1.2)
    axs[1].set_xlabel("lead time (h)")
    axs[1].set_ylabel(r"median $\Delta$O$_3$ (ppb)")
    axs[1].set_title("Tropics reverse: suppression $\\to$ increase", fontsize=21)
    axs[1].legend(loc="best", fontsize=16)
    axs[1].grid(ls=":", lw=0.6, alpha=0.6)

    fig.suptitle(r"+5 ppb NO$_x$: inter-initialization mean and IQR band (n=730)",
                 fontsize=24)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"nox_robustness_leadtime.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_partials()
    print(f"Loaded {df['init_time'].nunique()} inits, {len(df)} rows.")
    figure_violins(df)
    figure_leadtime(df)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
