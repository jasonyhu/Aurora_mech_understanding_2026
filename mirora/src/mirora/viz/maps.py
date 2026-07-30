"""
Cartopy-based map plotting utilities for Aurora experiments.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def setup_matplotlib_style(font_size: int = 12):
    """Configure matplotlib for publication-quality plots with STIX fonts."""
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": font_size,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def plot_rollout_grid(
    data: list[np.ndarray],
    times: list[datetime],
    title: str,
    save_path: Path,
    cmap: str = "Blues",
    cbar_label: str = "concentration",
):
    """Plot a 2×5 grid of 10-step rollout predictions."""
    rolled = [np.roll(d, d.shape[1] // 2, axis=1) for d in data]
    vmin, vmax = np.percentile(np.stack(rolled), [1, 99])

    fig, axs = plt.subplots(
        2, 5, figsize=(20, 7), layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(title, fontsize=16)

    im = None
    for ax, field, t in zip(axs.flatten(order="F"), rolled, times):
        im = ax.imshow(field, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=(-180, 180, -90, 90), zorder=0)
        ax.coastlines(linewidth=0.5, color="gray", zorder=1)
        ax.set_title(f"{t.strftime('%m-%d %H')}Z", pad=4, fontsize=9)
        ax.axis("off")

    assert im is not None
    cbar = fig.colorbar(im, ax=axs, orientation="horizontal", aspect=60, fraction=0.05, pad=0.05)
    cbar.set_label(cbar_label, fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_abs_diff_grid(
    diffs: list[np.ndarray],
    times: list[datetime],
    title: str,
    save_path: Path,
    cmap: str = "RdBu_r",
    cbar_label: str = "difference",
):
    """Plot a 2×5 grid of absolute differences (experiment - base)."""
    rolled = [np.roll(d, d.shape[1] // 2, axis=1) for d in diffs]
    v_limit = np.percentile(np.abs(np.stack(rolled)), 95)

    fig, axs = plt.subplots(
        2, 5, figsize=(20, 7), layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(title, fontsize=16)

    im = None
    for ax, field, t in zip(axs.flatten(order="F"), rolled, times):
        im = ax.imshow(field, vmin=-v_limit, vmax=v_limit, cmap=cmap,
                       extent=(-180, 180, -90, 90), zorder=0)
        ax.coastlines(linewidth=0.5, color="black", zorder=1)
        ax.set_title(f"{t.strftime('%m-%d %H')}Z", pad=4, fontsize=9)
        ax.axis("off")

    assert im is not None
    cbar = fig.colorbar(im, ax=axs, orientation="horizontal", aspect=60, fraction=0.05, pad=0.05)
    cbar.set_label(cbar_label, fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pct_diff_grid(
    diff_pcts: list[np.ndarray],
    times: list[datetime],
    title: str,
    save_path: Path,
    cmap: str = "PRGn_r",
    cbar_label: str = "difference (%)",
    clip_range: tuple[float, float] = (-500, 500),
):
    """Plot a 2×5 grid of percent differences."""
    rolled = [np.roll(d, d.shape[1] // 2, axis=1) for d in diff_pcts]
    clipped = [np.clip(d, *clip_range) for d in rolled]
    v_limit = np.percentile(np.abs(np.stack(clipped)), 95)

    fig, axs = plt.subplots(
        2, 5, figsize=(20, 7), layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(title, fontsize=16)

    im = None
    for ax, field, t in zip(axs.flatten(order="F"), clipped, times):
        im = ax.imshow(field, vmin=-v_limit, vmax=v_limit, cmap=cmap,
                       extent=(-180, 180, -90, 90), zorder=0)
        ax.coastlines(linewidth=0.5, color="black", zorder=1)
        ax.set_title(f"{t.strftime('%m-%d %H')}Z", pad=4, fontsize=9)
        ax.axis("off")

    assert im is not None
    cbar = fig.colorbar(im, ax=axs, orientation="horizontal", aspect=60, fraction=0.05, pad=0.05)
    cbar.set_label(cbar_label, fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
