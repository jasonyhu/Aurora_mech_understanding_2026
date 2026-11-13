"""
Cartopy-based map plotting utilities for Aurora experiments.
"""

import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def setup_matplotlib_style(font_size: int = 12, family: str = "STIXGeneral", axes_linewidth: float | None = None):
    """Configure matplotlib for publication-quality plots (STIX maths glyphs, white background)."""
    style = {
        "mathtext.fontset": "stix",
        "font.family": family,
        "font.size": font_size,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    if axes_linewidth is not None:
        style["axes.linewidth"] = axes_linewidth
    plt.rcParams.update(style)


def roll_lon(field: np.ndarray) -> np.ndarray:
    """Shift a [..., W] field on Aurora's 0..360 longitude grid to -180..180."""
    return np.roll(field, field.shape[-1] // 2, axis=-1)


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    """Longitudes in 0..360 -> -180..180."""
    return np.where(lon > 180, lon - 360, lon)


def symmetric_limit(field: np.ndarray, percentile: float = 99) -> float:
    """Symmetric colour limit: the given percentile of |field|, falling back to max|field| (or 1)."""
    v = float(np.nanpercentile(np.abs(field), percentile))
    if not np.isfinite(v) or v == 0:
        v = float(np.nanmax(np.abs(field))) or 1.0
    return v


def draw_global_field(
    ax,
    field: np.ndarray,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    coast_color: str = "#333333",
    coast_width: float = 0.6,
    borders: bool = False,
    border_width: float = 0.4,
    border_color: str = "#999999",
    roll: bool = True,
    **imshow_kwargs,
):
    """Draw a [H, W] lat-lon field (north up) on a global PlateCarree axis.

    Rolls Aurora's 0..360 longitudes to -180..180 (unless roll=False), adds
    coastlines (and optionally country borders) and hides the ticks.
    """
    im = ax.imshow(
        roll_lon(field) if roll else field,
        extent=(-180, 180, -90, 90), origin="upper", transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation=imshow_kwargs.pop("interpolation", "nearest"),
        **imshow_kwargs,
    )
    ax.set_global()
    ax.coastlines(linewidth=coast_width, color=coast_color)
    if borders:
        ax.add_feature(cfeature.BORDERS, linewidth=border_width, edgecolor=border_color)
    ax.set_xticks([])
    ax.set_yticks([])
    return im
