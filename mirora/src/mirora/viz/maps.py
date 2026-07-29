"""
Cartopy-based map plotting utilities for Aurora experiments.
"""

import matplotlib.pyplot as plt


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
