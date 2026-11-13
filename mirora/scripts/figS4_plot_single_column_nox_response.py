"""
Figure S4 (Text S1): a single-grid-cell NOx spike (Beijing, +100 ppb, all levels) produces
non-local, global changes in Aurora's surface forecast. Plots global dO3 and dNO2
maps at +12h and +72h, marking the source cell.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from mirora.config import MMR_TO_PPB
from mirora.data.loading import fetch_rollout, surface
from mirora.stats import lat_weights, wmean
from mirora.utils import setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style

import consts
from figS4_run_single_column_nox_perturbation import EXP_NAME, TARGET_LAT, TARGET_LON

logger = logging.getLogger(__name__)

setup_matplotlib_style(20)

FIG_DIR = consts.NOX_FIGURES_PATH / "single_cell_nox"
FIG_DIR.mkdir(parents=True, exist_ok=True)

O3_PPB = MMR_TO_PPB["go3"]
NO2_PPT = MMR_TO_PPB["no2"] * 1e3


def field(preds, var, step):
    return surface(preds[step], var).astype(np.float64)


def plot_map(diff, label, unit, save_path):
    vlim = float(np.nanpercentile(np.abs(diff), 99.5)) or 1.0
    fig, ax = plt.subplots(figsize=(13, 6.5),
                           subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})
    im = draw_global_field(ax, diff, "RdBu_r", -vlim, vlim, borders=True)
    # mark source cell (TARGET_LON is already in -180..180)
    ax.plot(TARGET_LON, TARGET_LAT, marker="*", color="black", markersize=22,
            markeredgecolor="white", markeredgewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", aspect=40, fraction=0.05, pad=0.04, extend="both")
    cbar.set_label(f"{label} ({unit})", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {save_path.name}  (|max|={np.nanmax(np.abs(diff)):.3g} {unit})")


def main():
    setup_logging()
    base = fetch_rollout(consts.base_rollout_path(consts.MAIN_INIT_STR))
    sc = fetch_rollout(consts.rollout_path(consts.NOX_EXPERIMENTS_PATH / EXP_NAME, consts.MAIN_INIT_STR))

    for step, lead in [(0, 12), (5, 72)]:
        do3 = (field(sc, "go3", step) - field(base, "go3", step)) * O3_PPB
        dno2 = (field(sc, "no2", step) - field(base, "no2", step)) * NO2_PPT
        frac = wmean(np.abs(do3) > 0.5, lat_weights(base)) * 100
        logger.info(f"+{lead}h: global mean|dO3|={np.abs(do3).mean():.4f} ppb, "
                    f"area-wt fraction |dO3|>0.5ppb = {frac:.2f}%")
        plot_map(do3, r"$\Delta$O$_3$", "ppb",
                 FIG_DIR / f"single_cell_dO3_{lead:03d}h.pdf")
        plot_map(dno2, r"$\Delta$NO$_2$", "ppt",
                 FIG_DIR / f"single_cell_dNO2_{lead:03d}h.pdf")


if __name__ == "__main__":
    main()
