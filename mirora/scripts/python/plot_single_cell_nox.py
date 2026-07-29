"""
SI figure: a single-grid-cell NOx spike (Beijing, +100 ppb, all levels) produces
non-local, global changes in Aurora's surface forecast. Plots global dO3 and dNO2
maps at +12h and +72h, marking the source cell.
"""

import pickle
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 20,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INIT_STR = "20220901_0000"
LEVEL = 0
EXP_NAME = "single_cell_nox_beijing"
TARGET_LAT, TARGET_LON = 39.9, 116.4
FIG_DIR = consts.FIGURES_PATH / "single_cell_nox"
FIG_DIR.mkdir(parents=True, exist_ok=True)

M_AIR, M_O3, M_NO2 = 28.97, 48.0, 46.0055
O3_PPB = (M_AIR / M_O3) * 1e9
NO2_PPT = (M_AIR / M_NO2) * 1e12


def load(exp):
    path = consts.EXPERIMENTS_PATH / exp / INIT_STR / "predictions.pickle"
    with open(path, "rb") as f:
        return pickle.load(f)


def field(preds, var, step):
    return preds[step].atmos_vars[var][0, 0, LEVEL].numpy().astype(np.float64)


def plot_map(diff, lat, lon, src_xy, label, unit, save_path):
    rolled = np.roll(diff, diff.shape[1] // 2, axis=1)
    vlim = float(np.nanpercentile(np.abs(rolled), 99.5)) or 1.0
    fig, ax = plt.subplots(figsize=(13, 6.5),
                           subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})
    im = ax.imshow(rolled, extent=[-180, 180, -90, 90], origin="upper",
                   transform=ccrs.PlateCarree(), cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_global()
    ax.coastlines(linewidth=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#999999")
    # mark source cell
    sx = src_xy[1] if src_xy[1] <= 180 else src_xy[1] - 360
    ax.plot(sx, src_xy[0], marker="*", color="black", markersize=22,
            markeredgecolor="white", markeredgewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", aspect=40, fraction=0.05, pad=0.04, extend="both")
    cbar.set_label(f"{label} ({unit})", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {save_path.name}  (|max|={np.nanmax(np.abs(rolled)):.3g} {unit})")


def main():
    base = load("base")
    sc = load(EXP_NAME)
    lat = np.asarray(base[0].metadata.lat, dtype=np.float64)
    lon = np.asarray(base[0].metadata.lon, dtype=np.float64)

    for step, lead in [(0, 12), (5, 72)]:
        do3 = (field(sc, "go3", step) - field(base, "go3", step)) * O3_PPB
        dno2 = (field(sc, "no2", step) - field(base, "no2", step)) * NO2_PPT
        w = np.clip(np.cos(np.deg2rad(lat)), 0, None)[:, None]
        reach = (np.abs(do3) > 0.5)
        frac = float((reach * np.broadcast_to(w, reach.shape)).sum()
                     / np.broadcast_to(w, reach.shape).sum() * 100)
        logger.info(f"+{lead}h: global mean|dO3|={np.abs(do3).mean():.4f} ppb, "
                    f"area-wt fraction |dO3|>0.5ppb = {frac:.2f}%")
        plot_map(do3, lat, lon, (TARGET_LAT, TARGET_LON), r"$\Delta$O$_3$", "ppb",
                 FIG_DIR / f"single_cell_dO3_{lead:03d}h.pdf")
        plot_map(dno2, lat, lon, (TARGET_LAT, TARGET_LON), r"$\Delta$NO$_2$", "ppt",
                 FIG_DIR / f"single_cell_dNO2_{lead:03d}h.pdf")


if __name__ == "__main__":
    main()
