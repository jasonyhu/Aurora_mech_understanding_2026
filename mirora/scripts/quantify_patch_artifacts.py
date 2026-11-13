"""
Quantify the 3x3 patch mosaic in Aurora's surface chemistry fields by species
and region (SI table), Sep 1 2022 00 UTC.

Metric: mean |finite difference| across patch boundaries divided by the mean
|finite difference| inside patches, in both directions. A ratio of 1 means no
mosaic. Aurora's 450x900 output divides evenly into 3x3 patches, so the
difference f[i+1] - f[i] crosses a patch edge when (i + 1) % 3 == 0, i.e. at
index i % 3 == 2 (confirmed empirically: it is Aurora's max-gradient phase for
every species). CAMS fields at the same lead serve as a control (no patch
grid, so their ratio should be ~1).

Regions, defined from the CAMS surface NO2 at the same lead:
high-NOx = top 10% of cells; remote = bottom 50%; global = all.

Writes analysis/patch_artifacts/patch_ratio_{sep01}.csv and prints the table.
--plot: figure of CAMS vs Aurora crops (North China Plain, +12 h) with the 3x3
patch edges drawn, plus the gradient at each within-patch position.
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mirora.config import CHEM_ATMOS_VARS, MMR_TO_PPB, PATCH_SIZE as PATCH, SURFACE_LEVEL_IDX as LEVEL_IDX
from mirora.data.loading import fetch_rollout, surface
from mirora.stats import lat_weights
from mirora.utils import get_cams_forecast_path, setup_logging
from mirora.viz.maps import setup_matplotlib_style

import consts

logger = logging.getLogger(__name__)

INIT_TIME = consts.MAIN_INIT_TIME
BOUNDARY_PHASE = 2  # diff index mod 3 that crosses a patch edge
SPECIES = list(CHEM_ATMOS_VARS)
STEPS = [0, 4, 9]  # +12h, +60h, +120h
OUT_DIR = consts.SCRATCH_PATH / "analysis" / "patch_artifacts"


def gradient_by_phase(field: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Mean |finite difference| over `region` for each phase (index mod 3), pooling x and y."""
    gx, gy = np.abs(np.diff(field, axis=1)), np.abs(np.diff(field, axis=0))
    rx, ry = region[:, :-1], region[:-1, :]
    by_phase = []
    for ph in range(PATCH):
        cols = np.arange(gx.shape[1]) % PATCH == ph
        rows = np.arange(gy.shape[0]) % PATCH == ph
        vals = np.concatenate([gx[:, cols][rx[:, cols]], gy[rows, :][ry[rows, :]]])
        by_phase.append(vals.mean())
    return np.array(by_phase)


def phase_ratio(field: np.ndarray, region: np.ndarray) -> tuple[float, int]:
    """Boundary/interior gradient ratio over `region`.
    Returns (ratio at the patch-edge phase, phase with the largest gradient)."""
    by_phase = gradient_by_phase(field, region)
    return float(by_phase[BOUNDARY_PHASE] / np.delete(by_phase, BOUNDARY_PHASE).mean()), int(by_phase.argmax())


def plot_patch_crops(step: int = 0, lat0: float = 42.0, lon0: float = 110.0, size: int = 30):
    """Figure: per species, a CAMS and an Aurora crop (size x size cells, starting at
    lat0/lon0, snapped to Aurora's patch grid) with the 3x3 patch edges drawn, plus the
    mean gradient at each within-patch position globally (Aurora vs CAMS)."""
    preds = fetch_rollout(consts.base_rollout_path(INIT_TIME))
    lat, lon = preds[0].metadata.lat.numpy(), preds[0].metadata.lon.numpy()
    r0 = int(np.argmin(np.abs(lat - lat0))) // PATCH * PATCH  # patch edges sit at multiples of 3
    c0 = int(np.argmin(np.abs(lon - lon0))) // PATCH * PATCH
    rows, cols = slice(r0, r0 + size), slice(c0, c0 + size)
    extent = (lon[c0] - 0.2, lon[c0 + size - 1] + 0.2, lat[r0 + size - 1] - 0.2, lat[r0] + 0.2)
    ds = xr.open_dataset(get_cams_forecast_path(consts.CAMS_DATA_PATH, INIT_TIME) / "data_plev.nc")
    sel = dict(forecast_period=step + 1, forecast_reference_time=0, pressure_level=LEVEL_IDX)
    labels = {sp: CHEM_ATMOS_VARS[sp] for sp in SPECIES}

    setup_matplotlib_style(18)
    fig, axs = plt.subplots(len(SPECIES), 3, figsize=(17, 4.6 * len(SPECIES)),
                            gridspec_kw={"width_ratios": [1, 1.12, 1.1], "wspace": 0.30, "hspace": 0.35})
    everywhere = np.ones((len(lat), len(lon)), dtype=bool)
    for i, sp in enumerate(SPECIES):
        a = surface(preds[step], sp).astype(np.float64) * MMR_TO_PPB[sp]
        c = ds[sp].isel(**sel).values[:len(lat)].astype(np.float64) * MMR_TO_PPB[sp]
        vmin, vmax = np.percentile(c[rows, cols], [2, 98])
        for j, (name, f) in enumerate([("CAMS", c), ("Aurora", a)]):
            ax = axs[i, j]
            im = ax.imshow(f[rows, cols], extent=extent, origin="upper", cmap="viridis",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            for k in range(0, size + 1, PATCH):  # 3x3 patch edges
                ax.axhline(lat[r0] + 0.2 - 0.4 * k, color="white", linewidth=0.6, alpha=0.8)
                ax.axvline(lon[c0] - 0.2 + 0.4 * k, color="white", linewidth=0.6, alpha=0.8)
            ax.set_title(f"{name} {labels[sp]}", fontsize=20)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            if name == "Aurora":  # one colorbar per row; both crops share its limits
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend="both")
                cb.set_label(f"{labels[sp]} (ppb)")
        ax = axs[i, 2]
        x = np.arange(PATCH)
        for off, (name, f, color) in zip((-0.2, 0.2), [("CAMS", c, "#7f7f7f"), ("Aurora", a, "#d62728")]):
            g = gradient_by_phase(f, everywhere)
            ax.bar(x + off, g / np.delete(g, BOUNDARY_PHASE).mean(), width=0.4, color=color, label=name)
        ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, ["inside", "inside", "patch edge"])
        ax.set_ylabel("Mean |gradient|\n(÷ inside mean)")
        ax.yaxis.tick_right()  # keep clear of the Aurora colorbar label
        ax.yaxis.set_label_position("right")
        ax.set_title(f"{labels[sp]}: gradient by position", fontsize=20)
        if i == 0:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    ds.close()
    out = consts.FIGURES_PATH / "patch_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"patch_crops_step{step}.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path} (crop rows {r0}-{r0 + size}, cols {c0}-{c0 + size})")


def main():
    setup_logging()
    if sys.argv[1:] == ["--plot"]:
        return plot_patch_crops()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = fetch_rollout(consts.base_rollout_path(INIT_TIME))
    n_lat = len(lat_weights(preds))
    ds = xr.open_dataset(get_cams_forecast_path(consts.CAMS_DATA_PATH, INIT_TIME) / "data_plev.nc")

    rows = []
    for step in STEPS:
        sel = dict(forecast_period=step + 1, forecast_reference_time=0, pressure_level=LEVEL_IDX)
        cams_no2 = ds["no2"].isel(**sel).values[:n_lat]
        regions = {"global": np.ones_like(cams_no2, dtype=bool),
                   "high-NOx (top 10%)": cams_no2 >= np.percentile(cams_no2, 90),
                   "remote (bottom 50%)": cams_no2 <= np.percentile(cams_no2, 50)}
        for sp in SPECIES:
            fields = {"Aurora": surface(preds[step], sp).astype(np.float64),
                      "CAMS": ds[sp].isel(**sel).values[:n_lat].astype(np.float64)}
            for source, f in fields.items():
                for rname, mask in regions.items():
                    ratio, phase = phase_ratio(f, mask)
                    rows.append({"lead_hours": 12 * (step + 1), "species": sp, "source": source,
                                 "region": rname, "boundary_interior_ratio": ratio, "boundary_phase": phase})
    ds.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "patch_ratio_sep01.csv", index=False)
    table = df.pivot_table(index=["lead_hours", "species"], columns=["source", "region"],
                           values="boundary_interior_ratio")
    pd.set_option("display.width", 250)
    logger.info(f"Boundary/interior gradient ratio (1 = no mosaic):\n{table.round(2).to_string()}")
    logger.info(f"Max-gradient phases (Aurora; 2 = patch edge): {sorted(df[df.source == 'Aurora'].boundary_phase.unique())}")


if __name__ == "__main__":
    main()
