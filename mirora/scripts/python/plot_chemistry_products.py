"""
Generate four publication-ready panels for an Adobe Illustrator composite figure:
  Plot 1: Leighton ratio from CAMS 12h forecast (Sep 1, 2022 00Z init → 12Z valid)
  Plot 2: Leighton ratio from Aurora 12h rollout (same valid time), white where negative
  Plot 3: NOx ratio from CAMS 12h forecast
  Plot 4: NOx ratio from Aurora 12h rollout

Plots 1-2 share a colorbar (viridis). Plots 3-4 share a colorbar (YlOrBr).
Each pair is stacked vertically with no titles (for Illustrator compositing).
"""

import pickle
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 11,
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

FIGURES_DIR = consts.FIGURES_PATH / "chemistry_products"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INIT_STR = "20220901_0000"
LEVEL = 0  # 1000 hPa
STEP = 0   # 12h forecast (forecast_period index 1 for CAMS, step 0 for Aurora)


def load_cams_fields(fp_idx=1, ds=None):
    """Load NO, NO2, O3, T from CAMS at forecast_period index fp_idx
    (fp_idx=1 -> +12h, fp_idx=k -> +12*k h). Reuses an open dataset if given."""
    own = ds is None
    if own:
        ds = xr.open_dataset(consts.CAMS_DATA_PATH / f"cams_{INIT_STR}" / "data_plev.nc")
    sel = dict(forecast_period=fp_idx, forecast_reference_time=0, pressure_level=LEVEL)
    no = ds["no"].isel(**sel).values
    no2 = ds["no2"].isel(**sel).values
    o3 = ds["go3"].isel(**sel).values
    t = ds["t"].isel(**sel).values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    if own:
        ds.close()
    return no, no2, o3, t, lats, lons


def load_aurora_fields(step=0, preds=None):
    """Load NO, NO2, O3, T from Aurora rollout `step` (step 0 -> +12h).
    Reuses preloaded predictions if given."""
    if preds is None:
        pred_path = consts.EXPERIMENTS_PATH / "base" / INIT_STR / "predictions.pickle"
        with open(pred_path, "rb") as f:
            preds = pickle.load(f)
    p = preds[step]
    no = p.atmos_vars["no"][0, 0, LEVEL].numpy()
    no2 = p.atmos_vars["no2"][0, 0, LEVEL].numpy()
    o3 = p.atmos_vars["go3"][0, 0, LEVEL].numpy()
    t = p.atmos_vars["t"][0, 0, LEVEL].numpy()
    lats = p.metadata.lat.numpy()
    lons = p.metadata.lon.numpy()
    return no, no2, o3, t, lats, lons


# Physical constants
N_A = 6.02214076e23        # Avogadro's number (mol⁻¹)
R_GAS = 8.314462           # gas constant (J mol⁻¹ K⁻¹)
M_AIR = 28.97e-3           # molar mass of dry air (kg/mol)
M_NO = 30.01e-3            # molar mass of NO (kg/mol)
M_NO2 = 46.0055e-3         # molar mass of NO2 (kg/mol)
M_O3 = 48.0e-3             # molar mass of O3 (kg/mol)
P_1000HPA = 1000.0 * 100   # 1000 hPa in Pa

# Pressure levels in Pa (matching LEVEL=0 → 1000 hPa)
PRESSURE_PA = P_1000HPA


def mmr_to_number_density(mmr, t, molar_mass, pressure=PRESSURE_PA):
    """Convert mass mixing ratio (kg/kg) to number density (molecules/cm³).

    n = mmr * (N_A * pressure) / (molar_mass * R * T)
    Then convert from molecules/m³ to molecules/cm³ (÷ 1e6).
    """
    n = mmr * (N_A * pressure) / (molar_mass * R_GAS * t)
    return n * 1e-6  # m⁻³ → cm⁻³


def compute_leighton(no, no2, o3, t):
    """J = k3 * [NO] * [O3] / [NO2] in s⁻¹. No masking — for CAMS (no negatives)."""
    no_nd = mmr_to_number_density(no, t, M_NO)
    no2_nd = mmr_to_number_density(no2, t, M_NO2)
    o3_nd = mmr_to_number_density(o3, t, M_O3)

    k3 = 3e-12 * np.exp(-1500.0 / t)
    with np.errstate(divide="ignore", invalid="ignore"):
        J = k3 * no_nd * o3_nd / no2_nd
    return J


def compute_leighton_masked(no, no2, o3, t):
    """J = k3 * [NO] * [O3] / [NO2]. NaN where any species is negative (Aurora)."""
    no_nd = mmr_to_number_density(no, t, M_NO)
    no2_nd = mmr_to_number_density(no2, t, M_NO2)
    o3_nd = mmr_to_number_density(o3, t, M_O3)

    k3 = 3e-12 * np.exp(-1500.0 / t)
    with np.errstate(divide="ignore", invalid="ignore"):
        J = k3 * no_nd * o3_nd / no2_nd
    neg_mask = (no < 0) | (no2 < 0) | (o3 < 0)
    J[neg_mask] = np.nan
    return J


def compute_nox_ratio(no, no2):
    """NOx ratio = [NO2] / ([NO] + [NO2]). No masking — for CAMS."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = no + no2
        ratio = no2 / denom
    return np.clip(ratio, 0, 1)


def compute_nox_ratio_masked(no, no2):
    """NOx ratio, NaN where any species is negative (Aurora)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = no + no2
        ratio = no2 / denom
    neg_mask = (no < 0) | (no2 < 0)
    ratio[neg_mask] = np.nan
    return np.clip(ratio, 0, 1)


def plot_stacked_pair(
    data_left, lats_left, lons_left,
    data_right, lats_right, lons_right,
    vmin, vmax, cmap,
    cbar_label,
    save_path,
    mask_negatives_bot=False,
):
    """Create a side-by-side pair of maps (CAMS left, Aurora right) with a shared bottom colorbar."""
    # Roll longitudes to center on prime meridian
    data_left_rolled = np.roll(data_left, data_left.shape[1] // 2, axis=1)
    data_right_rolled = np.roll(data_right, data_right.shape[1] // 2, axis=1)

    fig = plt.figure(figsize=(16, 5.5))

    # GridSpec: 2 columns for maps, colorbar row on the bottom
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 1],
        height_ratios=[1, 0.05],
        wspace=0.02, hspace=0.08,
        left=0.02, right=0.98, top=0.98, bottom=0.10,
    )

    ax_left = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    ax_right = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=0))
    cbar_ax = fig.add_subplot(gs[1, :])

    for ax, data_rolled, lats, lons in [
        (ax_left, data_left_rolled, lats_left, lons_left),
        (ax_right, data_right_rolled, lats_right, lons_right),
    ]:
        # Set NaN to white by using set_bad on the colormap
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("white")

        im = ax.imshow(
            data_rolled,
            extent=[-180, 180, -90, 90],
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=cmap_obj,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="#333333")
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#888888")

        # No title — for Illustrator compositing
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar spanning both columns
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(cbar_label, fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


N_STEPS = 10  # Aurora rollout steps 0..9 -> +12h..+120h


def main():
    out_dir = FIGURES_DIR / "all_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Aurora predictions + CAMS dataset once...")
    with open(consts.EXPERIMENTS_PATH / "base" / INIT_STR / "predictions.pickle", "rb") as f:
        preds = pickle.load(f)
    ds = xr.open_dataset(consts.CAMS_DATA_PATH / f"cams_{INIT_STR}" / "data_plev.nc")

    # Pass 1: compute all fields, collect finite Leighton values for a shared colorbar.
    per_step = []
    leighton_pool = []
    for step in range(N_STEPS):
        lead = 12 * (step + 1)
        c_no, c_no2, c_o3, c_t, c_lats, c_lons = load_cams_fields(fp_idx=step + 1, ds=ds)
        a_no, a_no2, a_o3, a_t, a_lats, a_lons = load_aurora_fields(step=step, preds=preds)

        cams_leighton = compute_leighton(c_no, c_no2, c_o3, c_t)
        aurora_leighton = compute_leighton_masked(a_no, a_no2, a_o3, a_t)
        cams_nox = compute_nox_ratio(c_no, c_no2)
        aurora_nox = compute_nox_ratio_masked(a_no, a_no2)

        leighton_pool.append(cams_leighton[np.isfinite(cams_leighton)])
        leighton_pool.append(aurora_leighton[np.isfinite(aurora_leighton)])
        per_step.append(dict(
            step=step, lead=lead,
            c_lats=c_lats, c_lons=c_lons, a_lats=a_lats, a_lons=a_lons,
            cams_leighton=cams_leighton, aurora_leighton=aurora_leighton,
            cams_nox=cams_nox, aurora_nox=aurora_nox,
        ))
    ds.close()

    # Shared Leighton colorbar across all steps (2-98 pct of pooled finite values).
    leighton_vmin, leighton_vmax = np.nanpercentile(np.concatenate(leighton_pool), [2, 98])
    leighton_vmin = max(leighton_vmin, 0)
    logger.info(f"Shared Leighton colorbar: [{leighton_vmin:.2e}, {leighton_vmax:.2e}]")

    # Pass 2: render one CAMS-vs-Aurora pair per step for each diagnostic.
    for d in per_step:
        tag = f"step{d['step']:02d}_{d['lead']:03d}h"
        plot_stacked_pair(
            d["cams_leighton"], d["c_lats"], d["c_lons"],
            d["aurora_leighton"], d["a_lats"], d["a_lons"],
            vmin=leighton_vmin, vmax=leighton_vmax, cmap="viridis",
            cbar_label=r"NO$_2$ photolysis rate $J_1$ (s$^{-1}$)",
            save_path=out_dir / f"leighton_{tag}.pdf",
        )
        plot_stacked_pair(
            d["cams_nox"], d["c_lats"], d["c_lons"],
            d["aurora_nox"], d["a_lats"], d["a_lons"],
            vmin=0, vmax=1, cmap="plasma",
            cbar_label=r"NO$_x$ ratio",
            save_path=out_dir / f"nox_ratio_{tag}.pdf",
        )

    logger.info(f"All {N_STEPS}-step panels saved to {out_dir}")


if __name__ == "__main__":
    main()
