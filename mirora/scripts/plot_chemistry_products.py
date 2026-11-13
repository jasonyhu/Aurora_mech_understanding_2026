"""
CAMS-vs-Aurora maps of two photochemical diagnostics at 1000 hPa for the main
init (Sep 1 2022 00Z), one figure per rollout step (+12h..+120h):
  leighton_stepNN_HHHh.pdf  — NO2 photolysis rate J from the Leighton relation
                              (viridis, colorbar shared across all steps)
  nox_ratio_stepNN_HHHh.pdf — NOx partitioning NO2/(NO+NO2) (plasma, 0-1)
CAMS on the left, Aurora on the right (white where any Aurora species is
nonpositive); no titles, for compositing.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

from mirora.chemistry import (compute_leighton, compute_leighton_masked, compute_nox_ratio,
                              compute_nox_ratio_masked, load_aurora_fields, load_cams_fields)
from mirora.data.loading import fetch_rollout
from mirora.utils import get_cams_forecast_path, setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style
import consts

logger = logging.getLogger(__name__)

FIGURES_DIR = consts.FIGURES_PATH / "chemistry_products"
N_STEPS = 10  # Aurora rollout steps 0..9 -> +12h..+120h


def plot_stacked_pair(data_left, data_right, vmin, vmax, cmap, cbar_label, save_path):
    """Create a side-by-side pair of maps (CAMS left, Aurora right) with a shared bottom colorbar."""
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

    # NaN (masked) cells are drawn white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    # No titles — for Illustrator compositing
    for ax, data in [(ax_left, data_left), (ax_right, data_right)]:
        im = draw_global_field(ax, data, cmap_obj, vmin, vmax,
                               borders=True, border_width=0.6, border_color="#888888")

    # Shared colorbar spanning both columns
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(cbar_label, fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def main():
    setup_logging()
    setup_matplotlib_style(11, axes_linewidth=0.6)
    out_dir = FIGURES_DIR / "all_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Aurora predictions + CAMS dataset once...")
    preds = fetch_rollout(consts.base_rollout_path(consts.MAIN_INIT_STR))
    ds = xr.open_dataset(get_cams_forecast_path(consts.CAMS_DATA_PATH, consts.MAIN_INIT_TIME) / "data_plev.nc")

    # Pass 1: compute all fields, collect finite Leighton values for a shared colorbar.
    per_step = []
    leighton_pool = []
    for step in range(N_STEPS):
        c_no, c_no2, c_o3, c_t = load_cams_fields(ds, step + 1)
        a_no, a_no2, a_o3, a_t = load_aurora_fields(preds[step])

        cams_leighton = compute_leighton(c_no, c_no2, c_o3, c_t)
        aurora_leighton = compute_leighton_masked(a_no, a_no2, a_o3, a_t)

        leighton_pool.append(cams_leighton[np.isfinite(cams_leighton)])
        leighton_pool.append(aurora_leighton[np.isfinite(aurora_leighton)])
        per_step.append(dict(
            step=step, lead=12 * (step + 1),
            cams_leighton=cams_leighton, aurora_leighton=aurora_leighton,
            cams_nox=compute_nox_ratio(c_no, c_no2), aurora_nox=compute_nox_ratio_masked(a_no, a_no2),
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
            d["cams_leighton"], d["aurora_leighton"],
            vmin=leighton_vmin, vmax=leighton_vmax, cmap="viridis",
            cbar_label=r"NO$_2$ photolysis rate $J_1$ (s$^{-1}$)",
            save_path=out_dir / f"leighton_{tag}.pdf",
        )
        plot_stacked_pair(
            d["cams_nox"], d["aurora_nox"],
            vmin=0, vmax=1, cmap="plasma",
            cbar_label=r"NO$_x$ ratio",
            save_path=out_dir / f"nox_ratio_{tag}.pdf",
        )

    logger.info(f"All {N_STEPS}-step panels saved to {out_dir}")


if __name__ == "__main__":
    main()
