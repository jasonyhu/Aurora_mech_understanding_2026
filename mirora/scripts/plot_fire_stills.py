"""
Wildfire case-study SI figures (consts.FIRE_EVENTS), one per event per variable:
  default          — 4x2 maps (CAMS left, Aurora right) at +24h, +48h, +72h, +96h
                     with FINN fire detections overlaid
  --distributions  — CAMS vs Aurora distributions inside the event bbox per lead,
                     plus a CSV of quantiles
"""

import argparse
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep

from mirora.config import MOLAR_MASS
from mirora.data.loading import fetch_rollout, load_finn_fires
from mirora.utils import setup_logging
from mirora.viz.maps import setup_matplotlib_style

import consts

logger = logging.getLogger(__name__)

FIGURES_DIR = consts.FIGURES_PATH / "fire_stills"

# GAMAP WhGrYlRd colormap: White → Light Blue → Green → Yellow → Orange → Red
# Matches Harvard GAMAP IDL `myct, /WhGrYlRd` palette (which interpolates through
# a light blue band between white and green).
WHGRYLRD = mcolors.LinearSegmentedColormap.from_list(
    "WhGrYlRd",
    [
        (0.00, "#FFFFFF"),  # white
        (0.15, "#B3D9FF"),  # light blue
        (0.35, "#1A9850"),  # green
        (0.55, "#FFFF33"),  # yellow
        (0.80, "#FF6600"),  # orange
        (1.00, "#A50026"),  # deep red
    ],
)


SPECIES = {
    # CAMS native units: kg m^-2. Convert kg/m² → mol/m² via × (1000 g/kg) / (M_CO g/mol).
    "tcco": {"label": "Total Column CO (mol m$^{-2}$)", "cmap": WHGRYLRD, "scale": 1000.0 / MOLAR_MASS["co"]},
    # PM2.5 is a particulate mass concentration; mol doesn't apply, leave as g/m³.
    "pm2p5": {"label": "Surface PM$_{2.5}$ (g m$^{-3}$)", "cmap": WHGRYLRD, "scale": 1000.0},
}


@lru_cache
def _prepared_country_geom(country_name: str):
    """A country's polygon from Natural Earth (50m resolution), prepared for point tests."""
    shp_path = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shp_path)
    for record in reader.records():
        attrs = record.attributes
        for key in ("NAME", "NAME_EN", "ADMIN"):
            if attrs.get(key) == country_name:
                return prep(record.geometry)
    raise RuntimeError(f"{country_name} geometry not found in Natural Earth dataset")


def get_finn_country_filter(country_name: str | None):
    """Return a function (fire_lats, fire_lons) -> filtered (fire_lats, fire_lons).

    If country_name is None, returns an identity filter (no geographic constraint).
    """
    if country_name is None:
        return lambda lats, lons: (lats, lons)
    prepared = _prepared_country_geom(country_name)

    def _filter(fire_lats: np.ndarray, fire_lons: np.ndarray):
        mask = np.array([
            prepared.contains(Point(float(lon), float(lat)))
            for lat, lon in zip(fire_lats, fire_lons)
        ], dtype=bool)
        return fire_lats[mask], fire_lons[mask]

    return _filter


def lead_label(step: int) -> str:
    return f"+{12 * (step + 1)}h"


def cams_sfc_path(event: dict):
    return consts.CAMS_DATA_PATH / f"cams_{event['init']}" / "data_sfc.nc"


STEPS = [1, 3, 5, 7]  # +24h, +48h, +72h, +96h


def generate_fire_still(event: dict, var_key: str, var_info: dict):
    lat_min, lat_max, lon_min, lon_max = event["bbox"]
    preds = fetch_rollout(consts.base_rollout_path(event["init"]))
    cams_ds = xr.open_dataset(cams_sfc_path(event))

    scale = var_info.get("scale", 1.0)

    # Compute shared colorbar range across all steps and both models
    all_vals = []
    for step in STEPS:
        cams_field = cams_ds[var_key].isel(forecast_period=step + 1, forecast_reference_time=0).values * scale
        aurora_field = preds[step].surf_vars[var_key][0, 0].numpy() * scale
        all_vals.append(cams_field.ravel())
        all_vals.append(aurora_field.ravel())
    all_vals = np.concatenate(all_vals)
    positive = all_vals[all_vals > 0]
    vmin = 0
    vmax = float(np.nanpercentile(positive, 98)) if len(positive) > 0 else 1

    # Create figure (n_rows rows). Size cells so their aspect matches the
    # bbox's lon/lat aspect — otherwise cartopy centres the map inside a
    # mis-aspected cell and leaves visible padding on the sides.
    n_rows = len(STEPS)
    cell_aspect = (lon_max - lon_min) / (lat_max - lat_min)  # W/H per cell
    target_cell_h = 2.5                                       # inches per row
    target_cell_w = cell_aspect * target_cell_h
    fig_w = (2 * target_cell_w + 0.04 * target_cell_w) / 0.87  # incl. cbar + margins
    fig_h = n_rows * target_cell_h / 0.95
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, 3,
        width_ratios=[1, 1, 0.04],
        height_ratios=[1] * n_rows,
        wspace=0.02, hspace=0.04,
        left=0.03, right=0.90, top=0.90, bottom=0.02,
    )

    # FINN legend, left-aligned, sitting just above the top row of plots
    legend_handle = plt.Line2D(
        [0], [0], marker="o", linestyle="None",
        markerfacecolor="black", markeredgecolor="none",
        markersize=6, label="FINN wildfire detection",
    )
    fig.legend(
        handles=[legend_handle], loc="upper left",
        bbox_to_anchor=(0.03, 0.97),
        fontsize=14, frameon=True, framealpha=0.9,
        fancybox=False, edgecolor="#2d2d2d",
    )

    finn_filter = get_finn_country_filter(event.get("finn_country"))

    cmap_input = var_info["cmap"]
    cmap = plt.get_cmap(cmap_input).copy() if isinstance(cmap_input, str) else cmap_input
    im = None

    for row_idx, step in enumerate(STEPS):
        valid_time = preds[step].metadata.time[0]

        # CAMS
        cams_field = cams_ds[var_key].isel(forecast_period=step + 1, forecast_reference_time=0).values * scale
        cams_lats = cams_ds["latitude"].values
        cams_lons = cams_ds["longitude"].values

        # Aurora
        aurora_field = preds[step].surf_vars[var_key][0, 0].numpy() * scale
        aurora_lats = preds[step].metadata.lat.numpy()
        aurora_lons = preds[step].metadata.lon.numpy()

        # FINN fires for this date
        finn = load_finn_fires(consts.FINN_DIR, event["name"], valid_time)

        for col_idx, (field, lats, lons, panel_title) in enumerate([
            (cams_field, cams_lats, cams_lons, "CAMS" if row_idx == 0 else ""),
            (aurora_field, aurora_lats, aurora_lons, "Aurora" if row_idx == 0 else ""),
        ]):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
            lon2d, lat2d = np.meshgrid(lons, lats)
            im = ax.pcolormesh(
                lon2d, lat2d, field,
                transform=ccrs.PlateCarree(),
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading="auto", rasterized=True,
            )
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f6", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)
            ax.coastlines(resolution="50m", linewidth=0.6, color="#2d2d2d", zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#888888", zorder=3)

            # FINN overlay (black dots) — optionally filtered to a country polygon
            if finn is not None:
                fire_lats, fire_lons = finn
                fire_lats, fire_lons = finn_filter(fire_lats, fire_lons)
                if len(fire_lats) > 0:
                    ax.scatter(
                        fire_lons, fire_lats,
                        transform=ccrs.PlateCarree(),
                        s=8, c="black", marker="o", alpha=0.85,
                        edgecolors="none", linewidths=0.0, zorder=5,
                    )

    # Shared colorbar
    assert im is not None
    cbar_ax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", extend="max")
    cbar.set_label(var_info["label"], fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    cams_ds.close()

    sub_dir = FIGURES_DIR / "no_labels"
    sub_dir.mkdir(parents=True, exist_ok=True)
    save_path = sub_dir / f"{event['name']}_{var_key}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {save_path.name}")


DIST_STEPS = [0, 1, 3, 5, 7, 9]  # +12h .. +120h


def generate_fire_distribution(event: dict, var_key: str, var_info: dict):
    """SI figure: CAMS vs Aurora distribution of `var_key` inside the event bbox at each
    lead time (log-density, to show the high-concentration tail), plus a CSV of quantiles."""
    lat_min, lat_max, lon_min, lon_max = event["bbox"]
    preds = fetch_rollout(consts.base_rollout_path(event["init"]))
    cams_ds = xr.open_dataset(cams_sfc_path(event))
    scale = var_info.get("scale", 1.0)

    lat = preds[0].metadata.lat.numpy()
    lon = (preds[0].metadata.lon.numpy() + 180) % 360 - 180  # 0..360 -> -180..180
    box = ((lat >= lat_min) & (lat <= lat_max))[:, None] & ((lon >= lon_min) & (lon <= lon_max))[None, :]

    fields = {}
    for step in DIST_STEPS:
        cams = cams_ds[var_key].isel(forecast_period=step + 1, forecast_reference_time=0).values[: len(lat)] * scale
        fields[step] = (cams[box], preds[step].surf_vars[var_key][0, 0].numpy()[box] * scale)
    cams_ds.close()

    pooled = np.concatenate([x for pair in fields.values() for x in pair])
    bins = np.linspace(*np.nanpercentile(pooled, [0.01, 99.99]), 60)  # covers both sources
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True, layout="constrained")
    rows = []
    for ax, step in zip(axs.flat, DIST_STEPS):
        label = lead_label(step)
        cams, aurora = fields[step]
        ax.hist(cams, bins=bins, density=True, histtype="step", linewidth=2.5, color="#2d2d2d", label="CAMS")
        ax.hist(aurora, bins=bins, density=True, histtype="step", linewidth=2.5, color="#C44E52", label="Aurora")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=22)
        ax.tick_params(labelsize=20)
        for name, x in (("CAMS", cams), ("Aurora", aurora)):
            q = np.nanpercentile(x, [50, 90, 99, 100])
            rows.append({"event": event["name"], "lead": label, "source": name,
                         "p50": q[0], "p90": q[1], "p99": q[2], "max": q[3]})
    axs[0, 0].legend(fontsize=20, frameon=False)
    fig.supxlabel(var_info["label"], fontsize=22)
    fig.supylabel("Probability density", fontsize=22)

    out_dir = FIGURES_DIR / "distributions"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{event['name']}_{var_key}_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / f"{event['name']}_{var_key}_quantiles.csv", index=False)
    logger.info(f"  {event['name']}: distribution + quantiles saved to {out_dir}")


def main():
    setup_logging()
    setup_matplotlib_style(20, family="sans-serif")
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", help="Only generate for this event name")
    parser.add_argument("--var", help="Only generate for this variable key (tcco / pm2p5)")
    parser.add_argument("--distributions", action="store_true",
                        help="Plot CAMS vs Aurora distributions per lead (SI figure) instead of the map stills")
    args = parser.parse_args()

    events = [e for e in consts.FIRE_EVENTS if args.event is None or e["name"] == args.event]
    species = {k: v for k, v in SPECIES.items() if args.var is None or k == args.var}
    if args.event and not events:
        raise SystemExit(f"Unknown event: {args.event}")
    if args.var and not species:
        raise SystemExit(f"Unknown variable: {args.var}")

    for event in events:
        for var_key, var_info in species.items():
            logger.info(f"Generating {event['label']} — {var_key}...")
            if args.distributions:
                generate_fire_distribution(event, var_key, var_info)
            else:
                generate_fire_still(event, var_key, var_info)

    logger.info(f"\nFire stills saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
