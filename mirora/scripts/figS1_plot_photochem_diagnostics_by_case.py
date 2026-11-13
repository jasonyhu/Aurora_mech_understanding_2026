"""
Figure S1: the Figure 2 photochemical diagnostics across the
diurnal cycle, season and lead time. One row per case:
  1. valid 12 UTC: 1 Sep 2022 00 UTC init, +12 h (= Figure 2)
  2. valid 00 UTC: 1 Sep 2022 12 UTC init, +12 h
  3. DJF composite of the +12 h fields over every init valid in Dec-Feb
  4. JJA composite (Jun-Aug)
  5. +120 h: 1 Sep 2022 00 UTC init
Columns: CAMS NOx ratio, Aurora NOx ratio, CAMS J1, Aurora J1 (Figure 2 colormaps).

Composites average each init's diagnostic field, never the concentrations:
Aurora is masked per init where NO/NO2 (NOx ratio) or NO/NO2/O3 (J1) is
nonpositive, so cells that go nonpositive only on some dates are still masked
on those dates. A composite cell is left white when it was masked on at least
half of the inits. The reported masked fraction is the area-weighted fraction
per init, averaged over inits.

Night check: CAMS J1 should vanish where the sun is down. For each single-time
case we compare J1 over night-side cells (solar zenith > ~96 deg), globally and
over Europe + Africa, which are at night at 00 UTC.

USAGE:
  python3 figS1_plot_photochem_diagnostics_by_case.py <task_id> <task_ct>   # accumulate DJF/JJA composites (CPU array)
  python3 figS1_plot_photochem_diagnostics_by_case.py --plot                # single cases, figure, stats
"""

import logging
import sys
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from mirora.chemistry import (compute_leighton, compute_leighton_masked, compute_nox_ratio,
                              compute_nox_ratio_masked, load_aurora_fields, load_cams_fields, nonpositive)
from mirora.config import SEASON_OF_MONTH
from mirora.data.loading import fetch_rollout
from mirora.stats import agreement, lat_weights, wmean
from mirora.utils import get_cams_forecast_path, setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style, wrap_lon


import consts

logger = logging.getLogger(__name__)

OUT_DIR = consts.SCRATCH_PATH / "analysis" / "diagnostic_cases"
FIG_DIR = consts.FIGURES_PATH / "chemistry_products"
SEASONS = ("DJF", "JJA")
DIAGS = ("cams_nox", "aurora_nox", "cams_j", "aurora_j")
NIGHT_COS_ZENITH = -0.1          # sun > ~6 deg below the horizon (excludes twilight)
J_LIT_THRESHOLD = 1e-3           # s^-1, ~10% of typical daytime surface J1
EUROPE_AFRICA = dict(lat=(-35, 70), lon=(-20, 50))

SINGLE_CASES = [  # (row label, init time, rollout step)
    ("12:00 UTC\n(00 UTC init, +12 h)", datetime(2022, 9, 1, 0), 0),
    ("00:00 UTC\n(12 UTC init, +12 h)", datetime(2022, 9, 1, 12), 0),
    ("+120 h\n(00 UTC init)", datetime(2022, 9, 1, 0), 9),
]


def diagnostic_fields(init_time: datetime, step: int) -> dict:
    """NOx ratio and Leighton J1 for CAMS and Aurora (Aurora masked where nonpositive) on Aurora's grid."""
    preds = fetch_rollout(consts.base_rollout_path(init_time))
    w_row = lat_weights(preds)
    n_lat = len(w_row)  # Aurora drops CAMS's -90 row
    with xr.open_dataset(get_cams_forecast_path(consts.CAMS_DATA_PATH, init_time) / "data_plev.nc") as ds:
        c_no, c_no2, c_o3, c_t = (x[:n_lat] for x in load_cams_fields(ds, step + 1))
    a_no, a_no2, a_o3, a_t = load_aurora_fields(preds[step])
    return {
        "cams_nox": compute_nox_ratio(c_no, c_no2),
        "aurora_nox": compute_nox_ratio_masked(a_no, a_no2),
        "cams_j": compute_leighton(c_no, c_no2, c_o3, c_t),
        "aurora_j": compute_leighton_masked(a_no, a_no2, a_o3, a_t),
        "masked": nonpositive(a_no, a_no2, a_o3),
        "w_row": w_row, "lat": preds[step].metadata.lat.numpy(), "lon": preds[step].metadata.lon.numpy(),
        "valid": init_time + timedelta(hours=12 * (step + 1)),
    }


def accumulate(task_id: int, task_ct: int):
    """Per-season sums/counts of each init's +12 h diagnostic fields, for this task's inits."""
    acc = {}
    for init_time in consts.val_init_times()[task_id::task_ct]:
        season = SEASON_OF_MONTH[(init_time + timedelta(hours=12)).month]
        if season not in SEASONS:
            continue
        f = diagnostic_fields(init_time, step=0)
        for name in DIAGS:
            ok = np.isfinite(f[name])
            acc.setdefault(f"{season}_{name}_sum", np.zeros(ok.shape))[ok] += f[name][ok]
            acc.setdefault(f"{season}_{name}_cnt", np.zeros(ok.shape, dtype=np.int32))[ok] += 1
        acc.setdefault(f"{season}_masked_fracs", []).append(wmean(f["masked"].astype(float), f["w_row"]))
        acc[f"{season}_n"] = acc.get(f"{season}_n", 0) + 1
        logger.info(f"  {init_time} ({season}) done")
    np.savez(OUT_DIR / f"partial_{task_id:03d}.npz", **{k: np.asarray(v) for k, v in acc.items()})


def season_composites() -> dict:
    """Combine the task partials into one composite field dict per season."""
    parts = [np.load(p) for p in sorted(OUT_DIR.glob("partial_*.npz"))]
    out = {}
    for season in SEASONS:
        n = sum(int(p[f"{season}_n"]) for p in parts if f"{season}_n" in p)
        comp = {"n": n, "masked_frac": float(np.mean(np.concatenate(
            [p[f"{season}_masked_fracs"] for p in parts if f"{season}_masked_fracs" in p])))}
        for name in DIAGS:
            s = sum(p[f"{season}_{name}_sum"] for p in parts if f"{season}_{name}_sum" in p)
            c = sum(p[f"{season}_{name}_cnt"] for p in parts if f"{season}_{name}_cnt" in p)
            with np.errstate(invalid="ignore", divide="ignore"):
                field = s / c
            if name.startswith("aurora"):
                field[c < 0.5 * n] = np.nan  # masked on at least half of the inits
            comp[name] = field
        out[season] = comp
    return out


def cos_zenith(lat: np.ndarray, lon: np.ndarray, t: datetime) -> np.ndarray:
    """Cosine of the solar zenith angle on a [lat, lon] grid (no equation-of-time correction)."""
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + t.timetuple().tm_yday) / 365)
    hour_angle = np.deg2rad(15 * (t.hour + t.minute / 60 - 12) + lon)
    phi = np.deg2rad(lat)[:, None]
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)[None, :]


def night_stats(f: dict) -> dict:
    """J1 statistics over night-side cells, globally and over Europe + Africa."""
    night = cos_zenith(f["lat"], f["lon"], f["valid"]) < NIGHT_COS_ZENITH
    lon180 = wrap_lon(f["lon"])
    box = ((f["lat"] >= EUROPE_AFRICA["lat"][0]) & (f["lat"] <= EUROPE_AFRICA["lat"][1]))[:, None] \
        & ((lon180 >= EUROPE_AFRICA["lon"][0]) & (lon180 <= EUROPE_AFRICA["lon"][1]))[None, :]
    w = np.broadcast_to(f["w_row"][:, None], night.shape)
    row = {"night_area_frac": float((night * w).sum() / w.sum())}
    for region, sel in [("night", night), ("night_eur_afr", night & box)]:
        for src in ("cams", "aurora"):
            j = f[f"{src}_j"]
            ok = sel & np.isfinite(j)
            row[f"{region}_{src}_median_J"] = float(np.median(j[ok])) if ok.any() else np.nan
            row[f"{region}_{src}_frac_J_gt_1e-3"] = float((w[ok] * (j[ok] > J_LIT_THRESHOLD)).sum() / w[ok].sum()) \
                if ok.any() else np.nan
        row[f"{region}_aurora_masked_frac"] = float((w * (sel & f["masked"])).sum() / (w * sel).sum()) \
            if sel.any() else np.nan
    day = cos_zenith(f["lat"], f["lon"], f["valid"]) > -NIGHT_COS_ZENITH
    for src in ("cams", "aurora"):
        j = f[f"{src}_j"]
        ok = day & np.isfinite(j)
        row[f"day_{src}_median_J"] = float(np.median(j[ok]))
    return row


def case_stats(label: str, f: dict) -> dict:
    w_cells = np.broadcast_to(f["w_row"][:, None], f["cams_nox"].shape)
    row = {"case": label.replace("\n", " ")}
    for diag, a, c in [("noxratio", f["aurora_nox"], f["cams_nox"]), ("leighton", f["aurora_j"], f["cams_j"])]:
        ok = np.isfinite(a) & np.isfinite(c)
        row.update({f"{diag}_{k}": v for k, v in agreement(a, c, ok, w_cells).items()})
    return row


def plot(rows: list[tuple[str, dict, float]], save_path):
    """5 x 4 grid of maps with one colorbar per diagnostic."""
    pooled = np.concatenate([f[k][np.isfinite(f[k])] for _, f, _ in rows for k in ("cams_j", "aurora_j")])
    j_vmax = float(np.percentile(pooled, 98))
    nox_cmap, j_cmap = plt.get_cmap("plasma").copy(), plt.get_cmap("viridis").copy()
    nox_cmap.set_bad("white")
    j_cmap.set_bad("white")
    columns = [("cams_nox", "CAMS NO$_x$ ratio", nox_cmap, 0, 1), ("aurora_nox", "Aurora NO$_x$ ratio", nox_cmap, 0, 1),
               ("cams_j", "CAMS $J_1$", j_cmap, 0, j_vmax), ("aurora_j", "Aurora $J_1$", j_cmap, 0, j_vmax)]

    fig = plt.figure(figsize=(26, 3.4 * len(rows) + 2))
    gs = fig.add_gridspec(len(rows) + 1, 4, height_ratios=[1] * len(rows) + [0.08],
                          wspace=0.03, hspace=0.12, left=0.09, right=0.99, top=0.95, bottom=0.05)
    images = {}
    for r, (label, f, masked_frac) in enumerate(rows):
        for c, (key, title, cmap, vmin, vmax) in enumerate(columns):
            ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
            images[key] = draw_global_field(ax, f[key], cmap, vmin, vmax,
                                            borders=True, border_width=0.6, border_color="#888888")
            if r == 0:
                ax.set_title(title, fontsize=24, pad=10)
            if c == 0:
                ax.text(-0.04, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=22)
            if key.startswith("aurora"):
                ax.text(0.02, 0.04, f"masked {100 * masked_frac:.1f}%", transform=ax.transAxes, fontsize=20,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
    for cols, key, label, extend in [(slice(0, 2), "cams_nox", r"NO$_x$ ratio, NO$_2$/(NO + NO$_2$)", "neither"),
                                     (slice(2, 4), "cams_j", r"NO$_2$ photolysis rate $J_1$ (s$^{-1}$)", "max")]:
        cax = fig.add_subplot(gs[-1, cols])
        pos = cax.get_position()
        cax.set_position([pos.x0 + 0.02, pos.y0, pos.width - 0.04, pos.height])  # gap between the two colorbars
        cbar = fig.colorbar(images[key], cax=cax, orientation="horizontal", extend=extend)
        cbar.set_label(label, fontsize=22)
        cbar.ax.tick_params(labelsize=20)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path} (J1 colorbar 0-{j_vmax:.2e} s^-1)")


def main():
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] != ["--plot"]:
        assert len(sys.argv) == 3, "USAGE: python3 figS1_plot_photochem_diagnostics_by_case.py <task_id> <task_ct> | --plot"
        return accumulate(int(sys.argv[1]), int(sys.argv[2]))

    setup_matplotlib_style(20)
    singles = {label: diagnostic_fields(t, step) for label, t, step in SINGLE_CASES}
    comps = season_composites()
    lat_lon = next(iter(singles.values()))
    rows, stats = [], []
    for label in [SINGLE_CASES[0][0], SINGLE_CASES[1][0], "DJF", "JJA", SINGLE_CASES[2][0]]:
        if label in SEASONS:
            f = {**comps[label], "w_row": lat_lon["w_row"]}
            masked_frac, row_label = f["masked_frac"], f"{label} composite\n(+12 h, n = {f['n']})"
        else:
            f = singles[label]
            masked_frac, row_label = wmean(f["masked"].astype(float), f["w_row"]), label
        rows.append((row_label, f, masked_frac))
        st = {**case_stats(row_label, f), "aurora_masked_frac": masked_frac}
        if label not in SEASONS:
            st.update(night_stats(f))
        stats.append(st)

    df = pd.DataFrame(stats)
    df.to_csv(OUT_DIR / "diagnostic_cases_stats.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info(f"Per-case statistics:\n{df.set_index('case').T.to_string()}")
    plot(rows, FIG_DIR / "diagnostic_cases.pdf")


if __name__ == "__main__":
    main()
