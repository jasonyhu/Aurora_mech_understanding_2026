"""
Photochemical diagnostics and baseline skill of the Aurora base rollouts vs
CAMS, over every init in the validation year: the RMSE vs CAMS analysis of
Table S1 and the masked fractions quoted in Section 3.1 / Figure S1.

Per init and lead step (+12 h .. +120 h), at the surface (1000 hPa):
  - masked fraction: cells with NO, NO2 or O3 <= 0 (Aurora, and the CAMS
    forecast at the same lead), raw and cos-lat area-weighted
  - NOx partitioning ratio and Leighton-derived J1, Aurora vs the CAMS
    forecast, on cells unmasked in both: Pearson r, Spearman rho, RMSE
  - per-species RMSE (ppb) against the CAMS analysis at the valid time
Rows carry the valid hour (00/12 UTC) and season, so the aggregate step can
group by season x hour x lead.

USAGE:
  python3 tableS1_compute_baseline_rmse_and_masking.py <task_id> <task_ct>   # per-task CSVs
  python3 tableS1_compute_baseline_rmse_and_masking.py --aggregate           # summary tables
"""

import logging
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

from mirora.chemistry import compute_leighton, compute_nox_ratio, load_aurora_fields, load_cams_fields, nonpositive
from mirora.config import MMR_TO_PPB, SEASON_OF_MONTH, SURFACE_LEVEL_IDX
from mirora.data.cams_dataloader import CAMSDataloader
from mirora.data.loading import fetch_rollout, surface
from mirora.stats import agreement, lat_weights, wmean
from mirora.utils import get_cams_forecast_path, setup_logging


import consts

logger = logging.getLogger(__name__)

OUT_DIR = consts.SCRATCH_PATH / "analysis" / "base_diagnostics"
N_STEPS = 10


def diagnose_init(init_time, cams: CAMSDataloader) -> list[dict]:
    preds = fetch_rollout(consts.base_rollout_path(init_time))
    w_row = lat_weights(preds)
    n_lat = len(w_row)  # Aurora drops CAMS's -90 row
    w_cells = np.broadcast_to(w_row[:, None], (n_lat, preds[0].atmos_vars["no"].shape[-1]))

    rows = []
    with xr.open_dataset(get_cams_forecast_path(consts.CAMS_DATA_PATH, init_time) / "data_plev.nc") as ds:
        for k in range(N_STEPS):
            valid = init_time + timedelta(hours=12 * (k + 1))
            no_a, no2_a, o3_a, t_a = load_aurora_fields(preds[k])
            no_c, no2_c, o3_c, t_c = (x[:n_lat] for x in load_cams_fields(ds, k + 1))
            m_a, m_c = nonpositive(no_a, no2_a, o3_a), nonpositive(no_c, no2_c, o3_c)
            row = {"init_time": init_time.strftime(r"%Y%m%d_%H%M"), "step": k, "lead_hours": 12 * (k + 1),
                   "valid_hour": valid.hour, "season": SEASON_OF_MONTH[valid.month],
                   "masked_aurora_raw": float(m_a.mean()), "masked_aurora_areaw": wmean(m_a.astype(float), w_row),
                   "masked_cams_raw": float(m_c.mean()), "masked_cams_areaw": wmean(m_c.astype(float), w_row)}

            ok = ~m_a & ~m_c
            for name, a, c in [
                ("noxratio", compute_nox_ratio(no_a, no2_a), compute_nox_ratio(no_c, no2_c)),
                ("leighton", compute_leighton(no_a, no2_a, o3_a, t_a), compute_leighton(no_c, no2_c, o3_c, t_c)),
            ]:
                row.update({f"{name}_{key}": v for key, v in agreement(a, c, ok, w_cells).items()})

            # Baseline skill vs the CAMS analysis valid at the same time
            if get_cams_forecast_path(consts.CAMS_DATA_PATH, valid).exists():
                for sp, to_ppb in MMR_TO_PPB.items():
                    a = surface(preds[k], sp, ppb=True)
                    c = cams.load_atmos_var(valid, sp, SURFACE_LEVEL_IDX, forecast_period=0)[:n_lat] * to_ppb
                    row[f"rmse_vs_analysis_{sp}_ppb"] = float(np.sqrt(np.average((a - c) ** 2, weights=w_cells)))
            rows.append(row)
    return rows


def aggregate():
    df = pd.concat(pd.read_csv(p, dtype={"init_time": str}) for p in sorted(OUT_DIR.glob("partial_*.csv")))
    logger.info(f"{df.init_time.nunique()} inits, {len(df)} rows")
    rmse_cols = [c for c in df.columns if c.startswith("rmse_vs_analysis_")]
    tables = {
        "masked_by_lead": df.groupby("lead_hours")[["masked_aurora_areaw", "masked_cams_areaw"]].mean(),
        "masked_by_season_hour_lead": df.groupby(["season", "valid_hour", "lead_hours"])[
            ["masked_aurora_areaw", "masked_cams_areaw"]].mean(),
        "fig2_stats_by_lead": df.groupby("lead_hours")[[c for c in df.columns if c.startswith(("noxratio_", "leighton_"))]]
                                .agg(["mean", "std"]),
        "rmse_vs_analysis_by_lead": df.groupby("lead_hours")[rmse_cols].agg(["mean", "std"]),
    }
    for name, t in tables.items():
        t.to_csv(OUT_DIR / f"{name}.csv")
        logger.info(f"{name}:\n{t.round(4).to_string()}")


def main():
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--aggregate"]:
        return aggregate()
    assert len(sys.argv) == 3, "USAGE: python3 tableS1_compute_baseline_rmse_and_masking.py <task_id> <task_ct> | --aggregate"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])

    cams = CAMSDataloader(consts.CAMS_DATA_PATH)
    rows = []
    for init_time in consts.val_init_times()[task_id::task_ct]:
        try:
            rows += diagnose_init(init_time, cams)
            logger.info(f"  {init_time} done")
        except FileNotFoundError as e:
            logger.warning(f"[skip] {init_time}: {e}")
    pd.DataFrame(rows).to_csv(OUT_DIR / f"partial_{task_id:03d}.csv", index=False)
    logger.info(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
