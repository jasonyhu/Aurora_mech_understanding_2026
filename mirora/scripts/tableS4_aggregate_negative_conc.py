"""
Build Table S4: percentage of surface (1000 hPa) grid cells with
negative concentrations, averaged over all inits.

Columns: CAMS (+12 h forecast), Aurora baseline +12 h and +120 h, and the
+0.5/+5/+20 ppb NOx perturbations at +12 h. Rows: CO, NO, NO2, O3, SO2, any.

- Perturbation + baseline +12 h: partial CSVs from tableS4_run_negative_conc_sweep.py,
  one set per NOx convention (analysis/nox_{mmr,ppbv}/negative_conc/).
- CAMS and baseline +120 h are convention-independent: computed here from the
  staged CAMS forecasts and the saved base rollouts, and cached in
  analysis/negative_conc_reference.csv.

Writes analysis/nox_{conv}/negative_conc/negative_conc_table.csv (raw and
cos-lat area-weighted percentages, plus the number of inits per column).
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from mirora.config import CHEM_ATMOS_VARS, SURFACE_LEVEL_IDX
from mirora.data.cams_dataloader import CAMSDataloader
from mirora.data.loading import fetch_rollout, surface
from mirora.stats import lat_weights, wmean
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

SPECIES = list(CHEM_ATMOS_VARS)

ANALYSIS = consts.SCRATCH_PATH / "analysis"
REFERENCE_CSV = ANALYSIS / "negative_conc_reference.csv"
STEP_120H = 9            # Aurora rollout step k is +12*(k+1) h
CAMS_FP_12H = 1          # CAMS forecast_period index 1 is +12 h
COLUMNS = ["CAMS", "base", "base_120h", "+0.5", "+5", "+20"]
ROW_NAMES = {"co": "CO", "no": "NO", "no2": "NO2", "go3": "O3", "so2": "SO2", "any": "Any species"}


def neg_fraction_rows(fields: dict[str, np.ndarray], w_row: np.ndarray, init_str: str, scenario: str) -> list[dict]:
    """Raw and area-weighted fraction of cells < 0 per species, plus 'any'."""
    masks = {sp: fields[sp] < 0 for sp in SPECIES}
    masks["any"] = np.logical_or.reduce([masks[sp] for sp in SPECIES])
    return [{"init_time": init_str, "scenario": scenario, "species": sp,
             "frac_neg_raw": float(m.mean()), "frac_neg_areaw": wmean(m.astype(np.float64), w_row)}
            for sp, m in masks.items()]


def surface_fields(pred) -> dict[str, np.ndarray]:
    return {sp: surface(pred, sp) for sp in SPECIES}


def build_reference() -> pd.DataFrame:
    """CAMS +12 h and Aurora base +120 h negative fractions for every init."""
    cams = CAMSDataloader(consts.CAMS_DATA_PATH)
    rows, cur = [], consts.val_start_time
    while cur <= consts.val_end_time:
        init_str = cur.strftime(r"%Y%m%d_%H%M")
        base_path = consts.base_rollout_path(cur)
        if not base_path.exists():
            logger.warning(f"missing base rollout {base_path}; skipping {init_str}")
        else:
            pred = fetch_rollout(base_path)[STEP_120H]
            w_row = lat_weights([pred])  # Aurora's grid = CAMS's minus the -90 row
            rows += neg_fraction_rows(surface_fields(pred), w_row, init_str, "base_120h")
            try:
                fields = {sp: cams.load_atmos_var(cur, sp, SURFACE_LEVEL_IDX, forecast_period=CAMS_FP_12H)[: len(w_row)] for sp in SPECIES}
                rows += neg_fraction_rows(fields, w_row, init_str, "CAMS")
            except OSError as e:  # missing or unreadable CAMS file
                logger.warning(f"missing CAMS for {init_str}: {e}")
        cur += timedelta(hours=12)
    return pd.DataFrame(rows)


def main():
    setup_logging()
    if REFERENCE_CSV.exists():
        reference = pd.read_csv(REFERENCE_CSV, dtype={"init_time": str})
    else:
        reference = build_reference()
        reference.to_csv(REFERENCE_CSV, index=False)
    logger.info(f"Reference: {reference['init_time'].nunique()} inits ({REFERENCE_CSV})")

    for conv in ("mmr", "ppbv"):
        in_dir = ANALYSIS / f"nox_{conv}" / "negative_conc"
        partials = sorted(in_dir.glob("negative_conc_partial_*.csv"))
        if not partials:
            logger.warning(f"[{conv}] no partial CSVs in {in_dir}, skipping")
            continue
        df = pd.concat([reference, *(pd.read_csv(p, dtype={"init_time": str}) for p in partials)])

        g = df.groupby(["species", "scenario"])
        pct = (100 * g[["frac_neg_raw", "frac_neg_areaw"]].mean()).unstack("scenario")
        n = g["init_time"].nunique().unstack("scenario")
        table = pd.concat({"raw_pct": pct["frac_neg_raw"], "areaw_pct": pct["frac_neg_areaw"], "n_inits": n}, axis=1)
        table = table.reindex(columns=COLUMNS, level=1).reindex(list(ROW_NAMES)).rename(index=ROW_NAMES)

        out_path = in_dir / "negative_conc_table.csv"
        table.to_csv(out_path)
        logger.info(f"[{conv}] area-weighted % of surface cells negative -> {out_path}\n"
                    f"{table['areaw_pct'].round(3).to_string()}")


if __name__ == "__main__":
    main()
