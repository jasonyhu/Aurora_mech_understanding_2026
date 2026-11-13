"""
Section 3.2 (median O3 shift across all 730 inits, e.g. +2.2 +/- 1.7 ppb at 72 h):
aggregate the per-task CSVs from sec3_2_run_nox_response_all_inits.py into
across-initialization distributions of the +5 ppb NOx surface-O3 statistics.

Reports, per lead time (with focus on 12h and 72h), the mean +/- std, median,
and IQR of each metric across all inits, plus a per-season split, to test
whether the Sep 1 2022 single-init numbers generalize.

USAGE: python3 sec3_2_aggregate_nox_response_all_inits.py
"""

import sys

import numpy as np
import pandas as pd

from mirora.config import SEASON_OF_MONTH

import consts

IN_DIR = consts.NOX_ANALYSIS_PATH / "nox_robustness"
SUMMARY_PATH = IN_DIR / "nox_robustness_summary.csv"

METRICS = [
    "diff_of_medians", "median_of_diff", "mean_diff", "mean_absdiff",
    "wmean_diff", "wmean_absdiff",
    "frac_suppressed_raw", "frac_suppressed_areaw",
    "frac_neg_raw", "frac_neg_areaw",
    "min_pert",
    "trop_median_of_diff", "trop_frac_increased_areaw",
]


def season(init_time: str) -> str:
    return SEASON_OF_MONTH[int(init_time[4:6])]


def load_partials() -> pd.DataFrame:
    """All per-task CSVs from sec3_2_run_nox_response_all_inits.py, with a 'season' column."""
    files = sorted(IN_DIR.glob("nox_robustness_partial_*.csv"))
    if not files:
        sys.exit(f"No partial CSVs found in {IN_DIR}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["init_time"] = df["init_time"].astype(str).str.zfill(13)
    df["season"] = df["init_time"].map(season)
    return df


def describe(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*70}\n{label}  (n_inits per step ~ {df.groupby('step').size().max()})\n{'='*70}")
    for step, lead in [(0, "12h"), (5, "72h")]:
        d = df[df["step"] == step]
        if d.empty:
            continue
        print(f"\n--- lead {lead} (step {step}) ---")
        for m in METRICS:
            vals = d[m].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            print(f"  {m:28s} mean={np.mean(vals):+9.4f}  std={np.std(vals):8.4f}  "
                  f"median={np.median(vals):+9.4f}  IQR=[{np.percentile(vals,25):+8.4f},"
                  f"{np.percentile(vals,75):+8.4f}]  min={vals.min():+8.4f}  max={vals.max():+8.4f}")


def main():
    df = load_partials()
    print(f"Loaded {len(df)} rows, {df['init_time'].nunique()} unique inits from {IN_DIR}.")

    describe(df, "ALL INITS")
    for s in ["JJA", "SON", "DJF", "MAM"]:
        sub = df[df["season"] == s]
        if not sub.empty:
            describe(sub, f"SEASON {s}")

    # Per-step summary table across all inits, saved for plotting.
    records = []
    for step in sorted(df["step"].unique()):
        d = df[df["step"] == step]
        rec = {"step": step, "lead_hours": int((step + 1) * 12), "n_inits": d["init_time"].nunique()}
        for m in METRICS:
            vals = d[m].to_numpy()
            vals = vals[np.isfinite(vals)]
            rec[f"{m}_mean"] = np.mean(vals)
            rec[f"{m}_std"] = np.std(vals)
            rec[f"{m}_median"] = np.median(vals)
            rec[f"{m}_q25"] = np.percentile(vals, 25)
            rec[f"{m}_q75"] = np.percentile(vals, 75)
        records.append(rec)
    pd.DataFrame(records).to_csv(SUMMARY_PATH, index=False)
    print(f"\nWrote per-step summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
