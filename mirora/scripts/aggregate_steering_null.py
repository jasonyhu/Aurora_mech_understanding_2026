"""
Aggregate the steering null/control partials (steering_null_control.py) and
report, for each headline target feature, where its ablation effect falls in the
distribution of magnitude-matched random features (SI table).

For each target and each effect metric (sum|ΔO3| ppb, sum|ΔNO2| ppt) reports:
  - target value
  - null mean ± std, median, 95th/99th percentile
  - target's percentile rank in the null and its z-score
  - a one-tailed empirical p-value = fraction of null >= target

USAGE: python3 aggregate_steering_null.py
"""

import sys

import numpy as np
import pandas as pd

import consts

IN_DIR = consts.EXPERIMENTS_PATH / "sae_steering_null"
SUMMARY = IN_DIR / "steering_null_summary.csv"

METRICS = [("sum_abs_dO3_ppb", "sum|ΔO3| (ppb·cells)"),
           ("sum_abs_dNO2_ppt", "sum|ΔNO2| (ppt·cells)"),
           ("mean_abs_dO3_ppb", "mean|ΔO3| (ppb)"),
           ("mean_abs_dNO2_ppt", "mean|ΔNO2| (ppt)")]


def main():
    files = sorted(IN_DIR.glob("null_partial_*.csv"))
    if not files:
        sys.exit(f"No partials in {IN_DIR}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Loaded {len(files)} partials, {len(df)} rows.")

    records = []
    for target in df["target_name"].unique():
        d = df[df["target_name"] == target]
        tgt = d[d["role"] == "target"]
        null = d[d["role"] == "null"]
        print(f"\n{'='*72}\n{target}   (null n={len(null)})\n{'='*72}")
        if tgt.empty:
            print("  [!] no target row found — check that all array tasks completed.")
            continue
        for col, label in METRICS:
            tv = float(tgt[col].iloc[0])
            nv = null[col].to_numpy()
            nv = nv[np.isfinite(nv)]
            pct = float((nv < tv).mean() * 100)
            z = float((tv - nv.mean()) / nv.std()) if nv.std() > 0 else float("nan")
            pval = float((nv >= tv).mean())          # one-tailed empirical
            print(f"  {label:22s} target={tv:.4e}  null mean={nv.mean():.4e}±{nv.std():.3e}  "
                  f"median={np.median(nv):.4e}  p95={np.percentile(nv,95):.4e}  "
                  f"p99={np.percentile(nv,99):.4e}")
            print(f"  {'':22s} -> target percentile={pct:.2f}%  z={z:+.2f}  "
                  f"empirical p(null>=target)={pval:.4f}")
            records.append({"target": target, "metric": col, "target_value": tv,
                            "null_mean": nv.mean(), "null_std": nv.std(),
                            "null_median": float(np.median(nv)),
                            "null_p95": float(np.percentile(nv, 95)),
                            "null_p99": float(np.percentile(nv, 99)),
                            "target_percentile": pct, "z_score": z, "p_value": pval})
    pd.DataFrame(records).to_csv(SUMMARY, index=False)
    print(f"\nWrote {SUMMARY}")


if __name__ == "__main__":
    main()
