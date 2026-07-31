"""
Validate the paper's +5/+20 ppb NOx perturbation claims for the Sep 1 2022
00:00Z init (Figure 3), at 1000 hPa surface level.

Claims under test:
  - +5 ppb: median surface O3 is 6.1 ppb lower than baseline at 12h
  - moderate short-range anticorrelation between NOx and O3 (r = -0.46)
  - perturbed +5 ppb O3 distribution extends into negative concentrations
  - +5 ppb: median shifts from 6.1 ppb below (12h) to 0.5 ppb above (72h)
  - +20 ppb: O3 suppressed across nearly all regions at 12h

Reports several candidate definitions so text numbers can be matched exactly.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

import consts

INIT_STR = datetime(2022, 9, 1, 0).strftime(r"%Y%m%d_%H%M")
LEVEL = 0  # 1000 hPa

M_AIR, M_O3, M_NO, M_NO2 = 28.97, 48.0, 30.006, 46.0055
O3_PPB = (M_AIR / M_O3) * 1e9
NO_PPB = (M_AIR / M_NO) * 1e9
NO2_PPB = (M_AIR / M_NO2) * 1e9


def load(exp):
    if exp == "base":
        p = consts.EXPERIMENTS_PATH / "base" / INIT_STR / "predictions.pickle"
    else:
        p = consts.EXPERIMENTS_PATH / exp / INIT_STR / "predictions.pickle"
    with open(p, "rb") as f:
        return pickle.load(f)


def surf(preds, var, step):
    return preds[step].atmos_vars[var][0, 0, LEVEL].numpy().astype(np.float64)


def o3_ppb(preds, step):
    return surf(preds, "go3", step) * O3_PPB


def nox_ppb(preds, step):
    return surf(preds, "no", step) * NO_PPB + surf(preds, "no2", step) * NO2_PPB


def report(base, pert, tag):
    print(f"\n===== {tag} =====")
    for step, lead in [(0, "12h"), (5, "72h")]:
        b = o3_ppb(base, step).ravel()
        p = o3_ppb(pert, step).ravel()
        print(f"  [{lead}] median(base)={np.median(b):+.3f} ppb  "
              f"median(pert)={np.median(p):+.3f} ppb  "
              f"median(pert)-median(base)={np.median(p)-np.median(b):+.3f} ppb  "
              f"median(pert-base)={np.median(p-b):+.3f} ppb")
        print(f"         mean(pert-base)={np.mean(p-b):+.3f} ppb  "
              f"frac cells O3 decreased={np.mean(p < b)*100:.1f}%  "
              f"frac perturbed O3 < 0={np.mean(p < 0)*100:.2f}%  "
              f"min perturbed O3={p.min():+.3f} ppb")

    # Correlation candidates (12h), NOx vs O3
    print("  --- correlation candidates at 12h (r) ---")
    b_o3, p_o3 = o3_ppb(base, 0).ravel(), o3_ppb(pert, 0).ravel()
    b_nox, p_nox = nox_ppb(base, 0).ravel(), nox_ppb(pert, 0).ravel()
    d_o3 = p_o3 - b_o3
    cands = {
        "corr(NOx_base, O3_base)": (b_nox, b_o3),
        "corr(NOx_pert, O3_pert)": (p_nox, p_o3),
        "corr(NOx_base, O3_pert)": (b_nox, p_o3),
        "corr(dO3, NOx_base)": (d_o3, b_nox),
        "corr(dO3, O3_base)": (d_o3, b_o3),
    }
    for name, (x, y) in cands.items():
        r = np.corrcoef(x, y)[0, 1]
        print(f"    {name:28s} r = {r:+.3f}")


def main():
    base = load("base")
    for exp in ["NOx_5ppb_increase", "NOx_20ppb_increase", "NOx_0.5ppb_increase"]:
        try:
            report(base, load(exp), exp)
        except FileNotFoundError as e:
            print(f"\n[skip] {exp}: {e}")


if __name__ == "__main__":
    main()
