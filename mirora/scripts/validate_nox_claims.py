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


import numpy as np

from mirora.config import MMR_TO_PPB
from mirora.data.loading import fetch_rollout, surface

import consts


def load(ppb=None):
    """Sep 1 2022 00Z rollout: base if `ppb` is None, else the +`ppb` NOx experiment."""
    init = consts.MAIN_INIT_STR
    return fetch_rollout(consts.base_rollout_path(init) if ppb is None
                         else consts.rollout_path(consts.nox_experiment_dir(ppb), init))


def surf(preds, var, step):
    return surface(preds[step], var).astype(np.float64)


def o3_ppb(preds, step):
    return surf(preds, "go3", step) * MMR_TO_PPB["go3"]


def nox_ppb(preds, step):
    return surf(preds, "no", step) * MMR_TO_PPB["no"] + surf(preds, "no2", step) * MMR_TO_PPB["no2"]


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
    base = load()
    for ppb in [5, 20, 0.5]:
        exp = f"NOx_{ppb:g}ppb_increase"
        try:
            report(base, load(ppb), exp)
        except FileNotFoundError as e:
            print(f"\n[skip] {exp}: {e}")


if __name__ == "__main__":
    main()
