"""
Null/control for the SAE steering result (SI table).

Features are selected by variance contribution, and zero-ablating them changes
surface O3/NO2. High-activation directions move the output *by construction*,
so the fair baseline is to ablate randomly chosen features MATCHED FOR
ACTIVATION MAGNITUDE and check that the selected feature's effect is an outlier.

Because W_dec rows are unit-normalized, zero-ablating feature k removes a vector
of L2 size exactly equal to k's activation magnitude — so "matched for
activation magnitude" holds the injected perturbation size constant (the fair
control). Matching is rank-based: we take a window of features whose per-token
MEAN activation is closest to the target's, then randomly sample N from it.

For each case-study feature (consts.CASE_STUDY_FEATURES) we zero-ablate (a) the
target feature and (b) N matched random features, each with the SAME single-step
(12h) rollout, and record sum/mean/max |ΔO3| (ppb) and |ΔNO2| (ppt) at the
surface. Aggregate with aggregate_steering_null.py to get the target's
percentile / z-score in the null.

Array job: the 2*(1+N) ablation units are split across tasks by global index.
Sampling is seeded so all tasks agree on the same null features.

USAGE: python3 steering_null_control.py <task_id> <task_ct>
"""

import argparse
import csv
import logging

import numpy as np
import torch

from mirora.config import MMR_TO_PPB
from mirora.data.loading import load_batch, surface
from mirora.inference.inference import load_aurora
from mirora.interp import capture_blocks, feature_replacement_hook, get_backbone_block, load_sae
from mirora.utils import setup_logging

import consts
from find_top_activating_features import base_predictions, run_rollout

logger = logging.getLogger(__name__)

INIT_TIME = consts.MAIN_INIT_TIME
N_NULL    = 100                         # random matched features per target
WINDOW    = 300                         # magnitude-rank window to sample from
SEED      = 0

OUT_DIR = consts.EXPERIMENTS_PATH / "sae_steering_null"

FIELDS = ["target_name", "role", "feat_idx", "mean_act",
          "sum_abs_dO3_ppb", "mean_abs_dO3_ppb", "max_abs_dO3_ppb",
          "sum_abs_dNO2_ppt", "mean_abs_dNO2_ppt", "max_abs_dNO2_ppt"]


def target_name(t):
    """Output key of a case-study feature, e.g. 'ozone_decoder_s1_b9_feat2712'."""
    return f"{t['species']}_{t['name']}"


def build_null_indices(mean_act, target_idx, n_null, window, seed):
    """Random features whose mean activation is closest (rank-wise) to target."""
    active = np.where(mean_act > 1e-12)[0]
    order = active[np.argsort(mean_act[active])]            # ascending by mean_act
    pos = np.where(order == target_idx)[0]
    if len(pos) == 0:                                      # target inactive: fall back
        cand = [int(i) for i in order if int(i) != target_idx]
    else:
        p = int(pos[0])
        lo = max(0, p - window // 2)
        hi = min(len(order), lo + window)
        lo = max(0, hi - window)
        cand = [int(i) for i in order[lo:hi] if int(i) != target_idx]
    rng = np.random.default_rng(seed)
    k = min(n_null, len(cand))
    return sorted(int(x) for x in rng.choice(np.array(cand), size=k, replace=False))


def ablate_and_measure(model, device, block, sae, feat_idx, base_o3, base_no2):
    preds = run_rollout(model, device, INIT_TIME, block=block, hook=feature_replacement_hook(sae, feat_idx))
    do3  = (surface(preds[0], "go3") - base_o3)  * MMR_TO_PPB["go3"]
    dno2 = (surface(preds[0], "no2") - base_no2) * MMR_TO_PPB["no2"] * 1e3  # ppt
    n_o3  = np.isfinite(do3).sum()
    n_no2 = np.isfinite(dno2).sum()
    return {
        "sum_abs_dO3_ppb":  float(np.nansum(np.abs(do3))),
        "mean_abs_dO3_ppb": float(np.nansum(np.abs(do3)) / max(n_o3, 1)),
        "max_abs_dO3_ppb":  float(np.nanmax(np.abs(do3))),
        "sum_abs_dNO2_ppt":  float(np.nansum(np.abs(dno2))),
        "mean_abs_dNO2_ppt": float(np.nansum(np.abs(dno2)) / max(n_no2, 1)),
        "max_abs_dNO2_ppt":  float(np.nanmax(np.abs(dno2))),
    }


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    parser.add_argument("task_ct", type=int)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    consts.ensure_checkpoint()

    logger.info("Loading Aurora model...")
    model, device = load_aurora(consts.CHECKPOINT_PATH)

    base_preds = base_predictions(model, device, INIT_TIME)
    base_o3  = surface(base_preds[0], "go3")
    base_no2 = surface(base_preds[0], "no2")

    # One forward pass capturing both target blocks' activations.
    targets = consts.CASE_STUDY_FEATURES
    with capture_blocks(model, [(t["component"], t["stage"], t["block"]) for t in targets]) as captured, \
            torch.inference_mode():
        model(load_batch(consts.BATCH_INITS_PATH, INIT_TIME).to(device))

    # Load SAEs, compute matched null pools, and assemble the global unit list.
    saes, units = {}, []
    for t in targets:
        name = target_name(t)
        sae = load_sae(consts.SAE_DIR, t["component"], t["stage"], t["block"], t["dim"], device)
        saes[name] = sae
        x = captured[(t["component"], t["stage"], t["block"])]
        with torch.no_grad():
            mean_act = sae.encode(x.reshape(-1, x.shape[-1])).mean(dim=0).cpu().numpy()
        null_idx = build_null_indices(mean_act, t["feat_idx"], N_NULL, WINDOW, SEED)
        logger.info(f"{name}: target mean_act={mean_act[t['feat_idx']]:.4e}; "
                    f"{len(null_idx)} matched null features "
                    f"(mean_act range {mean_act[null_idx].min():.3e}..{mean_act[null_idx].max():.3e})")

        units.append({**t, "name": name, "role": "target", "feat": t["feat_idx"],
                      "mean_act": float(mean_act[t["feat_idx"]])})
        for k in null_idx:
            units.append({**t, "name": name, "role": "null", "feat": int(k), "mean_act": float(mean_act[k])})

    mine = units[args.task_id::args.task_ct]
    logger.info(f"Task {args.task_id}/{args.task_ct}: {len(mine)} ablations (of {len(units)} total)")

    rows = []
    for u in mine:
        blk = get_backbone_block(model, u["component"], u["stage"], u["block"])
        m = ablate_and_measure(model, device, blk, saes[u["name"]], u["feat"], base_o3, base_no2)
        rows.append({"target_name": u["name"], "role": u["role"], "feat_idx": u["feat"],
                     "mean_act": u["mean_act"], **m})
        logger.info(f"  [{u['role']:6}] {u['name']} #{u['feat']}: "
                    f"sum|dO3|={m['sum_abs_dO3_ppb']:.3e}  sum|dNO2|={m['sum_abs_dNO2_ppt']:.3e}")

    out_path = OUT_DIR / f"null_partial_{args.task_id:03d}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
