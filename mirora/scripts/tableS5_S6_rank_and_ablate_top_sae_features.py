"""
Section 3.5 feature selection, and the per-init input to Tables S5 and S6
(see tableS5_S6_aggregate_sae_feature_recurrence.py). Rank the top-N SAE features globally by per-token variance contribution
across all 48 backbone blocks for one init (default Sep 1 2022 00 UTC), then
zero-ablate each one plus the two case-study features (consts.CASE_STUDY_FEATURES)
and measure the resulting +12 h change in surface O3 and surface NO2.
"""

import argparse
import csv
import json
import logging
from datetime import datetime

import numpy as np
import torch

from aurora import rollout
from mirora.config import MMR_TO_PPB, STAGES
from mirora.data.loading import fetch_rollout, load_batch, surface
from mirora.inference.inference import load_aurora
from mirora.interp import capture_blocks, feature_replacement_hook, get_backbone_block, load_sae, var_contribution
from mirora.stats import lat_weights, wmean
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

N_TOP         = 10
ROLLOUT_STEPS = 1

OUT_ROOT = consts.EXPERIMENTS_PATH / "sae_top_activating_features"
# Steering-figure colour-scale saturation limits (ppb), for the fraction of cells beyond them
O3_SAT_PPB, NO2_SAT_PPB = 4.0, 1.0


def run_rollout(model, device, init_time, steps=ROLLOUT_STEPS, block=None, hook=None):
    """Rollout from a freshly loaded init batch, optionally with a forward hook on `block`."""
    handle = block.register_forward_hook(hook) if hook is not None else None
    try:
        with torch.inference_mode():
            return [p.to("cpu") for p in rollout(model, load_batch(consts.BATCH_INITS_PATH, init_time).to(device),
                                                 steps=steps)]
    finally:
        if handle is not None:
            handle.remove()


def base_predictions(model, device, init_time, steps=ROLLOUT_STEPS):
    """Cached base rollout for `init_time` if on disk, else a fresh `steps`-step rollout."""
    path = consts.base_rollout_path(init_time)
    if path.exists():
        logger.info(f"Loaded cached base prediction: {path}")
        return fetch_rollout(path)
    logger.info(f"No cached base; running fresh base rollout ({steps} step)...")
    return run_rollout(model, device, init_time, steps)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=consts.MAIN_INIT_STR, help="YYYYMMDD_HHMM")
    args = parser.parse_args()
    init_time = datetime.strptime(args.init, r"%Y%m%d_%H%M")
    out_dir = OUT_ROOT / args.init
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Aurora model...")
    model, device = load_aurora(consts.CHECKPOINT_PATH)

    # -- Phase 1: capture activations at every block (single forward, inference mode) --
    logger.info(f"Forward pass ({init_time}) to collect block activations...")
    with capture_blocks(model, [(c, s, b) for c, s, b, _ in STAGES]) as captured, torch.inference_mode():
        model(load_batch(consts.BATCH_INITS_PATH, init_time).to(device))

    # -- Phase 1b: per-feature max-per-token variance contribution --
    logger.info("Computing variance contributions per feature for all 48 SAEs...")
    all_features = []
    for comp, stage, blk, dim in STAGES:
        sae = load_sae(consts.SAE_DIR, comp, stage, blk, dim, device)
        with torch.no_grad():
            h_act, contrib = var_contribution(sae, captured.pop((comp, stage, blk)).reshape(-1, dim))
            var_max = contrib.max(dim=0).values.cpu().numpy()
            mean_act = h_act.mean(dim=0).cpu().numpy()
            max_act  = h_act.max(dim=0).values.cpu().numpy()
            frac_active = (h_act > 0).float().mean(dim=0).cpu().numpy()

        for k in range(var_max.shape[0]):
            all_features.append({
                "component": comp, "stage": stage, "block": blk, "dim": dim,
                "feat_idx": int(k),
                "var_contrib_max": float(var_max[k]),
                "mean_act":        float(mean_act[k]),
                "max_act":         float(max_act[k]),
                "frac_tokens_active": float(frac_active[k]),
            })
        del sae, h_act, contrib
        torch.cuda.empty_cache()

    # -- Phase 2: rank globally, pick top N, plus the case-study features --
    all_features.sort(key=lambda r: r["var_contrib_max"], reverse=True)
    top = all_features[:N_TOP]
    for i, r in enumerate(top):
        r["rank"] = i + 1
    by_key = {(r["component"], r["stage"], r["block"], r["feat_idx"]): (i, r) for i, r in enumerate(all_features)}
    to_ablate = list(top)
    for t in consts.CASE_STUDY_FEATURES:
        i, r = by_key[(t["component"], t["stage"], t["block"], t["feat_idx"])]
        if r not in to_ablate:
            to_ablate.append({**r, "rank": i + 1})
    logger.info(f"\nTop {N_TOP} features by max per-token variance contribution:")
    for r in top:
        logger.info(
            f"  {r['component']} s{r['stage']} b{r['block']} feat #{r['feat_idx']}  "
            f"var_max={r['var_contrib_max']:.4e}  "
            f"mean_act={r['mean_act']:.4e}  max_act={r['max_act']:.4e}"
        )

    # -- Phase 3: base rollout (cached if available) --
    base_preds = base_predictions(model, device, init_time)
    base_o3  = surface(base_preds[0], "go3")
    base_no2 = surface(base_preds[0], "no2")

    # -- Phase 4: zero-ablate each candidate, measure ΔO3 and ΔNO2 in ppb/ppt --
    results = []
    w_row = lat_weights(base_preds)
    for r in to_ablate:
        name = f"{r['component']}_s{r['stage']}_b{r['block']}_feat{r['feat_idx']}"
        logger.info(f"\nAblating {name}...")
        sae = load_sae(consts.SAE_DIR, r["component"], r["stage"], r["block"], r["dim"], device)
        block = get_backbone_block(model, r["component"], r["stage"], r["block"])
        preds = run_rollout(model, device, init_time, block=block, hook=feature_replacement_hook(sae, r["feat_idx"]))

        do3_ppb  = (surface(preds[0], "go3") - base_o3)  * MMR_TO_PPB["go3"]
        dno2_ppt = (surface(preds[0], "no2") - base_no2) * MMR_TO_PPB["no2"] * 1e3

        out = {
            **r, "name": name,
            "sum_abs_dO3_ppb":  float(np.nansum(np.abs(do3_ppb))),
            "max_abs_dO3_ppb":  float(np.nanmax(np.abs(do3_ppb))),
            "sum_abs_dNO2_ppt": float(np.nansum(np.abs(dno2_ppt))),
            "max_abs_dNO2_ppt": float(np.nanmax(np.abs(dno2_ppt))),
            "mean_abs_dO3_ppb":        float(np.nanmean(np.abs(do3_ppb))),
            "mean_abs_dO3_ppb_areaw":  wmean(np.abs(do3_ppb), w_row),
            "mean_abs_dNO2_ppb":       float(np.nanmean(np.abs(dno2_ppt))) / 1e3,
            "mean_abs_dNO2_ppb_areaw": wmean(np.abs(dno2_ppt), w_row) / 1e3,
            "frac_cells_dO3_beyond_sat":  float(np.mean(np.abs(do3_ppb) > O3_SAT_PPB)),
            "frac_cells_dNO2_beyond_sat": float(np.mean(np.abs(dno2_ppt) / 1e3 > NO2_SAT_PPB)),
        }
        results.append(out)
        logger.info(
            f"  sum|ΔO3|={out['sum_abs_dO3_ppb']:.3e} ppb·cells  "
            f"max|ΔO3|={out['max_abs_dO3_ppb']:.3e} ppb  "
            f"sum|ΔNO2|={out['sum_abs_dNO2_ppt']:.3e} ppt·cells  "
            f"max|ΔNO2|={out['max_abs_dNO2_ppt']:.3e} ppt"
        )

    # -- Save & report --
    with open(out_dir / "top_features_results.json", "w") as f:
        json.dump(results, f, indent=2)
    cols = ["name", "rank", "component", "stage", "block", "feat_idx", "dim",
            "var_contrib_max", "mean_act", "max_act", "frac_tokens_active",
            "mean_abs_dO3_ppb", "mean_abs_dO3_ppb_areaw", "mean_abs_dNO2_ppb", "mean_abs_dNO2_ppb_areaw",
            "frac_cells_dO3_beyond_sat", "frac_cells_dNO2_beyond_sat",
            "sum_abs_dO3_ppb", "max_abs_dO3_ppb",
            "sum_abs_dNO2_ppt", "max_abs_dNO2_ppt"]
    with open(out_dir / "top_features_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])
    logger.info(f"\nSaved → {out_dir}/top_features_results.{{json,csv}}")

    logger.info("\nTop by sum |ΔO3|:")
    for r in sorted(results, key=lambda r: r["sum_abs_dO3_ppb"], reverse=True)[:5]:
        logger.info(f"  {r['name']:>40}  sum|ΔO3|={r['sum_abs_dO3_ppb']:.3e} ppb·cells")
    logger.info("\nTop by sum |ΔNO2|:")
    for r in sorted(results, key=lambda r: r["sum_abs_dNO2_ppt"], reverse=True)[:5]:
        logger.info(f"  {r['name']:>40}  sum|ΔNO2|={r['sum_abs_dNO2_ppt']:.3e} ppt·cells")


if __name__ == "__main__":
    main()
