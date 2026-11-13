"""
Rollouts behind Figure 7 (SAE feature steering): ablate each case-study feature (consts.CASE_STUDY_FEATURES)
in a 10-step rollout from the main init.

  zero: the feature's activation is set to 0 on every token (default; plotted)
  mean: it is set to its mean activation over all tokens of the init batch

Each (feature, ablation) rollout is saved to
experiments/sae_steering/{name}_{ablation}/predictions.pickle (+ metadata.txt);
existing rollouts are skipped.

USAGE: python3 fig7_run_sae_ablation_rollouts.py [--ablations zero mean]
"""

import argparse
import logging
import pickle

import torch

from mirora.data.loading import load_batch
from mirora.inference.inference import load_aurora
from mirora.interp import capture_blocks, feature_replacement_hook, get_backbone_block, load_sae
from mirora.utils import setup_logging

import consts
from tableS5_S6_rank_and_ablate_top_sae_features import run_rollout

logger = logging.getLogger(__name__)

INIT_TIME = consts.MAIN_INIT_TIME
ROLLOUT_STEPS = 10

EXP_DIR = consts.EXPERIMENTS_PATH / "sae_steering"


def compute_mean_activation(model, device, sae, key, feat_idx) -> float:
    """Mean of feature feat_idx's activation over all tokens (inactive tokens included)."""
    with capture_blocks(model, [key]) as captured, torch.no_grad():
        model(load_batch(consts.BATCH_INITS_PATH, INIT_TIME).to(device))
    x = captured[key]
    with torch.no_grad():
        h = sae.encode(x.reshape(-1, x.shape[-1]))
    return float(h[:, feat_idx].mean().item())


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", choices=["zero", "mean"], default=["zero"])
    args = parser.parse_args()

    logger.info("Loading Aurora model...")
    model, device = load_aurora(consts.CHECKPOINT_PATH)

    for target in consts.CASE_STUDY_FEATURES:
        logger.info(f"\n=== {target['name']} ===")
        pending = [a for a in args.ablations
                   if not (EXP_DIR / f"{target['name']}_{a}" / "predictions.pickle").exists()]
        if not pending:
            logger.info("  All ablations already have predictions.pickle — skipping target.")
            continue

        key = (target["component"], target["stage"], target["block"])
        sae = load_sae(consts.SAE_DIR, *key, target["dim"], device)
        mean_act = compute_mean_activation(model, device, sae, key, target["feat_idx"])
        logger.info(f"  Mean activation for feature #{target['feat_idx']}: {mean_act:.6f}")

        for ablation in pending:
            replacement = 0.0 if ablation == "zero" else mean_act
            out_dir = EXP_DIR / f"{target['name']}_{ablation}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Running {ablation}-ablation rollout (replacement={replacement:.6f})...")

            predictions = run_rollout(model, device, INIT_TIME, ROLLOUT_STEPS, block=get_backbone_block(model, *key),
                                      hook=feature_replacement_hook(sae, target["feat_idx"], replacement))

            pred_path = out_dir / "predictions.pickle"
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)
            (out_dir / "metadata.txt").write_text(
                f"target={target['name']}\nablation={ablation}\n"
                f"replacement_value={replacement}\nmean_activation={mean_act}\n"
                f"feat_idx={target['feat_idx']}\ninit_time={INIT_TIME.isoformat()}\n"
            )
            logger.info(f"    Saved {pred_path}")

    logger.info(f"\nDone. Rollouts under {EXP_DIR}")


if __name__ == "__main__":
    main()
