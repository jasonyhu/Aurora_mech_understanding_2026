"""
Rollouts behind Figures 3, S2 and S3 (Section 2.2.2): run NOx perturbation rollouts
(+0.5/+5/+20 ppb by default) and save the full predictions under consts.NOX_EXPERIMENTS_PATH (convention set by
MIRORA_NOX_CONVENTION; see consts.nox_offsets).

USAGE:
  # Sep 1 2022 00Z (the paper's running example): base rollout + every magnitude
  python3 fig3_run_global_nox_perturbations.py

  # Every validation-year init, sharded over <task_ct> parallel tasks
  python3 fig3_run_global_nox_perturbations.py --year [--ppb 5] <task_id> <task_ct>

Existing rollouts are skipped, so an interrupted job can simply be resubmitted.
"""

import argparse
import logging

from mirora.data.loading import load_batch
from mirora.experiments.perturbation import perturb_batch
from mirora.inference.inference import aap_rollout, load_aurora
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

ROLLOUT_STEPS = 10


def run_rollout(model, init_time, save_dir, atmos_offsets=None, save_backbone_activations=False):
    """Roll out the (optionally perturbed) init batch unless predictions already exist."""
    out_file = consts.rollout_path(save_dir, init_time)
    if out_file.exists():
        logger.info(f"Rollout already exists at {out_file}, skipping.")
        return
    batch = load_batch(consts.BATCH_INITS_PATH, init_time)
    if atmos_offsets:
        perturb_batch(batch, atmos_vars=atmos_offsets)
    aap_rollout(
        batch=batch,
        batch_date=init_time,
        aap_checkpoint_path=consts.CHECKPOINT_PATH,
        save_dir=save_dir,
        rollout_steps=ROLLOUT_STEPS,
        save_backbone_activations=save_backbone_activations,
        model=model,
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", action="store_true", help="every validation init instead of Sep 1 2022 00Z")
    parser.add_argument("--ppb", type=float, nargs="+", default=consts.NOX_MAGNITUDES_PPB)
    parser.add_argument("task", type=int, nargs="*", help="<task_id> <task_ct> (with --year)")
    args = parser.parse_args()

    logger.info(f"MIRORA_NOX_CONVENTION={consts.NOX_CONVENTION}")
    consts.ensure_checkpoint()
    model, _ = load_aurora(consts.CHECKPOINT_PATH)

    if args.year:
        task_id, task_ct = args.task if args.task else (0, 1)
        init_times = consts.val_init_times()[task_id::task_ct]
    else:
        init_times = [consts.MAIN_INIT_TIME]
        logger.info(f"=== Base rollout ({consts.MAIN_INIT_TIME}) ===")
        run_rollout(model, consts.MAIN_INIT_TIME, consts.EXPERIMENTS_PATH / "base", save_backbone_activations=True)

    for init_time in init_times:
        for ppb in args.ppb:
            logger.info(f"=== +{ppb:g} {consts.NOX_UNIT_LABEL} NOx perturbation ({init_time}) ===")
            run_rollout(model, init_time, consts.nox_experiment_dir(ppb), atmos_offsets=consts.nox_offsets(ppb))
    logger.info("Done.")


if __name__ == "__main__":
    main()
