"""
Regenerate the base rollout and NOx perturbation rollouts (+0.5/+5/+20 ppb)
for a single initialization, since predictions.pickle for this init was
purged from $SCRATCH (90-day inactivity purge).
"""

import pickle
import logging
import sys
from datetime import datetime

from mirora.inference.inference import aap_rollout
from mirora.experiments.perturbation import perturb_batch
from mirora.utils import get_init_batch_path

import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

INIT_TIME = datetime(2022, 9, 1, 0)
ROLLOUT_STEPS = 10

PERTURBATIONS_PPB = [0.5, 5, 20]


def load_batch():
    batch_path = get_init_batch_path(consts.BATCH_INITS_PATH, INIT_TIME)
    with open(batch_path, "rb") as f:
        return pickle.load(f)


def main():
    base_dir = consts.EXPERIMENTS_PATH / "base"
    base_rollout_file = base_dir / INIT_TIME.strftime(r"%Y%m%d_%H%M") / "predictions.pickle"

    if not base_rollout_file.exists():
        logger.info(f"=== Base rollout ({INIT_TIME}) ===")
        batch = load_batch()
        aap_rollout(
            batch=batch,
            batch_date=INIT_TIME,
            aap_checkpoint_path=consts.CHECKPOINT_PATH,
            save_dir=base_dir,
            rollout_steps=ROLLOUT_STEPS,
            save_backbone_activations=True,
        )
    else:
        logger.info(f"Base rollout already exists at {base_rollout_file}, skipping.")

    for ppb in PERTURBATIONS_PPB:
        exp_dir = consts.EXPERIMENTS_PATH / f"NOx_{ppb}ppb_increase" if ppb != int(ppb) \
            else consts.EXPERIMENTS_PATH / f"NOx_{int(ppb)}ppb_increase"
        exp_rollout_file = exp_dir / INIT_TIME.strftime(r"%Y%m%d_%H%M") / "predictions.pickle"

        if exp_rollout_file.exists():
            logger.info(f"+{ppb}ppb rollout already exists at {exp_rollout_file}, skipping.")
            continue

        logger.info(f"=== +{ppb} ppb NOx perturbation ({INIT_TIME}) ===")
        batch = load_batch()
        perturb_batch(batch, atmos_vars={"no": ppb * 1e-9, "no2": ppb * 1e-9})
        aap_rollout(
            batch=batch,
            batch_date=INIT_TIME,
            aap_checkpoint_path=consts.CHECKPOINT_PATH,
            save_dir=exp_dir,
            rollout_steps=ROLLOUT_STEPS,
            save_backbone_activations=False,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
