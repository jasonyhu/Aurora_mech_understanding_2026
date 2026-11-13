"""
Regenerate any missing base rollouts for the wildfire case-study inits
(consts.FIRE_EVENTS) so plot_fire_stills.py can render the SI figures.
Predictions only (no backbone activations). ~a few min each.
"""

import logging
from datetime import datetime

from mirora.data.loading import load_batch
from mirora.inference.inference import aap_rollout, load_aurora
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

ROLLOUT_STEPS = 10


def main():
    setup_logging()
    todo = []
    for event in consts.FIRE_EVENTS:
        out = consts.base_rollout_path(event["init"])
        if out.exists():
            logger.info(f"{out} exists, skipping.")
        else:
            todo.append(datetime.strptime(event["init"], r"%Y%m%d_%H%M"))
    if todo:
        consts.ensure_checkpoint()
        model, _ = load_aurora(consts.CHECKPOINT_PATH)
    for init_time in todo:
        logger.info(f"=== Base rollout {init_time} ===")
        aap_rollout(
            batch=load_batch(consts.BATCH_INITS_PATH, init_time),
            batch_date=init_time,
            aap_checkpoint_path=consts.CHECKPOINT_PATH,
            save_dir=consts.EXPERIMENTS_PATH / "base",
            rollout_steps=ROLLOUT_STEPS,
            save_backbone_activations=False,
            model=model,
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
