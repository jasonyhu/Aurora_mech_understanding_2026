"""
Regenerate base rollouts for the four additional wildfire case-study inits
(Siberia, Alaska, Amazon, Argentina) so plot_fire_stills.py can render the
SI figures. Predictions only (no backbone activations). ~a few min each.
"""

import pickle
import logging
import sys
from datetime import datetime

from mirora.inference.inference import aap_rollout
from mirora.utils import get_init_batch_path
import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROLLOUT_STEPS = 10
INITS = [
    datetime(2022, 7, 10, 12),  # Siberia (Yakutia)
    datetime(2022, 7, 3, 0),    # Alaska/Yukon
    datetime(2022, 9, 6, 0),    # Amazon Basin
    datetime(2022, 6, 15, 12),  # Chile/Argentina
]


def main():
    base_dir = consts.EXPERIMENTS_PATH / "base"
    for init_time in INITS:
        out = base_dir / init_time.strftime(r"%Y%m%d_%H%M") / "predictions.pickle"
        if out.exists():
            logger.info(f"{out} exists, skipping.")
            continue
        logger.info(f"=== Base rollout {init_time} ===")
        with open(get_init_batch_path(consts.BATCH_INITS_PATH, init_time), "rb") as f:
            batch = pickle.load(f)
        aap_rollout(
            batch=batch,
            batch_date=init_time,
            aap_checkpoint_path=consts.CHECKPOINT_PATH,
            save_dir=base_dir,
            rollout_steps=ROLLOUT_STEPS,
            save_backbone_activations=False,
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
