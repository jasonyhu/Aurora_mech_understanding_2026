"""Train the 48 AuroraScope L1 sparse autoencoders, one per operator block (one task per
block; task_id selects the block in config.STAGES, 0-23 encoder / 24-47 decoder).
Checkpoints are written to MODEL_PATH/sae/."""

from argparse import ArgumentParser

from mirora.config import DEFAULT_EXPANSION_FACTOR, DEFAULT_L1_COEFF
from mirora.interp import train_sae
from mirora.utils import setup_logging

import consts


def main():
    setup_logging()
    parser = ArgumentParser()
    parser.add_argument("task_id", type=int)
    parser.add_argument("task_ct", type=int, nargs="?")  # accepted for array launchers, unused
    parser.add_argument("--expansion_factor", type=int, default=DEFAULT_EXPANSION_FACTOR)
    parser.add_argument("--l1_coeff", type=float, default=DEFAULT_L1_COEFF)
    args = parser.parse_args()

    train_sae.train(
        consts.CHECKPOINT_PATH,
        consts.BATCH_INITS_PATH,
        consts.MODEL_PATH,
        consts.val_start_time,
        consts.val_end_time,
        stage_idx=args.task_id,
        expansion_factor=args.expansion_factor,
        l1_coeff=args.l1_coeff,
    )


if __name__ == "__main__":
    main()
