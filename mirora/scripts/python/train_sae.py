"""Train one L1 sparse autoencoder per backbone layer (SLURM array;
task_id selects the layer, 0-23 encoder / 24-47 decoder)."""

import sys
from argparse import ArgumentParser

from mirora.interp import train_sae

import consts

parser = ArgumentParser()
parser.add_argument("task_id", type=int)
parser.add_argument("task_ct", type=int)
parser.add_argument("--expansion_factor", type=int, default=8)
parser.add_argument("--l1_coeff", type=float, default=3e-4)
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
