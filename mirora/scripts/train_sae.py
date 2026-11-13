"""Train one sparse autoencoder per backbone block (one task per block; task_id selects the
block in config.STAGES, 0-23 encoder / 24-47 decoder). --method l1 (default) writes
to MODEL_PATH/sae/, --method topk (BatchTopK) to MODEL_PATH/sae_topk/."""

from argparse import ArgumentParser

from mirora.config import DEFAULT_EXPANSION_FACTOR, DEFAULT_L1_COEFF, DEFAULT_TOPK_ALPHA_AUX
from mirora.interp import train_sae
from mirora.utils import setup_logging

import consts


def main():
    setup_logging()
    parser = ArgumentParser()
    parser.add_argument("task_id", type=int)
    parser.add_argument("task_ct", type=int, nargs="?")  # accepted for array launchers, unused
    parser.add_argument("--method", choices=["l1", "topk"], default="l1")
    parser.add_argument("--expansion_factor", type=int, default=DEFAULT_EXPANSION_FACTOR)
    parser.add_argument("--l1_coeff", type=float, default=DEFAULT_L1_COEFF)
    parser.add_argument("--k", type=int, default=None, help="topk: active features per token (default d_hidden/32)")
    parser.add_argument("--k_aux", type=int, default=None, help="topk: dead features in the auxiliary loss (default k)")
    parser.add_argument("--alpha_aux", type=float, default=DEFAULT_TOPK_ALPHA_AUX)
    args = parser.parse_args()

    train_sae.train(
        consts.CHECKPOINT_PATH,
        consts.BATCH_INITS_PATH,
        consts.MODEL_PATH,
        consts.val_start_time,
        consts.val_end_time,
        stage_idx=args.task_id,
        method=args.method,
        expansion_factor=args.expansion_factor,
        l1_coeff=args.l1_coeff,
        k=args.k,
        k_aux=args.k_aux,
        alpha_aux=args.alpha_aux,
    )


if __name__ == "__main__":
    main()
