"""Build the pickled Aurora init Batch for every 12-hourly validation init from the
OAK CAMS data and write it to OAK_BATCH_INITS.
Usage: python3 save_init_batches.py <task_id> <task_ct> (task k of n takes every n-th init)."""

import sys
import logging
from datetime import timedelta

from mirora.data.aurora_batches import save_init_batch
from mirora.utils import setup_logging
import consts

logger = logging.getLogger(__name__)


def main():
    assert len(sys.argv) == 3, "INTENDED USAGE: python3 save_init_batches.py <task_id> <task_ct>"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])
    setup_logging()

    cur_time = consts.val_start_time + timedelta(hours=12 * task_id)
    while cur_time <= consts.val_end_time:
        try:
            save_init_batch(cur_time, consts.OAK_BATCH_INITS, consts.OAK_CAMS_DATA, consts.STATIC_VARS_PATH)
        except FileNotFoundError as err:
            logger.warning(f"CAMS data to create batch at {cur_time} cannot be found.\n{err}")
        cur_time += timedelta(hours=12 * task_ct)


if __name__ == "__main__":
    main()
