import sys
import logging
from argparse import ArgumentParser
from datetime import timedelta
from mirora.data.aurora_batches import save_init_batch, verify_all_init_batches
import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def save_all_val_batches(thread_id: int=0, thread_ct: int=1):
    """
    Save all AuroraAirPollution initialization batches.
    """
    # if thread_id == 0:
    #     verify_all_init_batches(consts.BATCH_INITS_PATH, consts.val_start_time, consts.val_end_time)
    cur_time = consts.val_start_time + timedelta(hours=12 * thread_id)
    while cur_time <= consts.val_end_time:
        try:
            save_init_batch(cur_time, consts.OAK_BATCH_INITS, consts.OAK_CAMS_DATA, consts.STATIC_VARS_PATH)
        except FileNotFoundError as err:
            logger.warning(f"CAMS data to create batch at {cur_time} cannot be found.\n{err}")
        cur_time += timedelta(hours=12 * thread_ct)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--thread_id', default=0)
    parser.add_argument('--thread_ct', default=1)
    args = parser.parse_args()
    save_all_val_batches(int(args.thread_id), int(args.thread_ct))
