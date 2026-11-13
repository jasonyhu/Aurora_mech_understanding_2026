"""Download CAMS global atmospheric-composition forecasts for every 12-hourly init
in the data range into OAK_CAMS_DATA. Thin wrapper around mirora.setup.download_cams.

Shards the inits across `task_ct` parallel tasks (task k takes every task_ct-th init):

    python3 download_cams.py <task_id> <task_ct>

Use `0 1` for a single process, or launch tasks 0..N-1 concurrently. Each task is
resumable and skips inits that are already downloaded."""

import sys
import logging
from datetime import timedelta

from mirora.setup import download_cams
from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)


def main():
    assert len(sys.argv) == 3, "INTENDED USAGE: python3 download_cams.py <task_id> <task_ct>"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])
    setup_logging()

    cur_time = consts.data_start_time + timedelta(hours=12 * task_id)
    while cur_time <= consts.data_end_time:
        if not download_cams.verify_one_download(consts.OAK_CAMS_DATA, cur_time):
            logger.info(f"Task {task_id} / {task_ct}: Downloading missing data for {cur_time}")
            download_cams.download_cams(consts.OAK_CAMS_DATA, cur_time)
        else:
            logger.info(f"Task {task_id} / {task_ct}: Skipping {cur_time}, already downloaded.")
        cur_time += timedelta(hours=12 * task_ct)


if __name__ == "__main__":
    main()
