"""Download CAMS global atmospheric-composition forecasts for a range of init
times (SLURM array over dates). Thin wrapper around mirora.setup.download_cams."""

import os
import sys
import logging
from datetime import timedelta

from mirora.setup import download_cams

import consts

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

assert len(sys.argv) == 3, "INTENDED USAGE: python3 download_cams.py <task_idx> <task_ct>"
task_idx, task_ct = int(sys.argv[1]), int(sys.argv[2])

if task_idx == 0 and not download_cams.verify_all_downloads(consts.OAK_CAMS_DATA, consts.data_start_time, consts.data_end_time):
    os.system('sbatch mirora/scripts/slurm/download_cams.sh')

cur_time = consts.data_start_time + timedelta(hours=12*task_idx)

while cur_time <= consts.data_end_time:
    if not download_cams.verify_one_download(consts.OAK_CAMS_DATA, cur_time):
        logger.info(f"Task {task_idx} / {task_ct}: Downloading missing data for {cur_time}")
        download_cams.download_cams(consts.OAK_CAMS_DATA, cur_time)
    else:
        logger.info(f"Task {task_idx} / {task_ct}: Skipping {cur_time}, already downloaded.")
    cur_time += timedelta(hours=12*task_ct)