#!/usr/bin/env python3
"""Fetch and process everything the manuscript figures need, from scratch.

The figures are built from public CAMS global atmospheric-composition forecasts
plus the public Aurora checkpoint; no private data is required. This script
downloads only the handful of initialisation times the figures actually use
(FIGURE_INITS, not the full validation year) and processes them into the rollouts
`reproduce_figures.py` consumes.

    # 0. one-time: a Copernicus ADS API key in ~/.cdsapirc  (see REPRODUCE.md)
    python3 mirora/scripts/prepare_figure_data.py --steps cams
    python3 mirora/scripts/prepare_figure_data.py --steps batches
    python3 mirora/scripts/prepare_figure_data.py --steps rollouts   # GPU

    # or everything in order:
    python3 mirora/scripts/prepare_figure_data.py --all

Every step is resumable: anything already on disk is skipped.

Rough cost: ~17 GB of CAMS downloads (slow -- the CDS queue dominates), then
~30 min on one >=32 GB GPU for the rollouts.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from mirora.data.aurora_batches import save_init_batch, verify_init_batch
from mirora.setup import download_cams as cams_mod
from mirora.utils import get_init_batch_path, setup_logging

import consts

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# The only initialisations any manuscript figure depends on.
FIGURE_INITS: dict[datetime, str] = {
    consts.MAIN_INIT_TIME: "main init - chemistry, ozone, SAE grid, steering, SI",
    **{datetime.strptime(e["init"], r"%Y%m%d_%H%M"): f"wildfire - {e['label']}" for e in consts.FIRE_EVENTS},
    datetime(2022, 11, 1, 0): "SAE activation examples",
}

# Rollout-producing scripts, run in this order. Each is already idempotent.
ROLLOUT_SCRIPTS = [
    ("run_nox_perturbations.py", "base + NOx +0.5/+5/+20 ppb (2022-09-01)"),
    ("run_fire_base_rollouts.py", "base rollouts for the wildfire case studies"),
    ("run_single_cell_nox.py", "single-grid-cell NOx spike (SI)"),
]


def cams_times_needed() -> list[datetime]:
    """Each init batch is built from CAMS at t and t-12h."""
    times: set[datetime] = set()
    for init in FIGURE_INITS:
        times.add(init)
        times.add(init - timedelta(hours=12))
    return sorted(times)


def step_cams(dry_run: bool) -> int:
    times = cams_times_needed()
    logger.info(f"CAMS forecasts needed: {len(times)} "
                f"(2 per figure init: t and t-12h)")
    missing = [t for t in times
               if not cams_mod.verify_one_download(consts.CAMS_DATA_PATH, t)]
    logger.info(f"{len(times) - len(missing)} already present, {len(missing)} to download")
    if dry_run:
        for t in missing:
            logger.info(f"  would download {t}")
        return 0
    for t in missing:
        logger.info(f"Downloading CAMS {t} (CDS queueing can take a while)")
        try:
            cams_mod.download_cams(consts.CAMS_DATA_PATH, t)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            logger.error(f"FAILED {t}: {exc}")
    still = [t for t in times
             if not cams_mod.verify_one_download(consts.CAMS_DATA_PATH, t)]
    if still:
        logger.error(f"{len(still)} CAMS download(s) still missing; re-run this step.")
        return 1
    logger.info("All required CAMS forecasts present.")
    return 0


def step_batches(dry_run: bool) -> int:
    if not consts.STATIC_VARS_PATH.exists():
        logger.error(f"Missing Aurora static vars at {consts.STATIC_VARS_PATH}. "
                     f"Download the checkpoint first (see REPRODUCE.md).")
        return 1
    rc = 0
    for init in FIGURE_INITS:
        path = get_init_batch_path(consts.BATCH_INITS_PATH, init)
        if verify_init_batch(path):
            logger.info(f"init batch exists, skipping: {init}")
            continue
        if dry_run:
            logger.info(f"  would build init batch {init}")
            continue
        logger.info(f"Building init batch {init}")
        try:
            save_init_batch(init, consts.BATCH_INITS_PATH,
                            consts.CAMS_DATA_PATH, consts.STATIC_VARS_PATH)
        except FileNotFoundError as err:
            logger.error(f"Cannot build {init}: missing CAMS. Run --steps cams first.\n{err}")
            rc = 1
    return rc


def step_rollouts(dry_run: bool) -> int:
    if not consts.CHECKPOINT_PATH.exists():
        logger.error(f"Missing Aurora checkpoint at {consts.CHECKPOINT_PATH}. "
                     f"See REPRODUCE.md.")
        return 1
    rc = 0
    for script, desc in ROLLOUT_SCRIPTS:
        path = HERE / script
        if not path.exists():
            logger.warning(f"skipping missing script {script}")
            continue
        logger.info(f"=== {desc} ===")
        logger.info(f"    $ {sys.executable} {path}")
        if dry_run:
            continue
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            logger.error(f"{script} exited {result.returncode}")
            rc = 1
    return rc


STEPS = {"cams": step_cams, "batches": step_batches, "rollouts": step_rollouts}


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", choices=list(STEPS),
                   help="which stages to run")
    p.add_argument("--all", action="store_true",
                   help="run cams -> batches -> rollouts in order")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would happen without doing it")
    p.add_argument("--list-inits", action="store_true",
                   help="show the initialisations the figures depend on")
    args = p.parse_args()

    if args.list_inits:
        print(f"{'INIT':<20} PURPOSE")
        print("-" * 70)
        for init, why in FIGURE_INITS.items():
            print(f"{init:%Y-%m-%d %H:%MZ}   {why}")
        print(f"\n{len(cams_times_needed())} CAMS forecasts required "
              f"(each init also needs t-12h).")
        return 0

    selected = list(STEPS) if args.all else (args.steps or [])
    if not selected:
        p.print_help()
        return 0

    logger.info(f"Storage root: {consts.SCRATCH_PATH}")
    for name in selected:
        logger.info(f"########## step: {name} ##########")
        rc = STEPS[name](args.dry_run)
        if rc != 0:
            logger.error(f"step '{name}' reported problems; stopping.")
            return rc
    logger.info("Done. Now run: python3 mirora/scripts/reproduce_figures.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
