"""
Negative-concentration sweep for the SI table: fraction of surface (1000 hPa)
grid cells with negative CO, NO, NO2, O3, SO2 (and any species) in Aurora's
+12 h forecast, for the base run and the +0.5/+5/+20 ppb NOx perturbations,
across every init in the validation year.

Each scenario is a single 12 h step. The per-species row builder is shared with
aggregate_negative_conc.py; writes a small per-task CSV to
NOX_ANALYSIS_PATH/negative_conc/ (i.e. per MIRORA_NOX_CONVENTION). The CAMS and
base +120 h columns come from aggregate_negative_conc.py.

USAGE: python3 run_negative_conc_sweep.py <task_id> <task_ct>
"""

import copy
import logging
import sys

import pandas as pd
import torch
from aurora import rollout

from mirora.data.loading import load_batch
from mirora.experiments.perturbation import perturb_batch
from mirora.inference.inference import load_aurora
from mirora.stats import lat_weights
from mirora.utils import setup_logging

import consts
from aggregate_negative_conc import neg_fraction_rows, surface_fields

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    assert len(sys.argv) == 3, "USAGE: python3 run_negative_conc_sweep.py <task_id> <task_ct>"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])

    out_dir = consts.NOX_ANALYSIS_PATH / "negative_conc"
    out_dir.mkdir(parents=True, exist_ok=True)

    init_times = consts.val_init_times()
    mine = init_times[task_id::task_ct]
    logger.info(f"Task {task_id}/{task_ct}: {len(mine)} of {len(init_times)} inits, "
                f"convention={consts.NOX_CONVENTION}")

    consts.ensure_checkpoint()
    model, _ = load_aurora(consts.CHECKPOINT_PATH)

    rows = []
    for init_time in mine:
        init_str = init_time.strftime(r"%Y%m%d_%H%M")
        base_batch = load_batch(consts.BATCH_INITS_PATH, init_time)
        scenarios = {"base": base_batch}
        for ppb in consts.NOX_MAGNITUDES_PPB:
            scenarios[f"+{ppb}"] = perturb_batch(copy.deepcopy(base_batch), atmos_vars=consts.nox_offsets(ppb))

        for name, batch in scenarios.items():
            with torch.inference_mode():
                pred = next(iter(rollout(model, batch, steps=1))).to("cpu")
            rows += neg_fraction_rows(surface_fields(pred), lat_weights([pred]), init_str, name)
        logger.info(f"  {init_str} done")

    out_path = out_dir / f"negative_conc_partial_{task_id:03d}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
