import os
from datetime import datetime
from pathlib import Path

data_start_time = datetime(2022, 5, 31, 12)
data_end_time = datetime(2023, 5, 31, 12)

val_start_time = datetime(2022, 6, 1, 0)
val_end_time   = datetime(2023, 5, 31, 12)  # inclusive

def _storage_root(override_var: str, base_var: str, subpath: str) -> Path:
    """Resolve a storage root from env vars, with no hardcoded user/group paths.

    Uses ``$<override_var>`` if set (treated as the full mirora root); otherwise
    ``$<base_var>/<subpath>``. On Sherlock ``$SCRATCH`` and ``$OAK`` are defined
    automatically, so this works for any user without editing.
    """
    override = os.environ.get(override_var)
    if override:
        return Path(override)
    base = os.environ.get(base_var)
    if base:
        return Path(base) / subpath
    raise EnvironmentError(
        f"Set ${override_var} to your MIRORA storage root "
        f"(on Sherlock ${base_var} is set automatically)."
    )

# Storage roots — override on a non-Sherlock system via MIRORA_SCRATCH / MIRORA_OAK.
SCRATCH_PATH = _storage_root("MIRORA_SCRATCH", "SCRATCH", "mirora")
OAK_PATH = _storage_root("MIRORA_OAK", "OAK", "MLchem/mirora")

MODEL_PATH       = SCRATCH_PATH / "model"
CAMS_DATA_PATH   = SCRATCH_PATH / "cams_forecasts"
EXPERIMENTS_PATH = SCRATCH_PATH / "experiments"
FIGURES_PATH     = SCRATCH_PATH / "figures"
BATCH_INITS_PATH = SCRATCH_PATH / "init_batches"
LEIGHTON_PATH    = SCRATCH_PATH / "leighton"

OAK_CAMS_DATA    = OAK_PATH / "cams_forecasts"
OAK_BATCH_INITS  = OAK_PATH / "init_batches"

CHECKPOINT_PATH  = MODEL_PATH / "aurora-0.4-air-pollution.ckpt"
STATIC_VARS_PATH = MODEL_PATH / "aurora-0.4-air-pollution-static.pickle"

def get_base_experiment_path(init_time: datetime) -> Path:
    """
    Retrieves the directory path to the AuroraAirPollution rollout initialized at `start_time`.
    The 10 rollout timesteps range from `init_time` + 12 hours to `init_time` + 120 hours.
    """
    batch_path = EXPERIMENTS_PATH / "base" / init_time.strftime(r"%Y%m%d_%H%M")
    return batch_path

# Variable descriptions and pressure levels live canonically in mirora.config
# (SURF_VAR_DESCRIPTIONS, ATMOS_VAR_DESCRIPTIONS, PRESSURE_LEVELS).
