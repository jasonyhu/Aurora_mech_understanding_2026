import os
from datetime import datetime
from pathlib import Path

from mirora.config import M_AIR, MOLAR_MASS
from mirora.data.loading import available_init_times

data_start_time = datetime(2022, 5, 31, 12)
data_end_time = datetime(2023, 5, 31, 12)

val_start_time = datetime(2022, 6, 1, 0)
val_end_time   = datetime(2023, 5, 31, 12)  # inclusive

# Running example used by paper figures
MAIN_INIT_TIME = datetime(2022, 9, 1, 0)
MAIN_INIT_STR  = MAIN_INIT_TIME.strftime(r"%Y%m%d_%H%M")

def _storage_root(override_var: str, base_var: str, subpath: str, default: Path | None = None) -> Path:
    """Resolve a storage root from env vars, with no hardcoded user/group paths.

    Uses ``$<override_var>`` if set (treated as the full mirora root); otherwise
    ``$<base_var>/<subpath>``; otherwise `default`. On Sherlock ``$SCRATCH`` and
    ``$OAK`` are defined automatically, so this works for any user without editing.
    """
    override = os.environ.get(override_var)
    if override:
        return Path(override)
    base = os.environ.get(base_var)
    if base:
        return Path(base) / subpath
    if default is not None:
        return default
    raise EnvironmentError(
        f"Set ${override_var} to your MIRORA storage root "
        f"(on Sherlock ${base_var} is set automatically)."
    )

# Storage roots — override on a non-Sherlock system via MIRORA_SCRATCH / MIRORA_OAK.
# OAK (the persistent archive tier) falls back to SCRATCH for a single-tier setup.
SCRATCH_PATH = _storage_root("MIRORA_SCRATCH", "SCRATCH", "mirora")
OAK_PATH = _storage_root("MIRORA_OAK", "OAK", "MLchem/mirora", default=SCRATCH_PATH)

MODEL_PATH       = SCRATCH_PATH / "model"
CAMS_DATA_PATH   = SCRATCH_PATH / "cams_forecasts"
EXPERIMENTS_PATH = SCRATCH_PATH / "experiments"
FIGURES_PATH     = SCRATCH_PATH / "figures"
BATCH_INITS_PATH = SCRATCH_PATH / "init_batches"
SAE_DIR          = MODEL_PATH / "sae"
FINN_DIR         = SCRATCH_PATH / "finn"

OAK_CAMS_DATA    = OAK_PATH / "cams_forecasts"
OAK_BATCH_INITS  = OAK_PATH / "init_batches"

CHECKPOINT_PATH  = MODEL_PATH / "aurora-0.4-air-pollution.ckpt"
STATIC_VARS_PATH = MODEL_PATH / "aurora-0.4-air-pollution-static.pickle"

def rollout_path(exp_dir: Path, init: datetime | str) -> Path:
    """predictions.pickle of the rollout initialised at `init` (datetime or 'YYYYMMDD_HHMM')."""
    init_str = init if isinstance(init, str) else init.strftime(r"%Y%m%d_%H%M")
    return exp_dir / init_str / "predictions.pickle"


def base_rollout_path(init: datetime | str) -> Path:
    """predictions.pickle of the unperturbed rollout initialised at `init`."""
    return rollout_path(EXPERIMENTS_PATH / "base", init)


def ensure_checkpoint():
    """Download the Aurora checkpoint into MODEL_PATH if it is missing (e.g. purged)."""
    if not CHECKPOINT_PATH.exists():
        from mirora.setup.download_model import download_model_checkpoint
        download_model_checkpoint(MODEL_PATH)

def val_init_times() -> list[datetime]:
    """Every 12-hourly init in the validation year that has a saved init batch."""
    return available_init_times(BATCH_INITS_PATH, val_start_time, val_end_time)

NOX_CONVENTION = os.environ.get("MIRORA_NOX_CONVENTION", "mmr")
NOX_MAGNITUDES_PPB = [0.5, 5, 20]

NOX_EXPERIMENTS_PATH = EXPERIMENTS_PATH / f"nox_{NOX_CONVENTION}"
NOX_ANALYSIS_PATH    = SCRATCH_PATH / "analysis" / f"nox_{NOX_CONVENTION}"
NOX_FIGURES_PATH     = FIGURES_PATH / f"nox_{NOX_CONVENTION}"

NOX_UNIT_LABEL = {
    "mmr": r"ppb (mass, each of NO and NO$_2$)",
}[NOX_CONVENTION]

def nox_experiment_dir(ppb: float) -> Path:
    """Experiment directory of the +`ppb` NOx perturbation rollouts."""
    return NOX_EXPERIMENTS_PATH / f"NOx_{ppb:g}ppb_increase"


def nox_offsets(ppb: float) -> dict[str, float]:
    """Mass-mixing-ratio offsets (kg/kg) for NO and NO2 for a nominal `ppb` perturbation."""
    if NOX_CONVENTION == "mmr":
        return {"no": ppb * 1e-9, "no2": ppb * 1e-9}
    half = ppb / 2 * 1e-9  # mole fraction per species
    return {sp: half * MOLAR_MASS[sp] / M_AIR for sp in ("no", "no2")}


# ==========================================
# SAE case-study features (steering figure)
# ==========================================
# The two features ablated in the steering figure: the largest O3 response and
# a feature chosen for NO2 (from tableS5_S6_rank_and_ablate_top_sae_features.py).
CASE_STUDY_FEATURES = [
    {"name": "decoder_s1_b9_feat2712", "species": "ozone", "component": "decoder",
     "stage": 1, "block": 9, "dim": 1024, "feat_idx": 2712},
    {"name": "encoder_s1_b5_feat2866", "species": "no2", "component": "encoder",
     "stage": 1, "block": 5, "dim": 1024, "feat_idx": 2866},
]

# ==========================================
# Wildfire case studies
# ==========================================
FIRE_EVENTS = [
    {"name": "spain_jul2022", "label": "Spain", "init": "20220715_1200",
     "bbox": [30, 50, -15, 10], "finn_country": "Spain"},
    {"name": "alaska_jul2022", "label": "Alaska", "init": "20220703_0000",
     "bbox": [40, 75, -170, -80], "finn_country": None},
    {"name": "central_russia_aug2022", "label": "Central Russia", "init": "20220819_1200",
     "bbox": [35, 75, 10, 80], "finn_country": "Russia"},
]
