# SLURM job scripts

Each `*.sh` here is a thin SLURM wrapper: it sets resource constraints, runs
`source spinup.sh`, then invokes the matching `../python/*.py` entry point
(passing `$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT` for array jobs).

Submit from the repository root:

```bash
sbatch mirora/scripts/slurm/train_sae.sh
```

## One-time setup: the `logs/` directory

`#SBATCH` directives cannot expand environment variables, so job logs are
written to a **relative `logs/` directory** (resolved against the directory you
submit from — the repo root). It is gitignored. Create it before submitting:

```bash
# On Sherlock: keep logs on scratch (NOT $HOME — org policy forbids job output
# in $HOME) by symlinking logs/ to a scratch location:
mkdir -p "$SCRATCH/mirora/logs"
ln -s "$SCRATCH/mirora/logs" logs

# Elsewhere: a plain directory is fine:
mkdir -p logs
```

If `logs/` does not exist, jobs fail immediately (slurmd cannot open the output
file).

## What you must edit for a different cluster / account

These lines are **Sherlock- and group-specific** — review them before running
anywhere else:

- **`#SBATCH --partition=...`** — e.g. `serc,gpu`, `serc,normal`, `normal,serc`.
  `serc` is a Stanford owner partition; replace with your cluster's partitions.
- **`#SBATCH --constraint=GPU_MEM:...`** — GPU-memory constraints (Aurora OOMs
  below ~32 GB). Adjust to your available GPUs.
- **`spinup.sh`** (repo root) — loads Sherlock Lmod modules
  (`python/3.12.1`, `cuda/12.8.0`, `ffmpeg/7.1`) and sets `HF_HOME`. On another
  system, replace with your own environment activation.

## Data paths

Storage roots come from `../python/consts.py`, which resolves them from
`$SCRATCH` / `$OAK` (or the `$MIRORA_SCRATCH` / `$MIRORA_OAK` overrides). No
paths are hardcoded to a username — set the override vars off Sherlock.
