# Scripts

> **Just want the figures?** Use [`REPRODUCE.md`](../../REPRODUCE.md) — you only
> need `prepare_figure_data.py` then `reproduce_figures.py`. This file documents
> the individual scripts those two drive.

Two layers:

- **`python/`** — entry points. Thin scripts that parse CLI args, import the
  `mirora` package and `consts.py` (storage paths), and call library functions.
- **`slurm/`** — SLURM wrappers that set resources, `source spinup.sh`, and run
  the matching `python/` script. See [`slurm/README.md`](slurm/README.md) for the
  one-time `logs/` setup and the cluster-specific directives.

Scripts with a wrapper are submitted with `sbatch mirora/scripts/slurm/<name>.sh`;
the rest run directly with `python3 mirora/scripts/python/<name>.py`.

---

## Drivers (start here)

| Script | Run | Purpose |
|---|---|---|
| `prepare_figure_data` | python | Fetch + process the public data every figure needs (`--all`, `--steps`, `--list-inits`, `--dry-run`) |
| `reproduce_figures` | python | Render the manuscript figures (`--list`, `--check`, `--all`, `--figure ID`, `--tier N`) |

## Data acquisition

| Script | Run | Purpose |
|---|---|---|
| `download_cams` | [sbatch] | Download CAMS composition forecasts (array over dates) |
| `download_finn` | python | Download FINN v2.5 gridded fire emissions (NCAR GDEX) |
| `download_saes` | python | Fetch the 48 trained SAEs from HuggingFace |
| `save_init_batches` | [sbatch] | Build pickled Aurora `Batch` init objects from CAMS |

## Model rollouts

| Script | Run | Purpose |
|---|---|---|
| `run_base_rollout` | [sbatch] GPU | Base rollout; saves predictions + backbone embeddings |
| `rerun_single_init_perturbations` | [sbatch] GPU | Base + NOx +0.5/+5/+20 ppb for the main init |
| `run_fire_base_rollouts` | python GPU | Base rollouts for the wildfire case studies |
| `run_single_cell_nox` | python GPU | Single-grid-cell NOx spike (SI) |

## Sparse autoencoders

| Script | Run | Purpose |
|---|---|---|
| `train_sae` | [sbatch] GPU | Train one L1 SAE per backbone block (48-task array) |
| `train_batch_topk_sae` | [sbatch] GPU | Train one BatchTopK SAE per block (48-task array) |
| `compute_fve` | [sbatch] GPU | Fraction of variance explained, L1 SAEs |
| `compute_fve_topk` | [sbatch] GPU | FVE + L0 sparsity, BatchTopK SAEs |

## Figures

| Script | Run | Figure |
|---|---|---|
| `plot_architecture_diagram` | python | Model architecture |
| `plot_chemistry_products` | [sbatch] | Leighton ratio + NOx ratio, CAMS vs Aurora |
| `plot_ozone_perturbation` | python | NOx → ozone response composite + KDEs |
| `plot_fire_stills` | [sbatch] | Wildfire case studies (`--event`, `--var`) |
| `run_pca_analysis` | [sbatch] | PCA of backbone embeddings |
| `plot_sae_thesis_grid` | [sbatch] GPU | SAE feature gallery |
| `plot_sae_examples` | [sbatch] GPU | SAE activation examples (6×2) |
| `plot_single_cell_nox` | python | Single-cell NOx spike maps (SI) |
| `make_feature_count_table` | python | SAE dictionary-size table (SI, LaTeX) |

## Validation

| Script | Run | Purpose |
|---|---|---|
| `validate_nox_claims` | python | Recompute the NOx perturbation statistics quoted in the paper |
| `consts` | (imported) | Storage paths; imported by the others, not run directly |
