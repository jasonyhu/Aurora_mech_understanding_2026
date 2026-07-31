# Scripts

> **Just want the figures?** Use [`REPRODUCE.md`](../../REPRODUCE.md) — you only
> need `prepare_figure_data.py` then `reproduce_figures.py`. This file documents
> the individual scripts those two drive.

Everything in `python/` is an entry point: a thin script that parses CLI args,
imports the `mirora` package and `consts.py` (storage paths), and calls library
functions. Run them directly:

```bash
python3 mirora/scripts/python/<name>.py
```

Scripts marked
**array** below shard their work across `task_ct` parallel tasks and take
`<task_id> <task_ct>` as positional arguments; run them serially with `0 1`, or
launch tasks `0..N-1` concurrently. Scripts marked **GPU** run Aurora live and
need >= 32 GB of GPU memory.

---

## Drivers (start here)

| Script | Run | Purpose |
|---|---|---|
| `prepare_figure_data` | python | Fetch + process the public data every figure needs (`--all`, `--steps`, `--list-inits`, `--dry-run`) |
| `reproduce_figures` | python | Render the manuscript figures (`--list`, `--check`, `--all`, `--figure ID`, `--tier N`) |

## Data acquisition

| Script | Run | Purpose |
|---|---|---|
| `download_cams` | array | Download CAMS composition forecasts (array over dates) |
| `download_finn` | python | Download FINN v2.5 gridded fire emissions (NCAR GDEX) |
| `download_saes` | python | Fetch the 48 trained SAEs from HuggingFace |
| `save_init_batches` | array | Build pickled Aurora `Batch` init objects from CAMS |

## Model rollouts

| Script | Run | Purpose |
|---|---|---|
| `run_base_rollout` | array, GPU | Base rollout; saves predictions + backbone embeddings |
| `rerun_single_init_perturbations` | GPU | Base + NOx +0.5/+5/+20 ppb for the main init |
| `run_fire_base_rollouts` | python GPU | Base rollouts for the wildfire case studies |
| `run_single_cell_nox` | python GPU | Single-grid-cell NOx spike (SI) |

## Sparse autoencoders

| Script | Run | Purpose |
|---|---|---|
| `train_sae` | array, GPU | Train one L1 SAE per backbone block (48-task array) |
| `train_batch_topk_sae` | array, GPU | Train one BatchTopK SAE per block (48-task array) |
| `compute_fve` | GPU | Fraction of variance explained, L1 SAEs |
| `compute_fve_topk` | GPU | FVE + L0 sparsity, BatchTopK SAEs |

## Figures

| Script | Run | Figure |
|---|---|---|
| `plot_architecture_diagram` | python | Model architecture |
| `plot_chemistry_products` | python | Leighton ratio + NOx ratio, CAMS vs Aurora |
| `plot_ozone_perturbation` | python | NOx → ozone response composite + KDEs |
| `plot_fire_stills` | python | Wildfire case studies (`--event`, `--var`) |
| `run_pca_analysis` | python | PCA of backbone embeddings |
| `plot_sae_feature_grid` | GPU | SAE feature gallery |
| `plot_sae_examples` | GPU | SAE activation examples (6×2) |
| `plot_single_cell_nox` | python | Single-cell NOx spike maps (SI) |

## Validation

| Script | Run | Purpose |
|---|---|---|
| `validate_nox_claims` | python | Recompute the NOx perturbation statistics quoted in the paper |
| `consts` | (imported) | Storage paths; imported by the others, not run directly |
