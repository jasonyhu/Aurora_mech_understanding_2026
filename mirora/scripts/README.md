# Scripts

> **Just want the figures?** Use [`REPRODUCE.md`](../../REPRODUCE.md) — you only
> need `prepare_figure_data.py` then `reproduce_figures.py`. This file documents
> the individual scripts those two drive.

Every script here is an entry point: a thin wrapper that parses CLI args, imports
the `mirora` package and `consts.py` (storage paths), and calls library
functions. Run them directly:

```bash
python3 mirora/scripts/<name>.py
```

Scripts marked **array** below shard their work across `task_ct` parallel tasks
and take `<task_id> <task_ct>` as positional arguments; run them serially with
`0 1`, or launch tasks `0..N-1` concurrently. Scripts marked **GPU** run Aurora
live and need >= 32 GB of GPU memory.

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
| `run_base_rollout` | array, GPU | Base rollout for every validation init; saves predictions + backbone embeddings |
| `run_nox_perturbations` | GPU | Base + NOx +0.5/+5/+20 ppb for the main init, or `--year [--ppb X]` (array) over every init |
| `run_fire_base_rollouts` | GPU | Base rollouts for the wildfire case studies (skips existing) |
| `run_single_cell_nox` | GPU | Single-grid-cell NOx spike (SI) |

The NOx unit convention is set by `MIRORA_NOX_CONVENTION`: `mmr` (default, used
in the paper; +X adds X·10⁻⁹ kg/kg to each of NO and NO₂) or `ppbv` (+X ppbv of
total NOx). Results for each convention go to separate `nox_{mmr,ppbv}/` folders.

## Sparse autoencoders

| Script | Run | Purpose |
|---|---|---|
| `train_sae` | array, GPU | Train one SAE per backbone block (48 tasks; `--method l1` or `topk`) |
| `compute_fve` | GPU | Fraction of variance explained (`--method l1` or `topk`) |
| `compare_sae_methods` | GPU | L1 vs BatchTopK comparison figures |
| `find_top_activating_features` | GPU | Rank SAE features by variance contribution and ablate the top 10 (`--init`) |
| `run_sae_steering` | GPU | Zero-ablate the two case-study features (`--ablations zero mean`) |
| `steering_null_control` | array, GPU | Case-study ablations vs matched random-feature nulls |
| `aggregate_steering_null` | python | Summarize the null control |
| `aggregate_sae_consistency` | python | Feature recurrence + ablation tables across 12 monthly inits (SI) |

## NOx robustness and diagnostics (SI)

| Script | Run | Purpose |
|---|---|---|
| `run_nox_robustness_sweep` | array, GPU | +5 ppb response at every validation init |
| `aggregate_nox_robustness` | python | Per-season summary of the sweep |
| `plot_nox_robustness` | python | Distribution figures for the sweep |
| `run_negative_conc_sweep` | array, GPU | Fraction of negative concentrations at +12 h, every init |
| `aggregate_negative_conc` | python | Negative-concentration table with CAMS + base references |
| `compute_perturbation_decay` | array | Remaining NO perturbation vs lead time |
| `aggregate_perturbation_decay` | python | Perturbation-decay table |
| `compute_base_diagnostics` | array | Masked fractions, CAMS agreement and RMSE vs CAMS analysis by lead (`--aggregate`) |
| `plot_diagnostic_cases` | array | CAMS vs Aurora chemistry diagnostics for selected cases (`--plot`) |
| `quantify_patch_artifacts` | python | Strength of the 3×3 patch mosaic by species and region |
| `rotate_pca_loadings` | python | Promax (oblique) rotation of the 10-component annual PCA |

## Figures

| Script | Run | Figure |
|---|---|---|
| `plot_architecture_diagram` | python | Model architecture |
| `plot_chemistry_products` | python | Leighton ratio + NOx ratio, CAMS vs Aurora |
| `plot_ozone_perturbation` | python | NOx → ozone response composite + KDEs (`--negative-map`, `--negative-species`) |
| `plot_fire_stills` | python | Wildfire case studies (`--event`, `--var`, `--distributions`) |
| `run_pca_analysis` | python | PCA of backbone embeddings (`--scales`, `--n-components`, `--no-title`) |
| `plot_sae_feature_grid` | GPU | SAE feature grid; `--top10`, or one `--feature C S B K` |
| `plot_sae_steering_diffs` | python | Steering difference maps (Fig 7) |
| `plot_sae_examples` | GPU | SAE activation examples (6×2) |
| `plot_single_cell_nox` | python | Single-cell NOx spike maps (SI) |

## Validation

| Script | Run | Purpose |
|---|---|---|
| `validate_nox_claims` | python | Recompute the NOx perturbation statistics quoted in the paper |
| `consts` | (imported) | Storage paths and shared settings; imported by the others, not run directly |
