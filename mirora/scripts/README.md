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

Script names start with the manuscript item they produce (`fig3_`, `figS4_`,
`tableS1_`, `sec3_2_`, …), followed by what they do (`run_` a rollout, `plot_`,
`compute_`/`aggregate_` a table). Pipeline scripts that feed many items have no
prefix.

Scripts marked **array** below shard their work across `task_ct` parallel tasks
and take `<task_id> <task_ct>` as positional arguments; run them serially with
`0 1`, or launch tasks `0..N-1` concurrently. Scripts marked **GPU** run Aurora
live and need >= 32 GB of GPU memory.

---

## Manuscript item → script

| Item | Script(s) |
|---|---|
| Figure 2 | `fig2_plot_nox_partitioning_leighton_j1` |
| Figure 3 | `fig3_run_global_nox_perturbations` → `fig3_figS2_S3_plot_ozone_nox_response` |
| Figure 4 | `fig4_run_wildfire_base_rollouts` → `fig4_figS5_S7_plot_wildfire_co_column` (Spain) |
| Figure 5 | `fig5_figS8_S9_tableS2_run_latent_pca` (single day) |
| Figure 6 | `fig6_plot_sae_feature_maps` |
| Figure 7 | `fig7_run_sae_ablation_rollouts` → `fig7_plot_sae_ablation_diff_maps` |
| Table 1 | `table1_compute_sae_fve` |
| Text S1 | `figS4_run_single_column_nox_perturbation` → `figS4_plot_single_column_nox_response` |
| Table S1 | `tableS1_compute_baseline_rmse_and_masking` (`--aggregate`) |
| Table S2 | `fig5_figS8_S9_tableS2_run_latent_pca` |
| Table S3 | none: dictionary sizes are 8 × d_model per block of `mirora.config.STAGES` |
| Table S4 | `tableS4_run_negative_conc_sweep` → `tableS4_aggregate_negative_conc` |
| Tables S5, S6 | `tableS5_S6_rank_and_ablate_top_sae_features` (once per monthly init) → `tableS5_S6_aggregate_sae_feature_recurrence` |
| Figure S1 | `tableS1_compute_baseline_rmse_and_masking` → `figS1_plot_photochem_diagnostics_by_case` (`--plot`) |
| Figure S2 | `fig3_figS2_S3_plot_ozone_nox_response --negative-map` |
| Figure S3 | `fig3_figS2_S3_plot_ozone_nox_response` (`o3_stacked_si`) |
| Figure S4 | `figS4_plot_single_column_nox_response` |
| Figures S5, S6 | `fig4_figS5_S7_plot_wildfire_co_column` (Alaska, Central Russia) |
| Figure S7 | `fig4_figS5_S7_plot_wildfire_co_column --distributions` |
| Figures S8, S9 | `fig5_figS8_S9_tableS2_run_latent_pca` (monthly, annual) |
| Figures S10, S11 | `fig5_figS8_S9_tableS2_run_latent_pca --n-components 10` → `figS10_S11_plot_promax_rotated_pcs` |
| Figure S12 | `figS12_plot_sae_feature_examples` |
| Section 3.1 masked fractions | `tableS1_compute_baseline_rmse_and_masking --aggregate` |
| Section 3.2 NOx statistics | `fig3_check_nox_perturbation_statistics` (Sep 1 2022), `sec3_2_run_nox_response_all_inits` → `sec3_2_aggregate_nox_response_all_inits` (all 730 inits) |
| Section 3.5 top-10 ablations | `tableS5_S6_rank_and_ablate_top_sae_features` |

---

## Drivers (start here)

| Script | Run | Purpose |
|---|---|---|
| `prepare_figure_data` | python | Fetch + process the public data every figure needs (`--all`, `--steps`, `--list-inits`, `--dry-run`) |
| `reproduce_figures` | python | Render the manuscript figures (`--list`, `--check`, `--all`, `--figure ID`, `--tier N`) |

## Data acquisition and pipeline

| Script | Run | Purpose |
|---|---|---|
| `download_cams_forecasts` | array | Download CAMS composition forecasts (array over dates) |
| `download_finn_fire_emissions` | python | Download FINN v2.5 gridded fire emissions (NCAR GDEX) for the wildfire cases |
| `download_aurorascope_saes` | python | Fetch the 48 trained AuroraScope SAEs from HuggingFace |
| `build_init_batches` | array | Build pickled Aurora `Batch` init objects from CAMS |
| `run_base_rollouts_with_embeddings` | array, GPU | Base rollout for every validation init; saves predictions + backbone embeddings |
| `train_aurorascope_saes` | array, GPU | Train one L1 SAE per operator block (48 tasks) |

## Photochemistry and species coupling (Section 3.1)

| Script | Run | Purpose |
|---|---|---|
| `fig2_plot_nox_partitioning_leighton_j1` | python | NOx partitioning ratio + Leighton-derived J1, CAMS vs Aurora |
| `tableS1_compute_baseline_rmse_and_masking` | array | Masked fractions, CAMS agreement and RMSE vs CAMS analysis by lead (`--aggregate`) |
| `figS1_plot_photochem_diagnostics_by_case` | array | Figure 2 diagnostics across diurnal cycle, season and lead (`--plot`) |

## NOx perturbations (Section 3.2, Text S1)

| Script | Run | Purpose |
|---|---|---|
| `fig3_run_global_nox_perturbations` | GPU | Base + NOx +0.5/+5/+20 ppb for the main init, or `--year [--ppb X]` (array) over every init |
| `fig3_figS2_S3_plot_ozone_nox_response` | python | NOx → ozone response composite + KDEs (`--negative-map` for Figure S2) |
| `fig3_check_nox_perturbation_statistics` | python | Recompute the Sep 1 2022 NOx perturbation statistics quoted in the paper |
| `sec3_2_run_nox_response_all_inits` | array, GPU | +5 ppb response at every validation init |
| `sec3_2_aggregate_nox_response_all_inits` | python | Across-init and per-season summary of that sweep |
| `tableS4_run_negative_conc_sweep` | array, GPU | Fraction of negative concentrations at +12 h, every init |
| `tableS4_aggregate_negative_conc` | python | Negative-concentration table with CAMS + base references |
| `figS4_run_single_column_nox_perturbation` | GPU | Single-grid-cell NOx spike over Beijing |
| `figS4_plot_single_column_nox_response` | python | Single-cell NOx spike maps |

## Wildfire plume persistence (Section 3.3)

| Script | Run | Purpose |
|---|---|---|
| `fig4_run_wildfire_base_rollouts` | GPU | Base rollouts for the three wildfire cases (skips existing) |
| `fig4_figS5_S7_plot_wildfire_co_column` | python | CAMS vs Aurora total column CO with FINN detections (`--event`, `--var`, `--distributions`) |

## Latent representations (Section 3.4)

| Script | Run | Purpose |
|---|---|---|
| `fig5_figS8_S9_tableS2_run_latent_pca` | python | PCA of encoder/decoder latent levels (`--scales`, `--n-components`, `--no-title`) |
| `figS10_S11_plot_promax_rotated_pcs` | python | Promax (oblique) rotation of the 10-component annual PCA |

## Sparse autoencoders (Section 3.5)

| Script | Run | Purpose |
|---|---|---|
| `table1_compute_sae_fve` | GPU | Fraction of variance explained and L0 per SAE |
| `fig6_plot_sae_feature_maps` | GPU | SAE feature grid; `--top10`, or one `--feature C S B K` |
| `tableS5_S6_rank_and_ablate_top_sae_features` | GPU | Rank SAE features by variance contribution and ablate the top 10 (`--init`) |
| `tableS5_S6_aggregate_sae_feature_recurrence` | python | Feature recurrence + ablation tables across 12 monthly inits |
| `fig7_run_sae_ablation_rollouts` | GPU | Ablate the two case-study features (`--ablations zero mean`) |
| `fig7_plot_sae_ablation_diff_maps` | python | Steering difference maps |
| `figS12_plot_sae_feature_examples` | GPU | SAE activation examples (6×2) |

## Other

| Script | Run | Purpose |
|---|---|---|
| `consts` | (imported) | Storage paths and shared settings; imported by the others, not run directly |
