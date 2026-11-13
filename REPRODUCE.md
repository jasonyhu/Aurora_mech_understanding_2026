# Reproducing figures

There are two starter scripts to reproduce the figures of the paper:
```bash
# 1. fetch + process the public data the figures need
python3 mirora/scripts/prepare_figure_data.py --all

# 2. render every figure
python3 mirora/scripts/reproduce_figures.py --all
```

---

## 1. Install

```bash
git clone https://github.com/jasonyhu/Aurora_mech_understanding_2026.git && cd Aurora_mech_understanding_2026
cd mirora && pip install -e ".[all]" && cd ..
```

## 2. Choose a folder to use as scratch

Set one environment variable:

```bash
export MIRORA_SCRATCH=/path/to/scratch/folder
```

## 3. Fetch data

**Copernicus ADS (CAMS forecasts)**

Dataset used: [CAMS global atmospheric composition forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts)
  (variables and lead times are set in `mirora/src/mirora/setup/download_cams.py`)

**Aurora checkpoint and sparse autoencoders**:

```bash
# Aurora Air Pollution checkpoint (microsoft/aurora)
python3 -c "import sys; sys.path.insert(0,'mirora/scripts'); import consts; consts.ensure_checkpoint()"

# download AuroraScope
python3 mirora/scripts/download_aurorascope_saes.py
```

**FINN emissions**: https://gdex.ucar.edu/accounts/profile/

```bash
python3 mirora/scripts/download_finn_fire_emissions.py
```

## 4. Fetch and process the data

```bash
python3 mirora/scripts/prepare_figure_data.py --list-inits
python3 mirora/scripts/prepare_figure_data.py --all
```

The figures depend on only 5 initialisation times, not the full validation year, so this downloads 10 CAMS forecasts (each init requires two timesteps).

## 5. Render the figures

```bash
python3 mirora/scripts/reproduce_figures.py --check   # confirm inputs
python3 mirora/scripts/reproduce_figures.py --all
```

Figures are written under `$MIRORA_SCRATCH/figures/`; the NOx-perturbation figures go to
`figures/nox_mmr/`.

```bash
reproduce_figures.py --list
reproduce_figures.py --figure ozone_perturbation
reproduce_figures.py --tier 1
reproduce_figures.py --dry-run --all
```

---

## Full pipeline

To rebuild everything from scratch, follow the numbered runbook in
[`mirora/scripts/README.md`](mirora/scripts/README.md):

1. `download_cams_forecasts.py` — the full CAMS validation year
2. `build_init_batches.py` — Aurora `Batch` objects
3. `run_base_rollouts_with_embeddings.py` — rollouts + first-step backbone embeddings (the ~2.5 TB step)
4. `train_aurorascope_saes.py` — train the 48 sparse autoencoders (~3 h each on one GPU)
5. `reproduce_figures.py --all`
---

## Citation

```bibtex
@misc{hu2026mechanisticinterpretabilityaifoundation,
      title={Toward Mechanistic Interpretability of an AI Foundation Model Fine-Tuned for Atmospheric Chemistry},
      author={Jason Y. Hu and Ivan Higuera-Mendieta and Patrick Obin Sturm and Makoto M. Kelp},
      year={2026},
      eprint={2607.20778},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.20778},
}
```
