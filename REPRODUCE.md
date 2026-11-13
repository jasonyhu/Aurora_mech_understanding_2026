# Reproducing the manuscript figures

All data in the project is public: CAMS global atmospheric-composition forecasts (Copernicus) and the public Aurora Air Pollution checkpoint.

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

## 2. Choose a storage location

Set one environment variable:

```bash
export MIRORA_SCRATCH=/path/to/mirora   # needs ~60 GB
```

The Aurora checkpoint and the SAEs are fetched through HuggingFace, which caches
to `~/.cache/huggingface` by default. If your home directory is small, point the
cache at the same volume:

```bash
export HF_HOME=$MIRORA_SCRATCH/.cache
```

## 3. Credentials for the public data

**Copernicus ADS (CAMS forecasts).** Free account, then put your key in
`~/.cdsapirc`:

- Register: https://ads.atmosphere.copernicus.eu/
- API key instructions: https://ads.atmosphere.copernicus.eu/how-to-api
- Dataset used: [CAMS global atmospheric composition forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts)
  (variables and lead times are set in `mirora/src/mirora/setup/download_cams.py`)

**Aurora checkpoint and sparse autoencoders**:

```bash
# Aurora Air Pollution checkpoint (microsoft/aurora)
python3 -c "import sys; sys.path.insert(0,'mirora/scripts'); import consts; consts.ensure_checkpoint()"

# download 48 trained sparse autoencoders (hujason/aurorascope)
python3 mirora/scripts/download_aurorascope_saes.py
```

**FINN fire emissions**: create a free NCAR GDEX
account; the script prompts for an API token on first run and saves it to
`~/.gdex_token`:

- Get a token: https://gdex.ucar.edu/accounts/profile/

```bash
python3 mirora/scripts/download_finn_fire_emissions.py
```

## 4. Fetch and process the data

```bash
python3 mirora/scripts/prepare_figure_data.py --list-inits   # what it will fetch
python3 mirora/scripts/prepare_figure_data.py --all
```

The figures depend on only **5 initialisation times**, not the full validation year, so this downloads 10 CAMS forecasts (each init requires two timesteps).

## 5. Render the figures

```bash
python3 mirora/scripts/reproduce_figures.py --check   # confirm inputs
python3 mirora/scripts/reproduce_figures.py --all
```

Figures are written under `$MIRORA_SCRATCH/figures/`; the NOx-perturbation figures go to
`figures/nox_mmr/` (see the NOx unit convention in [`mirora/scripts/README.md`](mirora/scripts/README.md)).

```bash
reproduce_figures.py --list                        # every figure, tier, GPU need
reproduce_figures.py --figure ozone_perturbation   # just one
reproduce_figures.py --tier 1                      # only the CPU-only figures
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

Steps 1–4 shard their work across parallel tasks and take `<task_id> <task_ct>`
as positional arguments — run them serially with `0 1`, or launch tasks
`0..N-1` concurrently.
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
