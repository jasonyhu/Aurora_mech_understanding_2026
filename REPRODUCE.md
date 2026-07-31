# Reproducing the manuscript figures

All data in the project is public: CAMS global atmospheric-composition forecasts (Copernicus) and the public Aurora Air Pollution checkpoint.

There are two starter scripts to reproduce the figures of the paper:
```bash
# 1. fetch + process the public data the figures need
python3 mirora/scripts/python/prepare_figure_data.py --all

# 2. render every figure
python3 mirora/scripts/python/reproduce_figures.py --all
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
python3 -c "import sys; sys.path.insert(0,'mirora/scripts/python'); import consts; \
from mirora.setup.download_model import download_model_checkpoint as d; d(consts.MODEL_PATH)"

# download 48 trained sparse autoencoders (hujason/aurorascope)
python3 mirora/scripts/python/download_saes.py
```

**FINN fire emissions**: create a free NCAR GDEX
account; the script prompts for an API token on first run and saves it to
`~/.gdex_token`:

- Get a token: https://gdex.ucar.edu/accounts/profile/

```bash
python3 mirora/scripts/python/download_finn.py
```

## 4. Fetch and process the data

```bash
python3 mirora/scripts/python/prepare_figure_data.py --list-inits   # what it will fetch
python3 mirora/scripts/python/prepare_figure_data.py --all
```

The figures depend on only **7 initialisation times**, not the full validation year, so this downloads 14 CAMS forecasts (each init requires two timesteps).

## 5. Render the figures

```bash
python3 mirora/scripts/python/reproduce_figures.py --check   # confirm inputs
python3 mirora/scripts/python/reproduce_figures.py --all
```

Figures are written under `$MIRORA_SCRATCH/figures/`.

```bash
reproduce_figures.py --list                        # every figure, tier, GPU need
reproduce_figures.py --figure ozone_perturbation   # just one
reproduce_figures.py --tier 1                      # only the CPU-only figures
reproduce_figures.py --dry-run --all
```

---

## What needs a GPU

| Tier | Figures | Requirement |
|---|---|---|
| **1** | Figs 1–4, all SI figures/tables/stats | CPU only, once step 4 has run |
| **2** | Figs 6–7 (SAE galleries) | A GPU — these run Aurora live to capture activations |
| **3** | Fig 5 (PCA of backbone embeddings) | The full validation year, see below |

Use GPUs with **>= 32 GB** RAM, and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Figure 5** The PCA panels are computed over backbone embeddings
for every rollout in the validation year, which is ~10.4 GB per initialisation across
730 initialisations (~7.6 TB). `prepare_figure_data.py` ignores this figure. To reproduce that panel, run the full pipeline
(`mirora/scripts/README.md`): download the full CAMS year, build all init
batches, then run `run_base_rollout.py` with backbone activations enabled.

---

## Full pipeline

To rebuild everything from scratch — including SAE training and the embeddings
behind Fig 5 — follow the numbered runbook in
[`mirora/scripts/README.md`](mirora/scripts/README.md):

1. `download_cams.py` — the full CAMS validation year
2. `save_init_batches.py` — Aurora `Batch` objects
3. `run_base_rollout.py` — rollouts + backbone embeddings (the multi-TB step)
4. `train_sae.py` — train the 48 sparse autoencoders (~3 h each on one GPU)
5. `reproduce_figures.py --all`

Steps 1–4 shard their work across parallel tasks and take `<task_id> <task_ct>`
as positional arguments — run them serially with `0 1`, or launch tasks
`0..N-1` concurrently. No cluster job scripts are shipped: resource requests and
scheduler directives are site-specific, so write your own wrappers.

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
