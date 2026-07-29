# Reproducing the manuscript figures

Everything here is built from **public data**: CAMS global atmospheric-composition
forecasts (Copernicus) and the public Aurora Air Pollution checkpoint. There is
no private dataset to request.

Two scripts do all the work:

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

Python >= 3.12. The heavy dependency is `microsoft-aurora` (pulls in PyTorch).

## 2. Choose a storage location

No paths are hardcoded. Set one environment variable:

```bash
export MIRORA_SCRATCH=/somewhere/with/space/mirora   # needs ~60 GB
```

On the Stanford Sherlock cluster, `$SCRATCH` and `$OAK` are picked up
automatically — just `source spinup.sh`.

## 3. Credentials for the public data

**Copernicus ADS (CAMS forecasts).** Free account, then put your key in
`~/.cdsapirc`:

- Register: https://ads.atmosphere.copernicus.eu/
- API key instructions: https://ads.atmosphere.copernicus.eu/how-to-api
- Dataset used: [CAMS global atmospheric composition forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts)
  (variables and lead times are set in `mirora/src/mirora/setup/download_cams.py`)

**Aurora checkpoint and sparse autoencoders** (both public on HuggingFace, no key
required):

```bash
# Aurora Air Pollution checkpoint (microsoft/aurora)
python3 -c "import sys; sys.path.insert(0,'mirora/scripts/python'); import consts; \
from mirora.setup.download_model import download_model_checkpoint as d; d(consts.MODEL_PATH)"

# 48 trained sparse autoencoders (hujason/aurorascope)
python3 mirora/scripts/python/download_saes.py
```

**FINN fire emissions** (only needed for the wildfire figure):

```bash
python3 mirora/scripts/python/download_finn.py   # NCAR GDEX, needs ~/.gdex_token
```

## 4. Fetch and process the data

```bash
python3 mirora/scripts/python/prepare_figure_data.py --list-inits   # what it will fetch
python3 mirora/scripts/python/prepare_figure_data.py --all
```

The figures depend on only **7 initialisation times**, not the full validation
year, so this downloads 14 CAMS forecasts (each init also needs `t-12h`) rather
than the whole archive. The stages are resumable and skip anything already
present:

| Stage | What it does | Cost |
|---|---|---|
| `cams` | Download the 14 required CAMS forecasts | ~17 GB; the CDS queue dominates |
| `batches` | Build the Aurora `Batch` init objects | minutes, CPU |
| `rollouts` | Base + NOx-perturbation + single-cell rollouts | ~30 min on one GPU |

Run a single stage with `--steps cams`, and preview with `--dry-run`.

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

Aurora runs out of memory on 11–12 GB cards. Use **>= 32 GB**, and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Why Fig 5 is different.** The PCA panels are computed over backbone embeddings
for *every* rollout in the validation year — ~10.4 GB per initialisation across
730 initialisations (~7.6 TB). `prepare_figure_data.py` deliberately does not
attempt this. To reproduce that panel, run the full pipeline
(`mirora/scripts/README.md`): download the full CAMS year, build all init
batches, then run `run_base_rollout.sh` with backbone activations enabled.

---

## Full pipeline

To rebuild everything from scratch — including SAE training and the embeddings
behind Fig 5 — follow the numbered runbook in
[`mirora/scripts/README.md`](mirora/scripts/README.md):

1. `download_cams.sh` — the full CAMS validation year
2. `save_init_batches.sh` — Aurora `Batch` objects
3. `run_base_rollout.sh` — rollouts + backbone embeddings (the multi-TB step)
4. `train_sae.sh` — train the 48 sparse autoencoders (~3 h each on one GPU)
5. `reproduce_figures.py --all`

These are SLURM array jobs; see
[`mirora/scripts/slurm/README.md`](mirora/scripts/slurm/README.md) for the
one-time `logs/` setup and which directives are cluster-specific.

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
