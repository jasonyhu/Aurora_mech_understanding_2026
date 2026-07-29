# Toward Mechanistic Interpretability of an AI Foundation Model Fine-Tuned for Atmospheric Chemistry

Code and figure-reproduction pipeline for **Hu et al. (2026)**,
[arXiv:2607.20778](https://arxiv.org/abs/2607.20778).

Jason Y. Hu, Ivan Higuera-Mendieta, Patrick Obin Sturm, Makoto M. Kelp

---

## What this is

We apply mechanistic interpretability methods — sparse autoencoders (SAEs) and
PCA of internal representations — to Microsoft's
[Aurora](https://github.com/microsoft/aurora) foundation model fine-tuned for
atmospheric chemistry (Aurora Air Pollution), and probe its learned chemistry
with targeted perturbation experiments:

- **NOx → ozone titration.** Uniform NOx perturbations (+0.5/+5/+20 ppb) reproduce
  the expected short-term ozone suppression under NOx-saturated conditions.
- **Wildfire case studies.** Five 2022 fire events compared against CAMS, with
  FINN fire detections overlaid.
- **Sparse autoencoders.** 48 SAEs (one per Swin3D backbone block) trained on
  backbone activations, with feature galleries and steering experiments.
- **Representation structure.** PCA of pre-/post-backbone embeddings recovers
  day/night and transport structure.

## Reproducing the figures

**→ [REPRODUCE.md](REPRODUCE.md)** is the full guide. Everything is built from
public data (Copernicus CAMS forecasts + the public Aurora checkpoint); there is
no private dataset to request.

```bash
cd mirora && pip install -e ".[all]" && cd ..
export MIRORA_SCRATCH=/somewhere/with/space/mirora

python3 mirora/scripts/python/prepare_figure_data.py --all   # fetch + process public data
python3 mirora/scripts/python/reproduce_figures.py --check   # confirm inputs
python3 mirora/scripts/python/reproduce_figures.py --all     # render every figure
```

`reproduce_figures.py --list` shows every manuscript figure, which script builds
it, and whether it needs a GPU.

## Layout

| Path | Contents |
|---|---|
| `mirora/src/mirora/` | The installable library: data loading, inference, SAEs, PCA, perturbation experiments, plotting |
| `mirora/scripts/python/` | Entry-point scripts — see [`mirora/scripts/README.md`](mirora/scripts/README.md) |
| `mirora/scripts/slurm/` | SLURM wrappers — see [`mirora/scripts/slurm/README.md`](mirora/scripts/slurm/README.md) |
| `REPRODUCE.md` | Step-by-step figure reproduction |

The library is importable directly:

```python
from mirora.interp import AuroraSAE, BatchTopKSAE
from mirora.data import AuroraInitBatchesDataset
from mirora.experiments import perturb_batch
```

Model/grid constants live in one place (`mirora/src/mirora/config.py`); storage
paths resolve from environment variables (`mirora/scripts/python/consts.py`), so
nothing is hardcoded to a user or cluster.

## Trained sparse autoencoders

The 48 SAEs (8× expansion, one per hookable backbone block) are released on
HuggingFace: [**hujason/aurorascope**](https://huggingface.co/hujason/aurorascope).

```bash
python3 mirora/scripts/python/download_saes.py
```

## Data sources

| Source | Used for | Access |
|---|---|---|
| [CAMS global atmospheric composition forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts) | Model inputs + evaluation baseline | Free Copernicus ADS account |
| [Aurora Air Pollution checkpoint](https://huggingface.co/microsoft/aurora) | The model under study | Public, HuggingFace |
| [FINN v2.5 fire emissions](https://gdex.ucar.edu/) (NCAR GDEX) | Wildfire figure overlays | Free NCAR account |

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

## License

MIT — see [LICENSE](LICENSE).
