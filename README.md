# Toward Mechanistic Interpretability of an AI Foundation Model Fine-Tuned for Atmospheric Chemistry

Code and figure-reproduction pipeline for **Hu et al. (2026)**,
[arXiv:2607.20778](https://arxiv.org/abs/2607.20778).

Jason Y. Hu, Ivan Higuera-Mendieta, Patrick Obin Sturm, Makoto M. Kelp

---

We apply mechanistic interpretability methods to the
[Aurora](https://github.com/microsoft/aurora) earth system foundation model, 
focusing on the checkpoint which has been fine-tuned for atmospheric chemistry 
(Aurora Air Pollution), and we probe its learned chemistry with targeted perturbation experiments.

See **[REPRODUCE.md](REPRODUCE.md)** for instructions to reproduce the figures.

```bash
cd mirora && pip install -e ".[all]" && cd ..
export MIRORA_SCRATCH=/path/to/scratch/folder

python3 mirora/scripts/prepare_figure_data.py --all   # fetch + process public data
python3 mirora/scripts/reproduce_figures.py --check   # confirm inputs
python3 mirora/scripts/reproduce_figures.py --all     # render every figure
```

The library is also importable directly:

```python
from mirora.inference.inference import load_aurora, aap_rollout
from mirora.interp import AuroraSAE, load_sae, capture_blocks, feature_replacement_hook
from mirora.data import load_batch, fetch_rollout, surface
from mirora.experiments import perturb_batch
```

## AuroraScope

AuroraScope, a suite of 48 trained SAEs trained on the Aurora backbone, can be found on
HuggingFace at [**hujason/aurorascope**](https://huggingface.co/hujason/aurorascope).

```bash
python3 mirora/scripts/download_aurorascope_saes.py
```

## Data sources

- [CAMS global atmospheric composition forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts)
- [Aurora Air Pollution checkpoint](https://huggingface.co/microsoft/aurora)
- [FINN v2.5 fire emissions](https://gdex.ucar.edu/) (NCAR GDEX)

## Citation

```bibtex
@misc{hu2026aurora,
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
