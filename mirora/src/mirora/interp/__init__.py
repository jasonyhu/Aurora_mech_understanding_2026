"""Interpretability tools: sparse autoencoders, PCA, feature analysis."""

from mirora.interp.aurora_sae import AuroraSAE
from mirora.interp.batch_topk_sae import BatchTopKSAE

__all__ = [
    "AuroraSAE",
    "BatchTopKSAE",
    "train_l1_sae",
    "train_batch_topk",
    "run_global_pca",
]


def __getattr__(name: str):
    # Expose the training / PCA entry points lazily so that ``import mirora.interp``
    # (and the common ``from mirora.interp.aurora_sae import AuroraSAE``) stays cheap
    # and does not eagerly import the full Aurora model or scikit-learn.
    if name == "train_l1_sae":
        from mirora.interp.train_sae import train
        return train
    if name == "train_batch_topk":
        from mirora.interp.train_batch_topk_sae import train_topk
        return train_topk
    if name == "run_global_pca":
        from mirora.interp.embedding_pca import run_global_pca
        return run_global_pca
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
