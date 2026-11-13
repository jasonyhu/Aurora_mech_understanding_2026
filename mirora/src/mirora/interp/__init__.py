"""Interpretability tools: sparse autoencoders and backbone feature analysis."""

from mirora.interp.aurora_sae import AuroraSAE
from mirora.interp.batch_topk_sae import BatchTopKSAE, find_topk_checkpoint
from mirora.interp.backbone import capture_blocks, get_backbone_block, get_spatial_resolution
from mirora.interp.features import feature_replacement_hook, load_sae, sae_path, var_contribution

__all__ = [
    "AuroraSAE",
    "BatchTopKSAE",
    "capture_blocks",
    "feature_replacement_hook",
    "find_topk_checkpoint",
    "get_backbone_block",
    "get_spatial_resolution",
    "load_sae",
    "sae_path",
    "var_contribution",
]
