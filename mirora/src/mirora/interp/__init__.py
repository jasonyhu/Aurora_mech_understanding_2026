"""Interpretability tools: sparse autoencoders and backbone feature analysis."""

from mirora.interp.aurora_sae import AuroraSAE
from mirora.interp.backbone import capture_blocks, get_backbone_block, get_spatial_resolution
from mirora.interp.features import feature_replacement_hook, load_sae, sae_path, var_contribution

__all__ = [
    "AuroraSAE",
    "capture_blocks",
    "feature_replacement_hook",
    "get_backbone_block",
    "get_spatial_resolution",
    "load_sae",
    "sae_path",
    "var_contribution",
]
