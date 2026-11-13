"""
Loading trained L1 SAEs and reading or editing their features inside the backbone.
"""

from pathlib import Path

import torch

from mirora.config import DEFAULT_EXPANSION_FACTOR
from mirora.interp.aurora_sae import AuroraSAE


def sae_path(sae_dir: Path, component: str, stage_idx: int, block_idx: int,
             expansion_factor: int = DEFAULT_EXPANSION_FACTOR) -> Path:
    """Checkpoint path of an L1 SAE; the default 8x expansion has no `_x` suffix."""
    suffix = "" if expansion_factor == DEFAULT_EXPANSION_FACTOR else f"_x{expansion_factor}"
    return sae_dir / f"sae_{component}_s{stage_idx}_b{block_idx}{suffix}.pt"


def load_sae(sae_dir: Path, component: str, stage_idx: int, block_idx: int, d_model: int,
             device: torch.device, expansion_factor: int = DEFAULT_EXPANSION_FACTOR) -> AuroraSAE:
    """Load a trained L1 SAE in eval mode on `device`."""
    sae = AuroraSAE(d_model=d_model, expansion_factor=expansion_factor)
    path = sae_path(sae_dir, component, stage_idx, block_idx, expansion_factor)
    sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return sae.to(device).eval()


def var_contribution(sae: AuroraSAE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token feature activations h [N, d_hidden] and each feature's share of the
    token's centred variance, h^2 / ||x - b_dec||^2 (W_dec rows are unit-norm)."""
    h = sae.encode(x)
    token_var = ((x - sae.b_dec) ** 2).sum(dim=-1, keepdim=True)
    return h, h ** 2 / (token_var + 1e-12)


def feature_replacement_hook(sae: AuroraSAE, feat_idx: int, value: float = 0.0):
    """Forward hook that sets feature `feat_idx` to `value` on every token of a block's
    output: x <- x - h_k W_dec[k] + value W_dec[k]. value=0 is zero-ablation."""
    W_dec_row = sae.W_dec[feat_idx:feat_idx + 1, :]

    def hook_fn(_m, _i, output):
        x, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, ())
        flat = x.reshape(-1, x.shape[-1])
        h_k = sae.encode(flat)[:, feat_idx:feat_idx + 1]
        new_contrib = torch.full_like(h_k, value) @ W_dec_row
        steered = (flat - h_k @ W_dec_row + new_contrib).reshape(x.shape)
        return (steered, *rest) if rest else steered

    return hook_fn
