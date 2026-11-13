"""
Batch TopK Sparse Autoencoder for Aurora backbone activations.

Instead of L1 regularization, enforces sparsity by keeping only the top-k
activations across the entire batch (BatchTopK). This allows adaptive per-token
sparsity while maintaining a fixed average sparsity level.

Reference: Bussmann et al., "BatchTopK Sparse Autoencoders" (2024)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mirora.config import DEFAULT_TOPK_ALPHA_AUX, DEFAULT_TOPK_DEAD_STEPS, DEFAULT_TOPK_K_DIVISOR


def find_topk_checkpoint(
    sae_dir: Path, component: str, stage_idx: int, block_idx: int, expansion_factor: int = 8,
) -> tuple[Path, int] | None:
    """Find a trained BatchTopK SAE checkpoint (not an epoch snapshot); returns (path, k) or None."""
    pattern = f"sae_topk_{component}_s{stage_idx}_b{block_idx}_x{expansion_factor}_k*.pt"
    matches = sorted(p for p in sae_dir.glob(pattern) if "epoch" not in p.name)
    if not matches:
        return None
    return matches[0], int(matches[0].stem.split("_k")[-1])


class BatchTopKSAE(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        expansion_factor: int = 8,
        k: int | None = None,
        k_aux: int | None = None,
        alpha_aux: float = DEFAULT_TOPK_ALPHA_AUX,
        dead_steps_threshold: int = DEFAULT_TOPK_DEAD_STEPS,
        ema_decay: float = 0.999,
    ):
        """
        Args:
            d_model: Input dimension.
            expansion_factor: Hidden dimension = d_model * expansion_factor.
            k: Number of active features per token. Defaults to d_hidden // DEFAULT_TOPK_K_DIVISOR.
            k_aux: Number of dead features used in auxiliary loss. Defaults to k.
            alpha_aux: Weight for auxiliary dead feature loss.
            dead_steps_threshold: Steps without firing before a feature is considered dead.
            ema_decay: Decay rate for the EMA threshold used at inference.
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * expansion_factor
        self.expansion_factor = expansion_factor
        self.k = k if k is not None else self.d_hidden // DEFAULT_TOPK_K_DIVISOR
        self.k_aux = k_aux if k_aux is not None else self.k
        self.alpha_aux = alpha_aux
        self.dead_steps_threshold = dead_steps_threshold
        self.ema_decay = ema_decay

        # Encoder/Decoder weights (same parameterization as AuroraSAE)
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_model, self.d_hidden)))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().t())
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Dead feature tracking
        self.register_buffer("num_steps_since_fired", torch.zeros(self.d_hidden, dtype=torch.long))
        # EMA threshold for inference
        self.register_buffer("ema_threshold", torch.tensor(0.0))

        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pre-activations (before TopK)."""
        x_cent = x - self.b_dec
        return x_cent @ self.W_enc + self.b_enc

    def _decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def _batch_topk(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Apply BatchTopK: flatten across tokens, select top k*N globally, reshape back."""
        N, D = pre_acts.shape
        total_k = self.k * N

        # Flatten and select top activations
        flat = pre_acts.reshape(-1)
        total_k = min(total_k, flat.shape[0])
        topk_vals, topk_idx = torch.topk(flat, total_k)

        # Create sparse mask
        h = torch.zeros_like(flat)
        h.scatter_(0, topk_idx, topk_vals)
        h = h.reshape(N, D)

        # Apply ReLU to ensure non-negativity
        h = F.relu(h)
        return h

    def _topk_inference(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """At inference, use EMA threshold instead of batch-level TopK."""
        return F.relu(pre_acts - self.ema_threshold)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: Input activations of shape (N, d_model).
            training: If True, use BatchTopK. If False, use EMA threshold.

        Returns:
            x_hat: Reconstructed activations (N, d_model).
            h: Sparse hidden activations (N, d_hidden).
            aux: Dict with auxiliary info: 'aux_loss', 'dead_mask', 'l0'.
        """
        pre_acts = self._encode(x)

        if training:
            h = self._batch_topk(pre_acts)

            # Update EMA threshold
            with torch.no_grad():
                # k-th largest value per token, averaged
                kth_vals, _ = torch.topk(pre_acts, self.k, dim=-1)
                kth_val = kth_vals[:, -1].mean()
                self.ema_threshold.lerp_(kth_val, 1.0 - self.ema_decay)

            # Update dead feature tracking
            with torch.no_grad():
                fired = (h > 0).any(dim=0)
                self.num_steps_since_fired += 1
                self.num_steps_since_fired[fired] = 0
        else:
            h = self._topk_inference(pre_acts)

        x_hat = self._decode(h)

        # Compute auxiliary loss for dead features
        aux: dict[str, torch.Tensor] = {}
        dead_mask = self.num_steps_since_fired > self.dead_steps_threshold
        aux["dead_mask"] = dead_mask
        aux["l0"] = (h > 0).float().sum(dim=-1).mean()
        aux["dead_frac"] = dead_mask.float().mean()

        if training and dead_mask.any() and self.alpha_aux > 0:
            # Reconstruct using top-k_aux dead features
            dead_pre_acts = pre_acts[:, dead_mask]
            k_aux = min(self.k_aux, dead_pre_acts.shape[-1])
            if k_aux > 0:
                topk_dead_vals, topk_dead_idx = torch.topk(dead_pre_acts, k_aux, dim=-1)
                h_dead = torch.zeros_like(dead_pre_acts)
                h_dead.scatter_(-1, topk_dead_idx, F.relu(topk_dead_vals))

                # Reconstruct using only dead features' decoder columns
                dead_indices = dead_mask.nonzero(as_tuple=True)[0]
                W_dec_dead = self.W_dec[dead_indices]
                x_hat_dead = h_dead @ W_dec_dead + self.b_dec
                aux["aux_loss"] = self.alpha_aux * F.mse_loss(x_hat_dead, x)
            else:
                aux["aux_loss"] = torch.tensor(0.0, device=x.device)
        else:
            aux["aux_loss"] = torch.tensor(0.0, device=x.device)

        return x_hat, h, aux
