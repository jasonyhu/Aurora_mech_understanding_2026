"""
Train sparse autoencoders on Swin3D backbone activations.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mirora.config import DEFAULT_EXPANSION_FACTOR, DEFAULT_L1_COEFF, DEFAULT_SAE_LR, STAGES
from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.inference.inference import load_aurora
from mirora.interp.aurora_sae import AuroraSAE
from mirora.interp.backbone import get_backbone_block
from mirora.interp.features import sae_path

logger = logging.getLogger(__name__)

NUM_EPOCHS = 3


def _first(batch_list):
    return batch_list[0]


def train_single_stage(
    model,
    dataloader: DataLoader,
    sae: torch.nn.Module,
    component: str,
    stage_idx: int,
    block_idx: int,
    device: torch.device,
    save_path: Path,
    lr: float = DEFAULT_SAE_LR,
):
    """Train `sae` on one backbone block's outputs; save per-epoch checkpoints, the final
    weights to `save_path`, and a per-batch JSON training log next to it."""
    tag = f"{component} s{stage_idx} b{block_idx}"
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    captured: list[torch.Tensor] = []
    training_log: list[dict] = []

    def hook_fn(module, input, output):
        x = output[0]
        captured.append(x.reshape(-1, x.shape[-1]).detach())

    handle = get_backbone_block(model, component, stage_idx, block_idx).register_forward_hook(hook_fn)
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_loss, count = 0.0, 0
            for batch_idx, batch in enumerate(dataloader):
                captured.clear()
                with torch.no_grad():
                    model(batch.to(device))
                if not captured:
                    continue
                x = torch.cat(captured, dim=0)

                x_hat, h = sae(x)
                recon_loss = F.mse_loss(x_hat, x)
                loss = recon_loss + sae.l1_coeff * h.abs().sum(dim=-1).mean()
                entry = {"l0": (h > 0).float().sum(dim=-1).mean().item()}

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sae.normalize_decoder()

                epoch_loss += loss.item()
                count += 1
                training_log.append({"epoch": epoch, "batch": batch_idx,
                                     "recon_loss": recon_loss.item(), "total_loss": loss.item(), **entry})
                if batch_idx % 50 == 0:
                    logger.info(f"{tag} | Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f}")

            ckpt_path = save_path.parent / "training_checkpoints" / f"{save_path.stem}_epoch{epoch}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save(sae.state_dict(), ckpt_path)
            logger.info(f"{tag} | Epoch {epoch} complete. Avg loss: {epoch_loss / max(count, 1):.6f}")
    finally:
        handle.remove()
        torch.save(sae.state_dict(), save_path)
        with open(save_path.with_name(f"training_log_{save_path.stem}.json"), "w") as f:
            json.dump(training_log, f)
        logger.info(f"Finished {tag}. Saved to {save_path}")


def train(
    aurora_ckpt: Path,
    batch_inits_dir: Path,
    save_dir: Path,
    start: datetime,
    end: datetime,
    stage_idx: int | None = None,
    expansion_factor: int = DEFAULT_EXPANSION_FACTOR,
    l1_coeff: float = DEFAULT_L1_COEFF,
):
    """Train one L1 SAE per backbone block (or only block `stage_idx` of config.STAGES),
    writing checkpoints to `save_dir/sae/`."""
    num_workers = len(os.sched_getaffinity(0))
    dataset = AuroraInitBatchesDataset(save_dir=batch_inits_dir, start=start, end=end)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_first, num_workers=num_workers)

    logger.info(f"Loading Aurora from {aurora_ckpt}")
    model, device = load_aurora(aurora_ckpt)

    model_dir = save_dir / "sae"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"expansion_factor={expansion_factor}, {num_workers} DataLoader workers; "
                f"saving to {model_dir}")

    for component, s_idx, b_idx, dim in ([STAGES[stage_idx]] if stage_idx is not None else STAGES):
        sae = AuroraSAE(d_model=dim, expansion_factor=expansion_factor, l1_coeff=l1_coeff)
        path = sae_path(model_dir, component, s_idx, b_idx, expansion_factor)
        logger.info(f"*** Training {path.name} (d_model={dim}) ***")
        train_single_stage(model, dataloader, sae.to(device), component, s_idx, b_idx, device, path)
