"""
Script to train a sparse autoencoder on Swin3D backbone activations.
"""

import logging
import os
from datetime import datetime
import gc
from pathlib import Path

from aurora import AuroraAirPollution, Batch, Metadata
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mirora.config import STAGES
from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.interp.aurora_sae import AuroraSAE

logger = logging.getLogger(__name__)

def aurora_collate_fn(batch_list: list[Batch]) -> Batch:
    """
    Collate function to merge a list of Aurora Batch objects into a single Batch.
    """
    # collate metadata
    # assume lat, lon, and atmos_levels are identical across Batches, aggregate the 'time' tuples.
    first_meta = batch_list[0].metadata
    
    collated_metadata = Metadata(
        lat=first_meta.lat,
        lon=first_meta.lon,
        atmos_levels=first_meta.atmos_levels,
        time=tuple(t for b in batch_list for t in b.metadata.time),
        rollout_step=first_meta.rollout_step
    )

    # collate surface variables, dim (b, t, h, w)
    surf_keys = batch_list[0].surf_vars.keys()
    collated_surf = {
        k: torch.cat([b.surf_vars[k] for b in batch_list], dim=0) 
        for k in surf_keys
    }

    # collate atmospheric variables, dim (b, t, c, h, w)
    atmos_keys = batch_list[0].atmos_vars.keys()
    collated_atmos = {
        k: torch.cat([b.atmos_vars[k] for b in batch_list], dim=0) 
        for k in atmos_keys
    }

    # static variables remain constant between batches.

    return Batch(
        surf_vars=collated_surf,
        static_vars=batch_list[0].static_vars,
        atmos_vars=collated_atmos,
        metadata=collated_metadata
    )

def train_single_stage(
    model: AuroraAirPollution,
    dataloader: DataLoader,
    target_decoder: bool,  # targets encoder layers otherwise
    stage_idx: int,
    block_idx: int,
    d_model: int,
    device: torch.device,
    num_epochs: int,
    save_dir: Path,
    expansion_factor: int=8,
    l1_coeff: float=3e-4,
    lr: float=4e-4
):
    """
    Trains a single SAE for a specific backbone layer.
    """
    component = "decoder" if target_decoder else "encoder"
    logger.info(f"*** Starting Training for {component} Stage {stage_idx}, Block {block_idx} (Dim: {d_model}, Expansion: {expansion_factor}, L1: {l1_coeff}) ***")

    sae = AuroraSAE(d_model=d_model, expansion_factor=expansion_factor, l1_coeff=l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    captured_activations = []

    def hook_fn(module, input, output):
        """Captures activations from the transformer block."""
        x = output[0]
        feat = x.reshape(-1, x.shape[-1])
        captured_activations.append(feat.detach())

    if target_decoder:
        target_layer = model.backbone.decoder_layers[stage_idx].blocks[block_idx]
    else:
        target_layer = model.backbone.encoder_layers[stage_idx].blocks[block_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            count = 0

            for batch_idx, batch in enumerate(dataloader):
                captured_activations.clear()
                batch = batch.to(device)

                with torch.no_grad():
                    _ = model(batch)

                if not captured_activations:
                    continue

                x = torch.cat(captured_activations, dim=0)

                x_hat, h = sae(x)

                recon_loss = F.mse_loss(x_hat, x)
                l1_loss = h.abs().sum(dim=-1).mean()
                loss = recon_loss + sae.l1_coeff * l1_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sae.normalize_decoder()

                epoch_loss += loss.item()
                count += 1

                if batch_idx % 50 == 0:
                    logger.info(f"{component} stage {stage_idx} block {block_idx} | Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f}")

            avg_loss = epoch_loss / max(count, 1)

            ckpt_path = save_dir / "training_checkpoints" / f"sae_{component}_s{stage_idx}_b{block_idx}_x{expansion_factor}_epoch{epoch}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save(sae.state_dict(), ckpt_path)
            logger.info(f"{component} stage {stage_idx} block {block_idx} | Epoch {epoch} Complete. Avg Loss: {avg_loss:.6f}")

    finally:
        handle.remove()
        final_save_path = save_dir / f"sae_{component}_s{stage_idx}_b{block_idx}_x{expansion_factor}.pt"
        torch.save(sae.state_dict(), final_save_path)
        logger.info(f"Finished {component} stage {stage_idx} block {block_idx}. Saved to {final_save_path}")

        del sae
        del optimizer
        del captured_activations
        torch.cuda.empty_cache()
        gc.collect()

def train(
    aurora_dir: Path,
    batch_inits_dir: Path,
    save_dir: Path,
    start: datetime,
    end: datetime,
    stage_idx: int | None = None,
    expansion_factor: int = 8,
    l1_coeff: float = 3e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 3
    num_workers = len(os.sched_getaffinity(0))

    logger.info("Initializing AuroraInitBatchesDataset...")
    dataset = AuroraInitBatchesDataset(
        save_dir=batch_inits_dir,
        start=start,
        end=end,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=aurora_collate_fn,
        num_workers=num_workers,
    )

    logger.info(f"Loading Aurora from {aurora_dir}")
    model = AuroraAirPollution()
    model.load_checkpoint_local(str(aurora_dir))
    model = model.to(device).eval()

    if stage_idx is not None:
        stages_to_train = [STAGES[stage_idx]]
    else:
        stages_to_train = STAGES

    model_dir = save_dir / "sae"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to {model_dir}")
    logger.info(f"Using {num_workers} DataLoader workers, expansion_factor={expansion_factor}, l1_coeff={l1_coeff}")

    for target_decoder, stage_idx, block_idx, dim in stages_to_train:
        train_single_stage(model, dataloader, target_decoder, stage_idx, block_idx, dim, device, num_epochs, model_dir, expansion_factor, l1_coeff)