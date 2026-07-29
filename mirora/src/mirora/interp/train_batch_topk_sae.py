"""
Training loop for Batch TopK sparse autoencoders on Swin3D backbone activations.
"""

import json
import logging
import os
import gc
from datetime import datetime
from pathlib import Path

from aurora import AuroraAirPollution
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.interp.batch_topk_sae import BatchTopKSAE
from mirora.interp.train_sae import aurora_collate_fn
from mirora.config import STAGES

logger = logging.getLogger(__name__)


def train_single_stage_topk(
    model: AuroraAirPollution,
    dataloader: DataLoader,
    target_decoder: bool,
    stage_idx: int,
    block_idx: int,
    d_model: int,
    device: torch.device,
    num_epochs: int,
    save_dir: Path,
    expansion_factor: int = 8,
    k: int | None = None,
    k_aux: int | None = None,
    alpha_aux: float = 1 / 32,
    lr: float = 4e-4,
):
    component = "decoder" if target_decoder else "encoder"
    sae = BatchTopKSAE(
        d_model=d_model,
        expansion_factor=expansion_factor,
        k=k,
        k_aux=k_aux,
        alpha_aux=alpha_aux,
    ).to(device)

    actual_k = sae.k
    logger.info(
        f"*** Starting BatchTopK Training for {component} Stage {stage_idx}, "
        f"Block {block_idx} (Dim: {d_model}, Expansion: {expansion_factor}, k: {actual_k}) ***"
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    captured_activations: list[torch.Tensor] = []
    training_log: list[dict] = []

    def hook_fn(module, input, output):
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
            epoch_recon = 0.0
            epoch_aux = 0.0
            epoch_l0 = 0.0
            epoch_dead = 0.0
            count = 0

            for batch_idx, batch in enumerate(dataloader):
                captured_activations.clear()
                batch = batch.to(device)

                with torch.no_grad():
                    _ = model(batch)

                if not captured_activations:
                    continue

                x = torch.cat(captured_activations, dim=0)
                x_hat, h, aux = sae(x, training=True)

                recon_loss = F.mse_loss(x_hat, x)
                aux_loss = aux["aux_loss"]
                loss = recon_loss + aux_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sae.normalize_decoder()

                l0 = aux["l0"].item()
                dead_frac = aux["dead_frac"].item()

                epoch_recon += recon_loss.item()
                epoch_aux += aux_loss.item()
                epoch_l0 += l0
                epoch_dead += dead_frac
                count += 1

                training_log.append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "recon_loss": recon_loss.item(),
                    "aux_loss": aux_loss.item(),
                    "l0": l0,
                    "dead_frac": dead_frac,
                })

                if batch_idx % 50 == 0:
                    logger.info(
                        f"{component} s{stage_idx} b{block_idx} | Epoch {epoch} | Batch {batch_idx} | "
                        f"Recon: {recon_loss.item():.6f} | Aux: {aux_loss.item():.6f} | "
                        f"L0: {l0:.1f} | Dead: {dead_frac:.3f}"
                    )

            avg_recon = epoch_recon / max(count, 1)
            avg_l0 = epoch_l0 / max(count, 1)
            avg_dead = epoch_dead / max(count, 1)

            ckpt_path = (
                save_dir / "training_checkpoints"
                / f"sae_topk_{component}_s{stage_idx}_b{block_idx}_x{expansion_factor}_k{actual_k}_epoch{epoch}.pt"
            )
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save(sae.state_dict(), ckpt_path)
            logger.info(
                f"{component} s{stage_idx} b{block_idx} | Epoch {epoch} Complete. "
                f"Avg Recon: {avg_recon:.6f} | Avg L0: {avg_l0:.1f} | Avg Dead: {avg_dead:.3f}"
            )

    finally:
        handle.remove()
        final_path = save_dir / f"sae_topk_{component}_s{stage_idx}_b{block_idx}_x{expansion_factor}_k{actual_k}.pt"
        torch.save(sae.state_dict(), final_path)
        logger.info(f"Finished {component} s{stage_idx} b{block_idx}. Saved to {final_path}")

        # Save training log
        log_path = save_dir / f"training_log_topk_{component}_s{stage_idx}_b{block_idx}.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f)

        del sae, optimizer, captured_activations
        torch.cuda.empty_cache()
        gc.collect()


def train_topk(
    aurora_dir: Path,
    batch_inits_dir: Path,
    save_dir: Path,
    start: datetime,
    end: datetime,
    stage_idx: int | None = None,
    expansion_factor: int = 8,
    k: int | None = None,
    k_aux: int | None = None,
    alpha_aux: float = 1 / 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 3
    num_workers = len(os.sched_getaffinity(0))

    logger.info("Initializing AuroraInitBatchesDataset...")
    dataset = AuroraInitBatchesDataset(save_dir=batch_inits_dir, start=start, end=end)

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=aurora_collate_fn, num_workers=num_workers,
    )

    logger.info(f"Loading Aurora from {aurora_dir}")
    model = AuroraAirPollution()
    model.load_checkpoint_local(str(aurora_dir))
    model = model.to(device).eval()

    if stage_idx is not None:
        stages_to_train = [STAGES[stage_idx]]
    else:
        stages_to_train = STAGES

    model_dir = save_dir / "sae_topk"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints to {model_dir}")
    logger.info(f"Using {num_workers} DataLoader workers, expansion_factor={expansion_factor}, k={k}")

    for target_decoder, s_idx, b_idx, dim in stages_to_train:
        train_single_stage_topk(
            model, dataloader, target_decoder, s_idx, b_idx, dim,
            device, num_epochs, model_dir,
            expansion_factor=expansion_factor, k=k, k_aux=k_aux, alpha_aux=alpha_aux,
        )
