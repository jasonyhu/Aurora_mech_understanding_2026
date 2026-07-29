"""
Methods for running inference on the Aurora model.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path

import torch
from aurora import AuroraAirPollution, Batch, rollout

from mirora.utils import get_rollout_dir

logger = logging.getLogger(__name__)

def aap_rollout(
        batch: Batch, 
        batch_date: datetime, 
        aap_checkpoint_path: Path, 
        save_dir: Path, 
        rollout_steps: int = 10, 
        save_backbone_activations: bool = False
    ) -> None:
    model = AuroraAirPollution()
    model.load_checkpoint_local(str(aap_checkpoint_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    pre_backbone_embeddings = []
    post_backbone_embeddings = []
    encoder_hook = None
    decoder_hook = None

    if save_backbone_activations:
        def save_encoder_input(module, args):
            pre_backbone_embeddings.append(args[0].detach().cpu())
            
        def save_decoder_input(module, args, output):
            post_backbone_embeddings.append(output.detach().cpu())
            
        encoder_hook = model.backbone.register_forward_pre_hook(save_encoder_input)
        decoder_hook = model.backbone.register_forward_hook(save_decoder_input)

    with torch.inference_mode():
        predictions = [pred.to("cpu") for pred in rollout(model, batch, steps=rollout_steps)]

    if save_backbone_activations:
        encoder_hook.remove()
        decoder_hook.remove()

    rollout_dir = get_rollout_dir(save_dir, batch_date)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    rollout_path = rollout_dir / "predictions.pickle"

    if save_backbone_activations:
        pre_backbone_path = rollout_dir / "pre_backbone_embeddings.pickle"
        post_backbone_path = rollout_dir / "post_backbone_embeddings.pickle"
        with open(pre_backbone_path, 'wb') as pre_backbone_file:
            pickle.dump(pre_backbone_embeddings, pre_backbone_file)
        with open(post_backbone_path, 'wb') as post_backbone_file:
            pickle.dump(post_backbone_embeddings, post_backbone_file)

    with open(rollout_path, 'wb') as pred_file:
        logger.info(f"Saving to {rollout_path}")
        pickle.dump(predictions, pred_file)
