from pathlib import Path
import logging

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

def download_model_checkpoint(save_dir: Path) -> None:
    repo = "microsoft/aurora"
    checkpoint = "aurora-0.4-air-pollution.ckpt"
    static_vars = "aurora-0.4-air-pollution-static.pickle"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoints {checkpoint}, {static_vars} from repo {repo} to directory {save_dir}.")
    hf_hub_download(repo, checkpoint, local_dir=save_dir)
    hf_hub_download(repo, static_vars, local_dir=save_dir)
