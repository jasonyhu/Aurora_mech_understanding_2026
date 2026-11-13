"""
Flattens any *.pt in the repo into the flat sae/ layout the plotting scripts
expect (sae_{component}_s{stage}_b{block}.pt).
"""

import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

REPO = "hujason/aurorascope"
SAE_DIR = consts.SAE_DIR


def main():
    setup_logging()
    SAE_DIR.mkdir(parents=True, exist_ok=True)
    cache = consts.MODEL_PATH / "sae_hf_download"
    logger.info(f"snapshot_download {REPO} -> {cache}")
    local = Path(snapshot_download(REPO, local_dir=cache, allow_patterns=["*.pt", "*.json"]))

    pts = sorted(local.rglob("*.pt"))
    logger.info(f"Found {len(pts)} .pt files in repo.")
    copied = 0
    for p in pts:
        dest = SAE_DIR / p.name
        if dest.exists():
            continue
        shutil.copy2(p, dest)
        copied += 1
    logger.info(f"Copied {copied} new SAE files into {SAE_DIR} "
                f"(total now {len(list(SAE_DIR.glob('*.pt')))}).")


if __name__ == "__main__":
    main()
