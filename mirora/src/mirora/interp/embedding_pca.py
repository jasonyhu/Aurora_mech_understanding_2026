"""
Run PCA visualizations on pre-backbone and post-backbone embeddings.
"""
import logging
import pickle
from pathlib import Path
from tqdm import tqdm

import sklearn
import numpy as np
from sklearn.decomposition import IncrementalPCA

from mirora.config import LATENT_LEVELS, SPATIAL_DIM, PATCH_H, PATCH_W

logger = logging.getLogger(__name__)

def run_global_pca(
    embedding_paths: list[Path], 
    embedding_dim: int,
):
    sklearn.set_config(assume_finite=True)
    
    num_levels = LATENT_LEVELS
    spatial_dim = SPATIAL_DIM  # PATCH_H * PATCH_W per latent level
    
    ipcas = [IncrementalPCA(n_components=3) for _ in range(num_levels)]    
    
    noon_sum = np.zeros((num_levels, spatial_dim, embedding_dim), dtype=np.float64)
    midnight_sum = np.zeros((num_levels, spatial_dim, embedding_dim), dtype=np.float64)

    noon_count = 0
    midnight_count = 0

    logger.info("Fitting IncrementalPCAs and accumulating sums per level...")
    for embedding_path in tqdm(embedding_paths):
        if not embedding_path.exists():
            logger.warning(f"Embeddings not found at {str(embedding_path)}!")
            continue

        with open(embedding_path, "rb") as f:
            # Load and reshape to separate the latent levels
            # Shape: (num_levels, spatial_dim, embedding_dim)
            embedding_3d = pickle.load(f)[0].squeeze().cpu().numpy().reshape(num_levels, spatial_dim, embedding_dim)
    
        # Fit each level's PCA on its respective slice
        for level in range(num_levels):
            ipcas[level].partial_fit(embedding_3d[level])

        if "_1200" in str(embedding_path):
            noon_sum += embedding_3d
            noon_count += 1
        elif "_0000" in str(embedding_path):
            midnight_sum += embedding_3d
            midnight_count += 1
        else:
            raise OSError("Ending not found for file")

    logger.info(f"Found {noon_count} noon rollouts and {midnight_count} midnight rollouts.")
    logger.info("Running PCA on averages and normalizing colors...")
    
    avg_noon = (noon_sum / max(1, noon_count)).astype(np.float32)
    avg_midnight = (midnight_sum / max(1, midnight_count)).astype(np.float32)

    noon_rgb_levels = []
    midnight_rgb_levels = []

    for level in range(num_levels):
        noon_pcs = ipcas[level].transform(avg_noon[level])
        midnight_pcs = ipcas[level].transform(avg_midnight[level])

        level_min = np.min([noon_pcs.min(axis=0), midnight_pcs.min(axis=0)], axis=0)
        level_max = np.max([noon_pcs.max(axis=0), midnight_pcs.max(axis=0)], axis=0)
        
        noon_rgb = (noon_pcs - level_min) / (level_max - level_min + 1e-8)
        midnight_rgb = (midnight_pcs - level_min) / (level_max - level_min + 1e-8)

        noon_rgb_levels.append(noon_rgb.reshape(PATCH_H, PATCH_W, 3))
        midnight_rgb_levels.append(midnight_rgb.reshape(PATCH_H, PATCH_W, 3))

    return noon_rgb_levels, midnight_rgb_levels
