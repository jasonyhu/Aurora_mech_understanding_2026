"""
Area-weighted statistics on Aurora's grid.
"""

import numpy as np
import torch
from scipy.stats import spearmanr


def lat_weights(preds) -> np.ndarray:
    """cos(lat) weight of each grid row [H], from the metadata of a list of Aurora Batches."""
    lat = preds[0].metadata.lat
    if isinstance(lat, torch.Tensor):
        lat = lat.cpu().numpy()
    return np.clip(np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64))), 0.0, None)


def wmean(mask_or_field: np.ndarray, w_row: np.ndarray) -> float:
    """Area-weighted mean over [H, W], weighting each cell by cos(lat) of its row."""
    W2 = np.broadcast_to(w_row[:, None], mask_or_field.shape)
    return float(np.sum(mask_or_field * W2) / np.sum(W2))


def agreement(a: np.ndarray, c: np.ndarray, ok: np.ndarray, w_cells: np.ndarray) -> dict:
    """Pearson r, Spearman rho, RMSE (raw and area-weighted) of a vs c over `ok` cells."""
    a, c, w = a[ok], c[ok], w_cells[ok]
    f = np.isfinite(a) & np.isfinite(c)
    a, c, w = a[f], c[f], w[f]
    return {"pearson": float(np.corrcoef(a, c)[0, 1]),
            "spearman": float(spearmanr(a, c).statistic),
            "rmse": float(np.sqrt(np.mean((a - c) ** 2))),
            "rmse_areaw": float(np.sqrt(np.average((a - c) ** 2, weights=w))),
            "n_cells": int(f.sum())}
