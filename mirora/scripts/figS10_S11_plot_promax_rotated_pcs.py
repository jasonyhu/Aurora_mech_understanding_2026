"""
Figures S10 (encoder latent levels) and S11 (decoder latent levels): oblique
(promax) rotation of the leading principal components at each latent level,
relaxing the orthogonality constraint of PCA.

Reads the 10-component annual fit written by
`fig5_figS8_S9_tableS2_run_latent_pca.py --n-components 10` (pca_embeddings/n10/yearly/
pca_loadings_{stage}.npz), rotates the loadings L = V * sqrt(lambda) with
promax, and maps the rotated component scores of the Sep 1 2022
day (12 UTC) embedding.

Outputs under figures/pca_embeddings/n10/rotated/:
  {stage}_promax_all_levels.{pdf,png}  2x5 maps of the 10 rotated components per level
  promax_{stage}.npz            pattern loadings, rotation matrix, factor correlations
  promax_summary.csv            per level: max |factor correlation|, rows' loading simplicity
"""

import logging
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirora.config import LATENT_LEVELS, PATCH_H, PATCH_W
from mirora.data.loading import load_embedding
from mirora.utils import get_rollout_dir, setup_logging
from mirora.viz.maps import draw_global_field, symmetric_limit

import consts

logger = logging.getLogger(__name__)

FIT_DIR = consts.FIGURES_PATH / "pca_embeddings" / "n10" / "yearly"
OUT_DIR = consts.FIGURES_PATH / "pca_embeddings" / "n10" / "rotated"
MAP_INIT = datetime(2022, 9, 1, 12)  # Sep 1 2022 12 UTC (day)


def varimax(L: np.ndarray, tol: float = 1e-6, max_iter: int = 500) -> np.ndarray:
    """Kaiser-normalized varimax; returns the orthogonal rotation matrix R (k x k)."""
    h = np.sqrt((L ** 2).sum(axis=1, keepdims=True))
    A = L / h
    p, k = A.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        B = A @ R
        u, s, vt = np.linalg.svd(A.T @ (B ** 3 - B @ np.diag((B ** 2).sum(axis=0)) / p))
        R = u @ vt
        if s.sum() < d * (1 + tol):
            break
        d = s.sum()
    return R


def promax(L: np.ndarray, m: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promax rotation. Returns (pattern P = L @ T, rotation T, factor correlations Phi)."""
    Rv = varimax(L)
    V = L @ Rv
    Q = V * np.abs(V) ** (m - 1)
    U = np.linalg.lstsq(V, Q, rcond=None)[0]
    U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))[None, :]
    T = Rv @ U
    Phi = np.linalg.inv(U.T @ U)
    return L @ T, T, Phi


def rotated_scores(x: np.ndarray, mean: np.ndarray, components: np.ndarray, ev: np.ndarray, T: np.ndarray):
    """Standardized PC scores Z = (x - mean) V / sqrt(lambda), rotated as F = Z T^{-T}."""
    Z = (x - mean) @ components.T / np.sqrt(ev)
    return Z @ np.linalg.inv(T).T


def plot_stage(level_scores: list[np.ndarray], title: str, save_path):
    """One figure per stage: a 2x5 block of rotated PCs for each latent level, stacked vertically."""
    fig = plt.figure(figsize=(25, 7.6 * len(level_scores) + 1))
    subfigs = fig.subfigures(len(level_scores), 1, hspace=0.04)
    for level, (sub, scores) in enumerate(zip(subfigs, level_scores)):
        sub.suptitle(f"Level {level}", fontsize=24, fontweight="bold")
        axs = sub.subplots(2, 5, subplot_kw={"projection": ccrs.PlateCarree()})
        sub.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04, hspace=0.18)
        for i, ax in enumerate(axs.flat):
            field = scores[:, i].reshape(PATCH_H, PATCH_W)
            v = symmetric_limit(field, 99)
            draw_global_field(ax, field, "RdBu_r", -v, v, coast_width=0.5, interpolation="antialiased")
            ax.set_title(f"Rotated PC {i + 1}", fontsize=20)
    fig.suptitle(title, fontsize=30, y=1.01)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def main():
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    map_dir = get_rollout_dir(consts.EXPERIMENTS_PATH / "base", MAP_INIT)
    rows = []
    for stage in ("pre_backbone", "post_backbone"):
        fit = np.load(FIT_DIR / f"pca_loadings_{stage}.npz")
        emb = load_embedding(map_dir, stage)  # (levels, spatial, D)
        saved, level_scores = {}, []
        for level in range(LATENT_LEVELS):
            V, ev, mean = fit["components"][level], fit["explained_variance"][level], fit["mean"][level]
            L = V.T * np.sqrt(ev)
            P, T, Phi = promax(L)
            saved.update({f"pattern_l{level}": P, f"rotation_l{level}": T, f"phi_l{level}": Phi})
            off = np.abs(Phi[~np.eye(len(Phi), dtype=bool)])
            # Loading simplicity: share of each variable's squared loading on its largest component
            simp = float(((P ** 2).max(axis=1) / (P ** 2).sum(axis=1)).mean())
            simp_unrot = float(((L ** 2).max(axis=1) / (L ** 2).sum(axis=1)).mean())
            rows.append({"stage": stage, "level": level, "max_abs_factor_corr": off.max(),
                         "mean_abs_factor_corr": off.mean(), "simplicity_promax": simp,
                         "simplicity_unrotated": simp_unrot})
            if emb is not None:
                level_scores.append(rotated_scores(emb[level], mean, V, ev, T))
            logger.info(f"  {stage} level {level}: max|phi|={off.max():.3f}, simplicity {simp_unrot:.3f} -> {simp:.3f}")
        np.savez(OUT_DIR / f"promax_{stage}.npz", **saved)
        if level_scores:
            plot_stage(level_scores, f"Promax-rotated PCA on {stage.replace('_', '-')} embeddings",
                       OUT_DIR / f"{stage}_promax_all_levels.pdf")
    pd.DataFrame(rows).to_csv(OUT_DIR / "promax_summary.csv", index=False)
    logger.info(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
