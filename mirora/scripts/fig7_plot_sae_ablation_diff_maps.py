"""
Figure 7: difference maps for the SAE steering rollouts (fig7_run_sae_ablation_rollouts.py).

For each case-study feature (zero-ablation) and species: one percent-change map
and one absolute-change map at the first rollout step (+12h), relative to the
base rollout. Absolute differences are in ppb (mass mixing ratio x M_air / M_species x 1e9).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from mirora.config import MMR_TO_PPB
from mirora.data.loading import fetch_rollout, surface
from mirora.utils import setup_logging
from mirora.viz.maps import draw_global_field, setup_matplotlib_style, symmetric_limit

import consts

logger = logging.getLogger(__name__)

EXP_DIR = consts.EXPERIMENTS_PATH / "sae_steering"
FIGURES_DIR = consts.FIGURES_PATH / "sae_steering"
ABLATIONS = ["zero"]  # mean-ablation rollouts (fig7_run_sae_ablation_rollouts.py --ablations mean) are not plotted

SPECIES = [
    # (short, var, var_label, abs_unit_name, abs_unit_factor)
    ("NO2", "no2", r"NO$_2$", "ppb", MMR_TO_PPB["no2"]),
    ("O3",  "go3", r"O$_3$",  "ppb", MMR_TO_PPB["go3"]),
]


def _plot_diff(field, var_label: str, kind: str, save_path: Path, abs_unit: str = ""):
    """kind in {'pct', 'abs'}. RdBu_r for both (one diverging palette for all differences)."""
    if kind == "pct":
        field = np.clip(field, -200, 200)
        cbar_label = f"\u0394{var_label} (%)"
    else:
        cbar_label = f"\u0394{var_label} ({abs_unit})" if abs_unit else f"\u0394{var_label}"

    # Use the 99th percentile of |diff| as the symmetric colorbar limit, so
    # the top 1% of extreme cells clip to the ends and the bulk of the data
    # gets the full colour range.
    v_limit = symmetric_limit(field, 99)

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    im = draw_global_field(ax, field, "RdBu_r", -v_limit, v_limit, coast_color="black", coast_width=0.5,
                           interpolation="antialiased")
    ax.axis("off")

    # Arrows on the colorbar mark that the top 1% of cells are clipped (saturated)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", aspect=40, fraction=0.05, pad=0.04, extend="both")
    cbar.set_label(cbar_label, fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {save_path.name}")


def process(steering_dir: Path, base):
    metadata = (steering_dir / "metadata.txt").read_text()
    logger.info(f"\n=== {steering_dir.name} ===")
    for line in metadata.strip().split("\n"):
        logger.info(f"  {line}")

    steered = fetch_rollout(steering_dir / "predictions.pickle")

    for short, var, var_label, abs_unit, abs_factor in SPECIES:
        steered_field = surface(steered[0], var)
        base_field = surface(base[0], var)

        abs_diff_native = steered_field - base_field
        abs_diff_display = abs_diff_native * abs_factor

        threshold = float(np.nanmean(np.abs(base_field))) * 0.01
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_pct = np.where(
                np.abs(base_field) > threshold,
                100 * abs_diff_native / base_field,
                np.nan,
            )

        _plot_diff(diff_pct, var_label, "pct",
                   FIGURES_DIR / f"{steering_dir.name}_{short}_pct.pdf")
        _plot_diff(abs_diff_display, var_label, "abs",
                   FIGURES_DIR / f"{steering_dir.name}_{short}_abs.pdf",
                   abs_unit=abs_unit)


def main():
    setup_logging()
    setup_matplotlib_style(22, family="sans-serif")
    if not EXP_DIR.exists():
        logger.error(f"No steering experiments found at {EXP_DIR}. Run fig7_run_sae_ablation_rollouts.py first.")
        sys.exit(1)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dirs = []
    for target in consts.CASE_STUDY_FEATURES:
        for ablation in ABLATIONS:
            d = EXP_DIR / f"{target['name']}_{ablation}"
            if not (d / "predictions.pickle").exists():
                logger.warning(f"Missing rollout: {d} — skipping.")
                continue
            dirs.append(d)

    if not dirs:
        logger.error(f"No active steered rollouts found in {EXP_DIR}.")
        sys.exit(1)

    base = fetch_rollout(consts.base_rollout_path(consts.MAIN_INIT_TIME))
    for d in dirs:
        process(d, base)

    logger.info(f"\nAll {len(dirs) * len(SPECIES) * 2} figures saved under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
