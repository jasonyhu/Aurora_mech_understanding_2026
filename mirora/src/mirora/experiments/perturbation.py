"""
Framework for running perturbation experiments on Aurora.

Provides functions to perturb Aurora Batch objects in-memory and generate
comparison plots (rollout, absolute diff, percent diff) against a baseline.

The map-grid plotting primitives live in :mod:`mirora.viz.maps`.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from aurora import Batch

from mirora.data.loading import fetch_rollout
from mirora.config import CHEM_ATMOS_VARS
from mirora.viz.maps import (
    plot_rollout_grid,
    plot_abs_diff_grid,
    plot_pct_diff_grid,
)

logger = logging.getLogger(__name__)

G = 9.80665  # standard gravity (m/s²)


def perturb_batch(
    batch: Batch,
    atmos_vars: dict[str, float],
    surf_vars: dict[str, float] | None = None,
) -> Batch:
    """Apply perturbations to an Aurora Batch in-place.

    Args:
        batch: Aurora Batch object.
        atmos_vars: {var_name: perturbation_value} for atmospheric variables (kg/kg).
        surf_vars: {var_name: perturbation_value} for surface variables (kg/m²).
            If None, total column variables corresponding to the atmospheric
            perturbations are estimated automatically.

    Returns:
        The modified batch (same object, mutated in-place).
    """
    for var, value in atmos_vars.items():
        if var in batch.atmos_vars:
            batch.atmos_vars[var] = batch.atmos_vars[var] + value

    if surf_vars is not None:
        for var, value in surf_vars.items():
            if var in batch.surf_vars:
                batch.surf_vars[var] = batch.surf_vars[var] + value
    else:
        # Auto-estimate total column perturbation for NOx
        tc_mapping = {"no": "tc_no", "no2": "tcno2"}
        for atmos_var, tc_var in tc_mapping.items():
            if atmos_var in atmos_vars and tc_var in batch.surf_vars:
                delta_tc = atmos_vars[atmos_var] * 100000 / G
                batch.surf_vars[tc_var] = batch.surf_vars[tc_var] + delta_tc

    return batch


def generate_comparison_plots(
    init_time: datetime,
    exp_rollout_path: Path,
    base_rollout_path: Path,
    plots_dir: Path,
    exp_label: str,
    variables: dict[str, str] | None = None,
    levels: dict[int, str] | None = None,
):
    """Generate rollout, absolute diff, and percent diff plots for all variables/levels.

    Args:
        init_time: Initialization datetime.
        exp_rollout_path: Path to experiment predictions.pickle.
        base_rollout_path: Path to base predictions.pickle.
        plots_dir: Directory to save plots.
        exp_label: Experiment label for titles (e.g. '+20ppb NOx').
        variables: {var_key: var_label} dict. Defaults to 5 chemistry species.
        levels: {level_idx: level_name} dict. Defaults to {0: '1000 hPa', 7: '500 hPa'}.
    """
    if variables is None:
        variables = CHEM_ATMOS_VARS
    if levels is None:
        levels = {0: "1000 hPa", 7: "500 hPa"}

    plots_dir.mkdir(parents=True, exist_ok=True)
    date_str = init_time.strftime("%Y-%m-%d %H:%MZ")

    exp_rollout = fetch_rollout(exp_rollout_path)
    base_rollout = fetch_rollout(base_rollout_path)
    times = [step.metadata.time[0] for step in base_rollout]

    for var_key, var_label in variables.items():
        for level_idx, level_name in levels.items():
            label = f"{var_label} at {level_name}"
            suffix = f"{var_key}_L{level_idx}"

            exp_data = [s.atmos_vars[var_key][0, 0, level_idx].numpy() for s in exp_rollout]
            base_data = [s.atmos_vars[var_key][0, 0, level_idx].numpy() for s in base_rollout]

            diffs = [e - b for e, b in zip(exp_data, base_data)]
            base_mean = np.mean(np.abs(np.stack(base_data)))
            threshold = base_mean * 0.01
            with np.errstate(divide="ignore", invalid="ignore"):
                diff_pcts = [
                    (d / np.where(np.abs(b) > threshold, b, np.nan)) * 100
                    for d, b in zip(diffs, base_data)
                ]

            plot_rollout_grid(
                exp_data, times,
                f"{label} (exp: {exp_label} | date: {date_str})",
                plots_dir / f"{suffix}_rollout.png",
                cbar_label=f"{label} concentration",
            )
            plot_abs_diff_grid(
                diffs, times,
                f"\u0394{label} absolute (exp: {exp_label} | date: {date_str})",
                plots_dir / f"{suffix}_diff_abs.png",
                cbar_label=f"\u0394{label} (experiment \u2212 base)",
            )
            plot_pct_diff_grid(
                diff_pcts, times,
                f"\u0394{label}% (exp: {exp_label} | date: {date_str})",
                plots_dir / f"{suffix}_diff_pct.png",
                cbar_label=f"\u0394{label} (%)",
            )

    logger.info(f"Plots saved to {plots_dir}")
