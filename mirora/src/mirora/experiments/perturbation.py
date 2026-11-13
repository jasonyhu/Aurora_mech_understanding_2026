"""
Framework for running perturbation experiments on Aurora.

Provides functions to perturb Aurora Batch objects in-memory.
"""

from aurora import Batch

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
