"""
Photochemical diagnostics (Leighton ratio, NOx partitioning) from mass mixing ratios.
"""

import numpy as np

from mirora.config import M_AIR, MOLAR_MASS, SURFACE_LEVEL_IDX
from mirora.data.loading import surface

N_A = 6.02214076e23        # Avogadro's number (mol⁻¹)
R_GAS = 8.314462           # gas constant (J mol⁻¹ K⁻¹)
P_1000HPA = 1000.0 * 100   # Pa


def mmr_to_number_density(mmr, t, species, pressure=P_1000HPA):
    """Mass mixing ratio (kg/kg) -> number density (molecules/cm³).

    n = mmr * (M_air / M_species) * (N_A * pressure) / (R * T), i.e. the mole
    fraction times the air number density, converted from m⁻³ to cm⁻³.
    """
    n = mmr * (M_AIR / MOLAR_MASS[species]) * (N_A * pressure) / (R_GAS * t)
    return n * 1e-6


def nonpositive(no, no2, o3):
    """Cells where any of NO, NO2, O3 is <= 0 (Aurora; CAMS has none)."""
    return (no <= 0) | (no2 <= 0) | (o3 <= 0)


def compute_leighton(no, no2, o3, t):
    """Photostationary-state NO2 photolysis rate J = k3 [NO][O3] / [NO2] (s⁻¹)."""
    k3 = 3e-12 * np.exp(-1500.0 / t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (k3 * mmr_to_number_density(no, t, "no") * mmr_to_number_density(o3, t, "go3")
                / mmr_to_number_density(no2, t, "no2"))


def compute_leighton_masked(no, no2, o3, t):
    """Leighton J, NaN where any species is nonpositive."""
    J = compute_leighton(no, no2, o3, t)
    J[nonpositive(no, no2, o3)] = np.nan
    return J


def compute_nox_ratio(no, no2):
    """NOx partitioning [NO2] / ([NO] + [NO2]), clipped to [0, 1]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(no2 / (no + no2), 0, 1)


def compute_nox_ratio_masked(no, no2):
    """NOx ratio, NaN where NO or NO2 is nonpositive."""
    ratio = compute_nox_ratio(no, no2)
    ratio[(no <= 0) | (no2 <= 0)] = np.nan
    return ratio


def load_cams_fields(ds, fp_idx):
    """NO, NO2, O3, T at 1000 hPa from CAMS forecast_period index fp_idx (+12*fp_idx h)."""
    sel = dict(forecast_period=fp_idx, forecast_reference_time=0, pressure_level=SURFACE_LEVEL_IDX)
    return tuple(ds[v].isel(**sel).values for v in ("no", "no2", "go3", "t"))


def load_aurora_fields(pred):
    """NO, NO2, O3, T at 1000 hPa from one Aurora prediction."""
    return tuple(surface(pred, v) for v in ("no", "no2", "go3", "t"))
