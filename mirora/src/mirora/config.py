"""
Centralized configuration for the Aurora Air Pollution model backbone,
variable mappings, and grid geometry.
"""

# ==========================================
# Swin3D Backbone Architecture
# ==========================================

# (num_blocks, embedding_dim) per stage
ENCODER_CONFIG = [(6, 512), (10, 1024), (8, 2048)]
DECODER_CONFIG = [(8, 2048), (10, 1024), (6, 512)]

# All 48 hookable blocks: (component, stage_idx, block_idx, d_model).
# Indices 0-23: encoder, 24-47: decoder.
STAGES: list[tuple[str, int, int, int]] = [
    (component, stage_idx, block_idx, dim)
    for component, config in (("encoder", ENCODER_CONFIG), ("decoder", DECODER_CONFIG))
    for stage_idx, (n_blocks, dim) in enumerate(config)
    for block_idx in range(n_blocks)
]

# Grid geometry
PATCH_SIZE = 3
LATENT_LEVELS = 4
GRID_H, GRID_W = 451, 900            # raw CAMS grid
PATCH_H, PATCH_W = 150, 300          # after patching
SPATIAL_DIM = PATCH_H * PATCH_W      # 45,000 tokens per latent level
TOTAL_TOKENS = LATENT_LEVELS * SPATIAL_DIM  # 180,000

# Backbone embedding dimensions
PRE_BACKBONE_DIM = 512
POST_BACKBONE_DIM = 1024  # after concatenative skip at decoder stage 2

# Swin3D window size in patch space
WINDOW_SIZE = (2, 6, 12)  # (Wc, Wh, Ww)

# ==========================================
# Variable Mappings
# ==========================================

CAMS_TO_AAP_SURF_VARS = {
    "u10":   "10u",
    "v10":   "10v",
    "t2m":   "2t",
    "msl":   "msl",
    "pm1":   "pm1",
    "pm2p5": "pm2p5",
    "pm10":  "pm10",
    "tcco":  "tcco",
    "tc_no": "tc_no",
    "tcno2": "tcno2",
    "gtco3": "gtco3",
    "tcso2": "tcso2",
}

CAMS_TO_AAP_ATMOS_VARS = {
    "t":   "t",
    "u":   "u",
    "v":   "v",
    "q":   "q",
    "z":   "z",
    "co":  "co",
    "no":  "no",
    "no2": "no2",
    "go3": "go3",
    "so2": "so2",
}

ATMOS_VAR_DESCRIPTIONS = {
    "t":   "Temperature",
    "u":   "Eastward wind",
    "v":   "Northward wind",
    "q":   "Specific humidity",
    "z":   "Geopotential height",
    "co":  "CO",
    "no":  "NO",
    "no2": r"NO$_2$",
    "go3": r"O$_3$",
    "so2": r"SO$_2$",
}

SURF_VAR_DESCRIPTIONS = {
    "10u":   "Eastward 10m wind",
    "10v":   "Northward 10m wind",
    "2t":    "2m temperature",
    "msl":   "Mean sea level pressure",
    "pm1":   r"PM$_1$",
    "pm2p5": r"PM$_{2.5}$",
    "pm10":  r"PM$_{10}$",
    "tcco":  "Total column CO",
    "tc_no": "Total column NO",
    "tcno2": r"Total column NO$_2$",
    "gtco3": "Total column ozone",
    "tcso2": r"Total column SO$_2$",
}

CHEM_ATMOS_VARS = {
    "co":  "CO",
    "no":  "NO",
    "no2": r"NO$_2$",
    "go3": r"O$_3$",
    "so2": r"SO$_2$",
}

# 13 pressure levels in hPa
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_LEVEL_IDX = 0  # 1000 hPa

# Molar masses (g/mol) and mass mixing ratio (kg/kg) -> ppb (by volume) factors
M_AIR = 28.97
MOLAR_MASS = {"co": 28.01, "no": 30.006, "no2": 46.0055, "go3": 48.0, "so2": 64.066}
MMR_TO_PPB = {sp: M_AIR / m * 1e9 for sp, m in MOLAR_MASS.items()}

SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

# ==========================================
# SAE Defaults
# ==========================================

DEFAULT_EXPANSION_FACTOR = 8
DEFAULT_L1_COEFF = 3e-4
DEFAULT_SAE_LR = 4e-4

# ==========================================
# BatchTopK SAE Defaults
# ==========================================

DEFAULT_TOPK_K_DIVISOR = 32  # k = d_hidden // this value (~3% active)
DEFAULT_TOPK_ALPHA_AUX = 1 / 32
DEFAULT_TOPK_DEAD_STEPS = 5000
