import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

logger = logging.getLogger(__name__)

def plot_rgb_grid(rgb: np.ndarray, title: str, save_path: Path):
    """
    Plots the [H, W, 3] RGB latent map `rgb` (0..360 longitudes) and saves it to `save_path`.

    If `title` is empty or None, no title is drawn.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))

    rgb = np.roll(rgb, 150, axis=1)
    ax.imshow(rgb, extent=[-180, 180, -90, 90], origin="upper",
              transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(color="white", linewidth=0.5)

    if title:
        plt.title(title)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"Figure saved to {save_path}")
    plt.close()
