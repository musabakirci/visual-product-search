"""Heatmap overlay utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import cm


def overlay_heatmap_on_image(
    image_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a heatmap on top of an image."""

    path = Path(image_path)
    with Image.open(path) as image:
        base = image.convert("RGB")

    heatmap_resized = heatmap
    if heatmap_resized.shape[:2] != (base.height, base.width):
        heatmap_image = Image.fromarray(
            np.uint8(np.clip(heatmap_resized, 0, 1) * 255)
        ).resize((base.width, base.height), resample=Image.BILINEAR)
        heatmap_resized = np.array(heatmap_image) / 255.0

    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap_resized)
    heatmap_rgb = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))

    blended = Image.blend(base, heatmap_rgb, alpha=alpha)
    return blended
