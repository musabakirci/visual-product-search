"""Grad-CAM implementation for ResNet50."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """Minimal Grad-CAM helper for a target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        if hasattr(target_layer, "register_full_backward_hook"):
            self._backward_handle = target_layer.register_full_backward_hook(
                self._backward_hook
            )
        else:
            self._backward_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        """Remove registered hooks."""

        self._forward_handle.remove()
        self._backward_handle.remove()

    def compute(self, input_size: tuple[int, int]) -> torch.Tensor:
        """Compute the Grad-CAM heatmap."""

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM requires both activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.remove()


def generate_gradcam(
    image_path: str,
    model: torch.nn.Module,
    preprocess_fn: Callable[[str], torch.Tensor],
    device: str = "cpu",
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for an image."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(path) as image:
        original_size = (image.height, image.width)

    model.eval()
    model.to(device)

    input_tensor = preprocess_fn(str(path)).to(device)
    input_tensor.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    with GradCAM(model, model.layer4) as cam:
        embedding = model(input_tensor)
        embedding = embedding.view(embedding.size(0), -1)
        embedding_norm = embedding / (embedding.norm(p=2, dim=1, keepdim=True) + 1e-8)
        score = (embedding_norm * embedding_norm.detach()).sum()
        score.backward()
        heatmap = cam.compute(input_tensor.shape[-2:])

    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )
    heatmap = heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    logger.info("Generated Grad-CAM heatmap for %s", path)
    return heatmap
