"""Model loading for embedding extraction."""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the default device (CPU)."""

    if torch.cuda.is_available():
        logger.info("CUDA is available, but CPU is used by default.")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_embedding_model() -> torch.nn.Module:
    """Load a ResNet50 model configured for embeddings."""

    device = get_device()
    logger.info("Loading ResNet50 model on %s", device)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model
