"""CLI demo for Grad-CAM explanations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embedding.model import get_embedding_model
from src.embedding.preprocess import preprocess_image
from src.explain.cache import get_cache_key, load_cached_explanation, save_explanation_image
from src.explain.gradcam import generate_gradcam
from src.explain.overlay import overlay_heatmap_on_image
from src.services.logging_service import configure_logging

logger = configure_logging()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate a Grad-CAM explanation.")
    parser.add_argument("image_path", help="Path to an image.")
    return parser.parse_args()


def main() -> None:
    """Generate and cache a Grad-CAM overlay."""

    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    model = get_embedding_model()
    cache_key = get_cache_key(str(image_path), "resnet50")
    cached = load_cached_explanation(cache_key)
    if cached:
        logger.info("Using cached explanation: %s", cached)
        print(str(cached))
        return

    heatmap = generate_gradcam(
        image_path=str(image_path),
        model=model,
        preprocess_fn=preprocess_image,
        device="cpu",
    )
    overlay = overlay_heatmap_on_image(str(image_path), heatmap)
    saved_path = save_explanation_image(cache_key, overlay)
    logger.info("Saved explanation to %s", saved_path)
    print(str(saved_path))


if __name__ == "__main__":
    main()
