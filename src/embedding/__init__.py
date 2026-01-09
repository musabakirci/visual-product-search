"""Embedding pipeline package."""

from src.embedding.extractor import extract_embedding
from src.embedding.model import get_embedding_model
from src.embedding.preprocess import preprocess_image
from src.embedding.storage import save_embedding

__all__ = [
    "extract_embedding",
    "get_embedding_model",
    "preprocess_image",
    "save_embedding",
]
