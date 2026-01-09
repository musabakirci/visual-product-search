"""Analytics package."""

from src.analytics.embedding_projection import project_embeddings
from src.analytics.metrics import (
    average_similarity,
    searches_over_time,
    top_matched_products,
    total_searches,
)

__all__ = [
    "average_similarity",
    "project_embeddings",
    "searches_over_time",
    "top_matched_products",
    "total_searches",
]
