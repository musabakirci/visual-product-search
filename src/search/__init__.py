"""Search package."""

from src.search.retrieval import log_similarity_results, search_similar_products
from src.search.similarity import cosine_similarity

__all__ = [
    "cosine_similarity",
    "log_similarity_results",
    "search_similar_products",
]
