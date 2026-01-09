"""Minimal Streamlit app for database smoke testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.analytics.metrics import average_similarity, total_searches
from src.config import get_settings
from src.db.models import ImageEmbedding, Product, SimilarityLog
from src.db.session import get_session
from src.embedding.extractor import extract_embedding
from src.embedding.model import get_embedding_model
from src.embedding.preprocess import preprocess_image
from src.explain.cache import get_cache_key, load_cached_explanation, save_explanation_image
from src.explain.gradcam import generate_gradcam
from src.explain.overlay import overlay_heatmap_on_image
from src.search.retrieval import log_similarity_results, search_similar_products
from src.services.logging_service import configure_logging
from src.utils.paths import get_data_dir, get_project_root

logger = configure_logging()
settings = get_settings()

def serialize_product(product: Product) -> dict:
    return {
        "id": product.id,
        "name": product.name,
        "category": product.category,
        "image_path": product.image_path,
    }

def load_summary(limit: int = 5) -> tuple[int, int, list[dict]]:
    """Return total product count, embedding count, and a limited list (serialized)."""

    with get_session() as session:
        total = session.execute(select(func.count(Product.id))).scalar_one()
        embeddings = session.execute(
            select(func.count(ImageEmbedding.id))
        ).scalar_one()

        products = session.execute(
            select(Product).order_by(Product.id).limit(limit)
        ).scalars().all()

        serialized_products = [serialize_product(p) for p in products]

    return total, embeddings, serialized_products



def resolve_image_path(image_path: str) -> Path:
    """Resolve an image path relative to the project root."""

    path = Path(image_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded file to a temporary location."""

    temp_dir = get_data_dir() / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    temp_path = temp_dir / f"query_{uuid4().hex}{suffix}"
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


def load_products_by_ids(session: Session, product_ids: list[int]) -> dict[int, dict]:
    """Load products by ID and return a lookup map."""

    if not product_ids:
        return {}
    stmt = select(Product).where(Product.id.in_(product_ids))
    products = session.execute(stmt).scalars().all()
    return {p.id: serialize_product(p) for p in products}



def load_projection_data() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load precomputed projections and cluster IDs from disk."""

    projections_dir = get_data_dir() / "projections"
    projections_path = projections_dir / "projections.npy"
    cluster_ids_path = projections_dir / "cluster_ids.npy"
    if not projections_path.exists() or not cluster_ids_path.exists():
        return None
    projections = np.load(projections_path)
    cluster_ids = np.load(cluster_ids_path)
    return projections, cluster_ids


def load_similarity_scores(session: Session) -> list[float]:
    """Load similarity scores from the logs."""

    stmt = select(SimilarityLog.similarity_score).where(
        SimilarityLog.similarity_score.is_not(None)
    )
    scores = session.execute(stmt).scalars().all()
    return [float(score) for score in scores if score is not None]


def get_or_create_explanation(
    image_path: str,
    model_name: str = "resnet50",
    device: str = "cpu",
) -> Optional[Path]:
    """Return a cached Grad-CAM overlay, generating it if needed."""

    try:
        cache_key = get_cache_key(image_path, model_name)
        cached = load_cached_explanation(cache_key)
        if cached is not None:
            return cached

        model = get_embedding_model()
        heatmap = generate_gradcam(
            image_path=image_path,
            model=model,
            preprocess_fn=preprocess_image,
            device=device,
        )
        overlay = overlay_heatmap_on_image(image_path, heatmap)
        return save_explanation_image(cache_key, overlay)
    except Exception as exc:
        logger.exception("Failed to create explanation for %s: %s", image_path, exc)
        return None


def format_similarity(score: float) -> tuple[str, float]:
    """Return similarity percent string and progress bar value."""

    clamped = max(0.0, min(1.0, float(score)))
    return f"{clamped * 100:.1f}%", clamped


def build_explanations(
    query_path: Path,
    results: list[dict],
    product_map: dict[int, dict],
    limit: int = 3,
) -> tuple[Optional[Path], dict[int, Path]]:

    query_explanation = get_or_create_explanation(str(query_path))
    product_explanations: dict[int, Path] = {}

    for item in results[:limit]:
        product = product_map.get(item["product_id"])
        if not product:
            continue

        product_path = resolve_image_path(product["image_path"])
        if not product_path.exists():
            continue

        explanation = get_or_create_explanation(str(product_path))
        if explanation:
            product_explanations[item["product_id"]] = explanation

    return query_explanation, product_explanations


def render_about_tab() -> None:
    """Render the About / Architecture tab."""

    st.header("About / Architecture")
    st.markdown(
        "This demo showcases an end-to-end visual product search pipeline built for fast iteration "
        "and explainability."
    )
    st.subheader("Pipeline")
    st.markdown("Image -> Embedding -> Similarity -> Clustering -> ANN -> Explanation")
    st.subheader("Tech stack")
    st.markdown(
        "- PyTorch ResNet50 embeddings\n"
        "- SQLite + SQLAlchemy\n"
        "- FAISS (optional ANN)\n"
        "- K-Means clustering\n"
        "- Streamlit UI\n"
        "- Grad-CAM explainability"
    )
    st.subheader("Notes")
    st.markdown(
        "This project is designed to be extended with better indexing, richer analytics, and "
        "advanced explainability in future steps."
    )


def render_analytics_tab(total_products: int, total_embeddings: int) -> None:
    """Render the Analytics & Embedding Space tab."""

    st.header("Analytics & Embedding Space")
    st.caption("This dashboard provides insight into model behavior and search quality.")
    with get_session() as session:
        search_count = total_searches(session)
        avg_score = average_similarity(session)
        similarity_scores = load_similarity_scores(session)

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total products", total_products)
    kpi_cols[1].metric("Total embeddings", total_embeddings)
    kpi_cols[2].metric("Total searches", search_count)
    kpi_cols[3].metric("Avg similarity", f"{avg_score:.4f}")

    projection_data = load_projection_data()
    if projection_data is None:
        st.info("Projection data not found. Run `python scripts/build_projection.py`.")
    else:
        projections, cluster_ids = projection_data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=cluster_ids,
            cmap="tab10",
            s=12,
            alpha=0.8,
        )
        ax.set_title("Embedding space (2D)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        st.pyplot(fig, use_container_width=True)

    if similarity_scores:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(similarity_scores, bins=20, color="#4C72B0", alpha=0.85)
        ax.set_title("Similarity score distribution")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("No similarity logs yet. Run a few searches to populate metrics.")


def render_visual_search_tab(
    total_products: int,
    total_embeddings: int,
    top_products: list[Product],
) -> None:
    """Render the Visual Search tab."""

    st.header("Visual Search")
    st.caption("Upload an image to find visually similar products.")
    status_cols = st.columns(3)
    status_cols[0].success("DB connection OK")
    status_cols[1].metric("Products", total_products)
    status_cols[2].metric("Embeddings", total_embeddings)

    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "query_path" not in st.session_state:
        st.session_state.query_path = None

    query_container = st.container()
    with query_container:
        left, center, right = st.columns([1, 2, 1])
        with center:
            uploaded_file = st.file_uploader(
                "Upload a query image",
                type=["jpg", "jpeg", "png"],
                key="query_uploader",
            )
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Query image", use_column_width=True)

            search_clicked = st.button("Search", type="primary", use_container_width=True)

    with st.expander("Search options", expanded=False):
        use_faiss = st.checkbox("Use FAISS (fast ANN)", value=False, key="use_faiss")
        same_cluster_first = st.checkbox(
            "Prioritize same product group",
            value=True,
            key="same_cluster_first",
        )
        top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5, key="top_k")
        st.caption("ANN mode may trade exactness for speed.")
        if use_faiss and same_cluster_first:
            st.caption("Cluster-prioritized search currently uses brute-force fallback.")

    with st.expander("Advanced input options", expanded=False):
        use_local_path = st.checkbox("Use a local file path", value=False, key="use_local_path")
        local_path = st.text_input("Local image path", key="local_path_input")
        if use_local_path and local_path:
            candidate = resolve_image_path(local_path)
            if candidate.exists():
                st.image(str(candidate), caption="Local query image")
            else:
                st.info("Image path not found.")

    st.caption("Change options and press Search to refresh results.")

    if search_clicked:
        query_path: Optional[Path] = None
        if use_local_path and local_path:
            candidate = resolve_image_path(local_path)
            if candidate.exists():
                query_path = candidate
            else:
                st.warning("Local image path not found.")
        elif uploaded_file is not None:
            query_path = save_uploaded_file(uploaded_file)
        else:
            st.warning("No image uploaded yet.")

        if query_path is not None and query_path.exists():
            with st.spinner("Searching for similar products..."):
                try:
                    query_embedding = extract_embedding(str(query_path))
                    with get_session() as session:
                        results = search_similar_products(
                            session=session,
                            query_embedding=query_embedding,
                            top_k=int(top_k),
                            same_cluster_first=same_cluster_first,
                            use_faiss=use_faiss,
                        )
                        log_similarity_results(session, str(query_path), results)
                    st.session_state.search_results = results
                    st.session_state.query_path = str(query_path)
                except Exception as exc:
                    logger.exception("Similarity search failed: %s", exc)
                    st.error("Similarity search failed")
                    st.exception(exc)

    results: list[dict] = st.session_state.search_results or []
    query_path_str = st.session_state.query_path
    query_path = Path(query_path_str) if query_path_str else None

    if not query_path:
        st.info("No image uploaded yet.")
    elif not results:
        st.info("No embeddings found - run `python scripts/build_embeddings.py`.")
    else:
        with get_session() as session:
            product_map = load_products_by_ids(
                session,
                [item["product_id"] for item in results],
            )

        explain_enabled = st.checkbox("Explain top matches (Grad-CAM)", value=False)
        if explain_enabled:
            st.caption("Explanations are indicative and may not be perfect.")

        query_explanation: Optional[Path] = None
        explanation_map: dict[int, Path] = {}
        if explain_enabled and query_path is not None:
            with st.spinner("Generating explanations..."):
                query_explanation, explanation_map = build_explanations(
                    query_path=query_path,
                    results=results,
                    product_map=product_map,
                    limit=3,
                )

        st.subheader("Results")
        hero = results[0]
        hero_product = product_map.get(hero["product_id"])
        hero_cluster = hero.get("cluster_id")
        hero_score_text, hero_progress = format_similarity(hero["similarity_score"])

        hero_cols = st.columns([1.2, 1.8])
        with hero_cols[0]:
            if hero_product:
                hero_image_path = resolve_image_path(hero_product["image_path"])

                if hero_image_path.exists():
                    st.image(str(hero_image_path), caption="Top match", use_column_width=True)
                else:
                    st.info("Image not available.")
            else:
                st.info("Product not found.")

            if query_path and query_path.exists():
                st.image(str(query_path), caption="Query image", use_column_width=True)
                if explain_enabled:
                    with st.expander("Why this query?"):
                        if query_explanation:
                            st.image(str(query_explanation), caption="Query Grad-CAM")
                            st.caption(
                                "Highlighted regions indicate areas most influential for similarity."
                            )
                        else:
                            st.info("Unable to generate query explanation.")

        with hero_cols[1]:
            if hero_product:
                st.markdown(f"### {hero_product['name']}")
                st.caption(hero_product["category"] or "Uncategorized")
                st.progress(hero_progress)
                cluster_text = "N/A" if hero_cluster is None else str(hero_cluster)
                st.caption(f"Similarity: {hero_score_text} | Cluster: {cluster_text}")
            else:
                st.markdown("### Top match")
                st.caption(f"Similarity: {hero_score_text}")

            if explain_enabled:
                with st.expander("Why this match?"):
                    explanation = explanation_map.get(hero["product_id"])
                    if explanation:
                        st.image(str(explanation), caption="Top match Grad-CAM")
                        st.caption(
                            "Highlighted regions indicate areas most influential for similarity."
                        )
                    else:
                        st.info("Explanation not available for this item.")

        remaining = results[1:]
        if remaining:
            st.subheader("More matches")
            columns_count = 3 if len(remaining) > 4 else 2
            cols = st.columns(columns_count)
            for idx, item in enumerate(remaining):
                product = product_map.get(item["product_id"])
                if not product:
                    continue
                score_text, _ = format_similarity(item["similarity_score"])
                cluster_text = "N/A" if item.get("cluster_id") is None else str(item["cluster_id"])
                col = cols[idx % columns_count]
                with col:
                    product_path = resolve_image_path(product["image_path"])
                    if product_path.exists():
                        st.image(str(product_path), use_column_width=True)
                    st.markdown(f"**{product['name']}**")
                    st.caption(f"{score_text} similarity | Cluster {cluster_text}")
                    if explain_enabled:
                        with st.expander("Why this match?"):
                            explanation = explanation_map.get(item["product_id"])
                            if explanation:
                                st.image(str(explanation), caption="Grad-CAM")
                                st.caption(
                                    "Highlighted regions indicate areas most influential for similarity."
                                )
                            else:
                                st.info("Explanation available for top 3 matches.")

        with st.expander("Catalog preview", expanded=False):
            if top_products:
                st.table(
                    [
                        {
                             "name": product["name"],
                             "category": product["category"],
                             "image_path": product["image_path"],
                        }
                        for product in top_products
                    ]
                )
            else:
                st.info("No products found. Run the seed script.")


st.set_page_config(page_title=settings.app_name)
st.title(settings.app_name)
st.caption("Visual similarity search demo.")

try:
    total_products, total_embeddings, top_products = load_summary()
    tabs = st.tabs(["Visual Search", "Analytics", "About / Architecture"])
    with tabs[0]:
        render_visual_search_tab(total_products, total_embeddings, top_products)
    with tabs[1]:
        render_analytics_tab(total_products, total_embeddings)
    with tabs[2]:
        render_about_tab()
except Exception as exc:
    logger.exception("App failed to load products: %s", exc)
    st.error("DB connection failed")
    st.exception(exc)
