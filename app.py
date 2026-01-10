"""Streamlit UI for visual product search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.analytics.metrics import average_similarity, total_searches
from src.config import get_settings, resolve_embedding_scope
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Fraunces:opsz,wght@9..144,600&display=swap');

        :root {
            --bg: #0b0f14;
            --bg-panel: #0f172a;
            --bg-elevated: #111827;
            --border: #1f2937;
            --text: #f8fafc;
            --muted: #cbd5e1;
            --accent: #22d3ee;
            --accent-strong: #14b8a6;
            --warning: #f97316;
            --shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            color: var(--text) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Fraunces', 'Georgia', serif;
            letter-spacing: 0.2px;
            color: var(--text) !important;
        }

        p, span, label, li, small, caption, div, section, button {
            color: var(--text) !important;
        }

        a, a:visited {
            color: var(--accent) !important;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-panel) !important;
            border-right: 1px solid var(--border) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--bg-elevated) !important;
            border: 1px solid var(--border) !important;
            padding: 12px 16px;
            border-radius: 14px;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetric"] * {
            color: var(--text) !important;
        }

        .panel-card,
        .result-card {
            background: var(--bg-elevated) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px;
            padding: 16px;
            box-shadow: var(--shadow);
        }

        .result-card {
            border-radius: 18px;
            padding: 12px;
        }

        .result-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 0.4rem;
            color: var(--text) !important;
        }

        .result-meta {
            color: var(--muted) !important;
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        .cluster-pill {
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: #0b2f3a !important;
            color: var(--accent) !important;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid #0e7490 !important;
        }

        .badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: #312006 !important;
            color: #fbbf24 !important;
            font-size: 0.7rem;
            font-weight: 600;
            margin-top: 0.35rem;
            border: 1px solid #92400e !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #0d9488, #22d3ee) !important;
            color: #0b0f14 !important;
            border-radius: 10px;
            border: 0 !important;
            padding: 0.6rem 1rem;
            font-weight: 700;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #0f766e, #06b6d4) !important;
        }

        input, textarea, select, [data-baseweb="input"] input {
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }

        input::placeholder, textarea::placeholder {
            color: #94a3b8 !important;
        }

        [data-testid="stFileUploader"] {
            background: var(--bg-panel) !important;
            border: 1px dashed var(--border) !important;
            border-radius: 12px;
            padding: 10px;
        }

        [data-testid="stExpander"] {
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px;
        }

        [data-testid="stExpander"] summary {
            color: var(--text) !important;
        }

        div[data-testid="stTabs"] button {
            color: var(--muted) !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--text) !important;
            border-bottom: 2px solid var(--accent) !important;
        }

        div[data-testid="stProgress"] > div > div {
            background-color: var(--warning) !important;
        }

        table, thead, tbody, tr, th, td {
            background: var(--bg-elevated) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def serialize_product(product: Product) -> dict:
    return {
        "id": product.id,
        "name": product.name,
        "category": product.category,
        "image_path": product.image_path,
    }


def load_summary(
    limit: int = 5,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> tuple[int, int, list[dict]]:
    """Return total product count, embedding count, and a limited list (serialized)."""

    resolved_version, resolved_type = resolve_embedding_scope(
        embedding_version,
        embedding_type,
    )
    with get_session() as session:
        total = session.execute(select(func.count(Product.id))).scalar_one()
        embeddings = session.execute(
            select(func.count(ImageEmbedding.id)).where(
                ImageEmbedding.embedding_version == resolved_version,
                ImageEmbedding.embedding_type == resolved_type,
            )
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


def render_sidebar() -> None:
    st.sidebar.markdown(f"### {settings.app_name}")
    st.sidebar.caption("Production-inspired visual product search demo.")
    st.sidebar.markdown(
        """
        **Pipeline**
        - Image to embedding (ResNet50)
        - Cosine similarity search
        - Optional FAISS ANN
        - Clustering and Grad-CAM
        """
    )
    st.sidebar.markdown(
        """
        **Tips**
        - Upload a JPG or PNG
        - Adjust Top-K for more results
        - Enable Grad-CAM to inspect matches
        """
    )


def render_top_metrics(total_products: int, total_embeddings: int) -> None:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Products", total_products)
    metric_cols[1].metric("Embeddings", total_embeddings)
    metric_cols[2].metric("Database", "OK")


def render_result_card(
    item: dict,
    product_map: dict[int, dict],
    explanation_map: dict[int, Path],
    explain_enabled: bool,
) -> None:
    product = product_map.get(item["product_id"])
    similarity_text, similarity_value = format_similarity(item["similarity_score"])
    cluster_text = "N/A" if item.get("cluster_id") is None else str(item["cluster_id"])

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    if product:
        product_path = resolve_image_path(product["image_path"])
        if product_path.exists():
            st.image(str(product_path), use_container_width=True)
        else:
            st.info("Image not available.")
        st.markdown(
            f"<div class='result-title'>{product['name']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Product not found.")
        st.markdown(
            f"<div class='result-title'>Product {item['product_id']}</div>",
            unsafe_allow_html=True,
        )

    if item.get("rank") == 1:
        st.markdown("<div class='badge'>Top match</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='result-meta'>Similarity {similarity_text}</div>",
        unsafe_allow_html=True,
    )
    st.progress(similarity_value)
    st.markdown(
        f"<div class='cluster-pill'>Cluster {cluster_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if explain_enabled:
        with st.expander("Why this match?"):
            explanation = explanation_map.get(item["product_id"])
            if explanation:
                st.image(str(explanation), use_container_width=True)
                st.caption(
                    "Grad-CAM highlights regions that most influenced the similarity score."
                )
            else:
                st.info("Grad-CAM not available for this item.")


def render_search_tab(
    total_products: int,
    total_embeddings: int,
    top_products: list[dict],
) -> None:
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "query_path" not in st.session_state:
        st.session_state.query_path = None

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Query panel")
        st.caption("Upload an image to find visually similar products.")

        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a query image",
            type=["jpg", "jpeg", "png"],
            key="query_uploader",
        )
        use_local_path = st.checkbox("Use a local file path", value=False)
        local_path = st.text_input("Local image path", key="local_path_input")

        top_k = st.slider(
            "Top-K results",
            min_value=1,
            max_value=10,
            value=5,
            key="top_k",
        )
        explain_enabled = st.checkbox("Enable Grad-CAM explanations", value=False)

        with st.expander("Search options", expanded=False):
            use_faiss = st.checkbox("Use FAISS (fast ANN)", value=False, key="use_faiss")
            same_cluster_first = st.checkbox(
                "Prioritize same product group",
                value=True,
                key="same_cluster_first",
            )
            st.caption("ANN mode may trade exactness for speed.")
            if use_faiss and same_cluster_first:
                st.caption(
                    "Cluster-prioritized search uses brute-force fallback when FAISS is enabled."
                )

        search_clicked = st.button("Search", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if use_local_path and local_path:
            candidate = resolve_image_path(local_path)
            if candidate.exists():
                st.markdown("#### Preview")
                st.image(str(candidate), caption="Local query image", use_container_width=True)
            else:
                st.info("Local image path not found.")
        elif uploaded_file is not None:
            st.markdown("#### Preview")
            st.image(uploaded_file, caption="Query image", use_container_width=True)

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

    with right:
        st.markdown("### Results")
        results: list[dict] = st.session_state.search_results or []
        query_path_str = st.session_state.query_path
        query_path = Path(query_path_str) if query_path_str else None

        if not query_path:
            st.info("Upload an image and run a search to see results.")
            return
        if not results:
            st.info("No embeddings found - run `python scripts/build_embeddings.py`.")
            return

        with get_session() as session:
            product_map = load_products_by_ids(
                session,
                [item["product_id"] for item in results],
            )

        explanation_map: dict[int, Path] = {}
        if explain_enabled:
            with st.spinner("Generating Grad-CAM explanations..."):
                _, explanation_map = build_explanations(
                    query_path=query_path,
                    results=results,
                    product_map=product_map,
                    limit=len(results),
                )

        st.caption(f"{len(results)} results")
        columns_count = 3 if len(results) > 4 else 2
        grid_cols = st.columns(columns_count, gap="large")
        for idx, item in enumerate(results):
            with grid_cols[idx % columns_count]:
                render_result_card(item, product_map, explanation_map, explain_enabled)

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


def render_analytics_tab(total_products: int, total_embeddings: int) -> None:
    st.header("Analytics")
    st.caption("Embedding space and search quality snapshots.")

    with get_session() as session:
        search_count = total_searches(session)
        avg_score = average_similarity(session)
        similarity_scores = load_similarity_scores(session)

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Total searches", search_count)
    kpi_cols[1].metric("Avg similarity", f"{avg_score:.4f}")
    kpi_cols[2].metric("Embeddings", total_embeddings)

    projection_data = load_projection_data()
    if projection_data is None:
        st.info("Projection data not found. Run `python scripts/build_projection.py`.")
    else:
        projections, cluster_ids = projection_data
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=cluster_ids,
            cmap="tab10",
            s=12,
            alpha=0.85,
        )
        ax.set_title("Embedding space (t-SNE)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        center_col = st.columns([1, 3, 1])[1]
        center_col.pyplot(fig, use_container_width=True)

    with st.expander("Similarity distribution", expanded=False):
        if similarity_scores:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(similarity_scores, bins=20, color="#4C72B0", alpha=0.85)
            ax.set_title("Similarity score distribution")
            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No similarity logs yet. Run a few searches to populate metrics.")


def render_about_tab() -> None:
    st.header("About / Architecture")
    st.markdown(
        "This demo mirrors a production-inspired ML search system with offline indexing and online retrieval."
    )
    st.subheader("System architecture")
    st.markdown(
        "- Offline: batch embedding generation, clustering, FAISS indexing, and projection artifacts.\n"
        "- Online: query embedding, cosine similarity search, and logging.\n"
        "- Explainability: Grad-CAM overlays cached to disk."
    )
    st.subheader("Technologies")
    st.markdown(
        "- ResNet50 embeddings\n"
        "- SQLite + SQLAlchemy\n"
        "- FAISS (optional ANN)\n"
        "- K-Means clustering\n"
        "- Streamlit UI\n"
        "- Grad-CAM explainability"
    )
    st.subheader("Design intent")
    st.markdown(
        "The goal is to showcase a clean, production-inspired pipeline that is easy to extend and demo."
    )


st.set_page_config(page_title=settings.app_name, layout="wide")
inject_styles()
render_sidebar()

st.title(settings.app_name)
st.caption("Visual similarity search demo.")

try:
    total_products, total_embeddings, top_products = load_summary()
    render_top_metrics(total_products, total_embeddings)
    st.divider()

    tabs = st.tabs(["Visual Search", "Analytics", "About / Architecture"])
    with tabs[0]:
        render_search_tab(total_products, total_embeddings, top_products)
    with tabs[1]:
        render_analytics_tab(total_products, total_embeddings)
    with tabs[2]:
        render_about_tab()
except Exception as exc:
    logger.exception("App failed to load products: %s", exc)
    st.error("DB connection failed")
    st.exception(exc)
