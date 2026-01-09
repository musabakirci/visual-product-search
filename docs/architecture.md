# Architecture

## Overview

The system combines offline pipelines (embedding generation, clustering, indexing, projection) with an online query flow that performs similarity search, logging, and explainability. The UI surfaces search results, analytics, and explanations without altering core ML behavior.

## Components

- UI (Streamlit): query input, results, analytics, explainability
- Search layer: cosine similarity, optional FAISS ANN
- Embedding layer: ResNet50 feature extraction and normalization
- Clustering: K-Means cluster assignments for product groups
- Indexing: FAISS IndexFlatIP for fast retrieval
- Storage: SQLite via SQLAlchemy and Alembic
- Analytics: similarity_logs metrics and embedding projection plots
- Explainability: Grad-CAM overlays with disk cache

## Data Flow

Offline pipelines:
1) Ingest product images (local paths)
2) Generate embeddings (ResNet50, L2-normalized)
3) Cluster embeddings (K-Means) and persist cluster_id
4) Build FAISS index and product_id mapping
5) Project embeddings to 2D (t-SNE) for visualization

Online query:
1) User uploads a query image
2) Extract query embedding
3) Retrieve top-K similar products (brute-force or FAISS)
4) Log results to similarity_logs
5) Generate Grad-CAM overlays on demand

## Diagram (ASCII)

```
            +-------------------+
            |   Streamlit UI    |
            +---------+---------+
                      |
                      v
            +-------------------+
            |  Query Embedding  |
            +---------+---------+
                      |
                      v
            +-------------------+        +---------------------+
            | Similarity Search |<-------| FAISS Index (ANN)   |
            +---------+---------+        +---------------------+
                      |
                      v
            +-------------------+
            |  Results + Logs   |
            +---------+---------+
                      |
                      v
            +-------------------+
            | Grad-CAM Explain  |
            +-------------------+

Offline pipelines:
Images -> Embeddings -> Clustering -> FAISS Index
                       -> t-SNE Projections -> Analytics
```

If you add a visual diagram, place it at `docs/diagrams/architecture.png` and update this page to reference it.
