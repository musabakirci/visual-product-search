# Embedding-Based Visual Product Search with Explainability

This project implements an end-to-end visual product search system that retrieves visually similar products from images. It combines CNN-based embeddings, cosine similarity, optional FAISS ANN acceleration, unsupervised clustering, an analytics dashboard, and Grad-CAM explainability. The result is a practical and interview-ready system that demonstrates strong ML engineering fundamentals from data pipelines to UI.

## Key Features

- CNN embeddings (ResNet50, normalized)
- Similarity search (cosine, brute-force + optional FAISS ANN)
- Unsupervised clustering for product grouping
- Analytics dashboard with embedding projections and search metrics
- Grad-CAM explainability with disk cache
- Clean Streamlit UI with visual-first results

## System Architecture

The system is split into offline pipelines (embedding, clustering, indexing, projection) and an online query flow that performs search, logging, and explainability. See `docs/architecture.md` for a detailed overview and diagram.

## Tech Stack

- Python 3.11+
- PyTorch (ResNet50)
- FAISS (IndexFlatIP)
- scikit-learn (K-Means, t-SNE)
- SQLAlchemy + Alembic (SQLite)
- Streamlit
- NumPy, Matplotlib, Pillow

## How to Run

1) Create and activate a virtual environment:
```
python -m venv .venv
```
Windows:
```
.venv\Scripts\activate
```
macOS/Linux:
```
source .venv/bin/activate
```

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Create an environment file:
```
copy .env.example .env
```
macOS/Linux:
```
cp .env.example .env
```

4) Initialize the database:
```
alembic upgrade head
```

5) Seed sample data:
```
python scripts/seed_products.py
```

6) Build embeddings:
```
python scripts/build_embeddings.py
```

7) Build clusters:
```
python scripts/build_clusters.py
```

8) Build FAISS index:
```
python scripts/build_faiss_index.py
```

9) Build embedding projections:
```
python scripts/build_projection.py
```

10) Run the app:
```
streamlit run app.py
```

## Project Structure

```
project/
  app.py
  README.md
  requirements.txt
  .env.example
  alembic.ini
  alembic/
    env.py
    versions/
  data/
    images/
    explanations/
    faiss/
    projections/
    tmp/
  docs/
    architecture.md
    pipeline.md
    decisions.md
    demo_walkthrough.md
    diagrams/
  scripts/
    seed_products.py
    build_embeddings.py
    build_clusters.py
    build_faiss_index.py
    build_projection.py
    search_demo.py
    explain_demo.py
  src/
    analytics/
    clustering/
    db/
    embedding/
    explain/
    search/
    services/
    utils/
    vector_index/
  tests/
    test_db_smoke.py
```

## Screenshots (Placeholders)

- Visual Search: `docs/diagrams/visual_search.png`
- Analytics: `docs/diagrams/analytics.png`
- Explainability: `docs/diagrams/explainability.png`

## Documentation

- `docs/architecture.md` - system components and data flow
- `docs/pipeline.md` - offline and online pipelines
- `docs/decisions.md` - technical trade-offs
- `docs/demo_walkthrough.md` - live demo script

## Limitations and Future Work

- Real-time ingestion and incremental indexing
- Multimodal search (text + image)
- Scalable vector databases for large catalogs
- Online learning and feedback loops

## CV and Interview Section

CV bullets:
- Short: Built an end-to-end visual product search system with CNN embeddings, ANN search, clustering, analytics, and Grad-CAM explainability.
- Medium: Implemented a ResNet50-based embedding pipeline with cosine similarity and optional FAISS acceleration, plus clustering and analytics dashboards. Added Grad-CAM explanations and a Streamlit UI for demos.
- Detailed: Designed and delivered a full visual product search system in Python using PyTorch, SQLAlchemy, and FAISS. Built offline pipelines for embeddings, clustering, FAISS indexing, and embedding projections, then integrated a Streamlit UI with Grad-CAM explainability and analytics metrics to demonstrate search quality and model behavior.

Interview talking points:
- System design: offline pipelines vs online query path and how they connect
- Scalability: FAISS indexing strategy, potential vector DB migration
- Failure modes: missing images, embedding drift, index staleness
- Performance vs accuracy trade-offs: brute-force vs ANN, cluster-first filtering
- Explainability: Grad-CAM overlays to validate model focus and build trust
