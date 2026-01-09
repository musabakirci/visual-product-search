🖼️ Embedding-Based Visual Product Search with Explainability

This project implements an end-to-end visual product search system that retrieves visually similar products directly from images.
It combines CNN-based embeddings, cosine similarity, optional FAISS ANN acceleration, unsupervised clustering, an analytics dashboard, and Grad-CAM explainability.

🎯 The result is a practical, production-style, and interview-ready system that demonstrates strong ML engineering fundamentals — from offline pipelines to an interactive UI.

✨ Key Features

🧠 CNN embeddings (ResNet50, L2-normalized)

🔍 Similarity search (cosine similarity, brute-force + optional FAISS ANN)

🧩 Unsupervised clustering for product grouping

📊 Analytics dashboard with embedding projections and search metrics

🔥 Grad-CAM explainability with disk-based caching

🖥️ Clean Streamlit UI with visual-first results

🏗️ System Architecture

The system is split into:

Offline pipelines

Embedding generation

Clustering

FAISS indexing

Embedding projections (t-SNE / UMAP)

Online query flow

Query embedding

Similarity search

Result logging

Explainability generation

📐 A detailed architecture overview and diagram are available in:
👉 docs/architecture.md

🧰 Tech Stack

🐍 Python 3.11+

🔥 PyTorch (ResNet50)

⚡ FAISS (IndexFlatIP)

📐 scikit-learn (K-Means, t-SNE)

🗄️ SQLAlchemy + Alembic (SQLite)

🖥️ Streamlit

📦 NumPy, Matplotlib, Pillow

🚀 How to Run
1️⃣ Create and activate a virtual environment
python -m venv .venv


Windows

.venv\Scripts\activate


macOS / Linux

source .venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Create environment file
copy .env.example .env


macOS / Linux

cp .env.example .env

4️⃣ Initialize the database
alembic upgrade head

5️⃣ Seed sample data
python scripts/seed_products.py

6️⃣ Build embeddings
python scripts/build_embeddings.py

7️⃣ Build clusters
python scripts/build_clusters.py

8️⃣ Build FAISS index
python scripts/build_faiss_index.py

9️⃣ Build embedding projections
python scripts/build_projection.py

🔟 Run the application
streamlit run app.py

🗂️ Project Structure
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

🖼️ Screenshots (Placeholders)

🔍 Visual Search → docs/diagrams/visual_search.png

📊 Analytics Dashboard → docs/diagrams/analytics.png

🔥 Explainability (Grad-CAM) → docs/diagrams/explainability.png

📚 Documentation

🏗️ docs/architecture.md — system components and data flow

🔄 docs/pipeline.md — offline and online pipelines

⚖️ docs/decisions.md — technical trade-offs

🎤 docs/demo_walkthrough.md — live demo script

⚠️ Limitations & Future Work

⏱️ Real-time ingestion and incremental indexing

🧠 Multimodal search (text + image)

🗃️ Scalable vector databases for large catalogs

🔁 Online learning and feedback loops
