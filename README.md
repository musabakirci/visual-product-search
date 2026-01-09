🖼️ Visual Product Search with Embeddings & Explainability

An end-to-end visual product search system that retrieves visually similar products from images using deep learning embeddings and vector similarity.
Built with a production mindset to demonstrate scalable retrieval, analytics, and explainable AI.

✨ Key Features

🧠 CNN-based image embeddings (ResNet50)

🔍 Vector similarity search (cosine, FAISS ANN)

🧩 Unsupervised clustering for product grouping

📊 Analytics dashboard (embedding space & similarity metrics)

🔥 Grad-CAM explainability for visual trust

🖥️ Interactive Streamlit UI

🏗️ System Design

Offline pipelines: embedding generation, clustering, ANN indexing, projection

Online flow: query embedding, similarity retrieval, logging, explainability

This separation enables fast queries and scalable indexing.

🧰 Tech Stack

PyTorch · FAISS · scikit-learn · SQLAlchemy · Streamlit · NumPy · Matplotlib

📊 Analytics & Explainability

Embedding space visualization (t-SNE)

Similarity score distributions

Grad-CAM heatmaps highlighting influential image regions

⚠️ Limitations & Future Work

Multimodal search (image + text)

Incremental indexing for live catalogs

Vector database integration for large-scale deployments
