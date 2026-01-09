# Pipelines

This document explains the end-to-end pipelines from data ingestion to explainability.

## 1) Data ingestion

- Product records store local image paths in the database.
- Sample data can be seeded with `scripts/seed_products.py`.

## 2) Embedding generation

- ResNet50 (ImageNet) is used as a feature extractor.
- The output is a 2048-dim embedding from the global average pooling layer.
- Embeddings are L2-normalized and stored as float32 BLOBs.

## 3) Clustering

- K-Means clusters the embeddings into product groups.
- Cluster IDs are persisted to `image_embeddings.cluster_id`.
- Re-running clustering overwrites existing cluster assignments.

## 4) FAISS indexing

- Embeddings are loaded and indexed using FAISS IndexFlatIP.
- A product_id mapping is saved alongside the FAISS index file.

## 5) Query-time search

- A query image is embedded using the same pipeline.
- Retrieval uses brute-force cosine similarity or optional FAISS ANN.
- Cluster-aware search can prioritize the predicted cluster.

## 6) Logging and analytics

- Each search result is logged to `similarity_logs`.
- Analytics compute total searches, average similarity, and distributions.
- Embeddings are projected to 2D for visualization.

## 7) Explainability

- Grad-CAM heatmaps are generated on demand using ResNet50 layer4.
- Overlays are cached to disk for faster reuse.
