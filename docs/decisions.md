# Technical Decisions

This document captures key design choices and trade-offs.

## ResNet50 embeddings

- Why: strong baseline, widely benchmarked, simple to deploy, no custom training required.
- Trade-off: not domain-specific; a fine-tuned model could improve accuracy.
- Alternatives considered: ViT models, CLIP image encoder, custom fine-tuning.

## Cosine similarity

- Why: standard for normalized embeddings; stable and interpretable.
- Trade-off: assumes embedding magnitude is not informative (normalized vectors).
- Alternatives considered: Euclidean distance, learned similarity metrics.

## FAISS IndexFlatIP

- Why: simplest exact ANN option for inner product; good baseline before IVF/HNSW.
- Trade-off: still linear in memory; not ideal for very large catalogs.
- Alternatives considered: FAISS IVF, HNSW, ScaNN, vector databases.

## K-Means clustering

- Why: fast, interpretable, easy to recompute, works well for coarse grouping.
- Trade-off: spherical clusters; sensitive to k; can miss complex structure.
- Alternatives considered: DBSCAN, HDBSCAN, Gaussian Mixtures.

## Grad-CAM explainability

- Why: lightweight, model-agnostic for CNNs, produces intuitive heatmaps.
- Trade-off: coarse spatial resolution; not a full causal explanation.
- Alternatives considered: Integrated Gradients, RISE, attention-based models.

## Streamlit UI

- Why: fast to build, ideal for demos, minimal infrastructure.
- Trade-off: not a production UI framework.
- Alternatives considered: React frontend, Dash, FastAPI + custom UI.
