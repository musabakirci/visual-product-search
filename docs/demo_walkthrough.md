# Demo Walkthrough

Use this script to present the project during a live demo or interview.

## Opening

- "This is an end-to-end visual product search system. It converts images into embeddings, finds similar items, and explains why those matches were returned."
- "It includes offline pipelines for embeddings, clustering, FAISS indexing, and projection, plus an online search UI with analytics."

## Visual Search tab

1) Upload a query image and click Search.
2) Point to the hero result and the grid of matches.
3) Explain the similarity score and cluster ID.
4) Toggle FAISS and cluster prioritization to show control over retrieval strategy.

Suggested narration:
- "The query is embedded with ResNet50 and compared against stored embeddings."
- "We can use exact cosine similarity or FAISS for faster approximate search."

## Explainability

1) Enable "Explain top matches (Grad-CAM)".
2) Show the query heatmap and the top match heatmap.
3) Highlight how the model focuses on product-relevant regions.

Suggested narration:
- "Grad-CAM provides a visual explanation of what the model attends to."
- "Explanations are cached on disk to keep the UI responsive."

## Analytics tab

1) Show KPI metrics (total searches, average similarity).
2) Display the embedding scatter plot.
3) Discuss how clusters and projections help assess model structure.

Suggested narration:
- "This dashboard provides evidence of model behavior and search quality."
- "We can identify cluster cohesion and search distribution at a glance."

## About tab

1) Summarize the architecture and pipeline.
2) Emphasize the full-stack ML scope: embeddings, indexing, analytics, explainability.

## How to answer: "Why is this different from basic image search?"

Key points:
- Uses deep embeddings rather than raw pixel similarity.
- Supports scalable ANN retrieval with FAISS.
- Adds clustering to structure the catalog and improve relevance.
- Provides analytics for quality monitoring.
- Includes explainability for trust and debugging.

## Closing

- "This system is a strong baseline for production-grade visual search, and it is designed to scale with better models and vector databases."
