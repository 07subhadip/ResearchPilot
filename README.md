---
title: ResearchPilot API
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ResearchPilot API

Production RAG system for ML research paper Q&A, powered by:
- **Retrieval**: Hybrid (Qdrant dense + BM25 sparse) with cross-encoder reranking
- **Generation**: Groq LLaMA-3.3-70B with streaming support
- **Embedding**: BAAI/bge-base-en-v1.5

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + system status |
| POST | `/query` | Standard (non-streaming) RAG query |
| POST | `/query/stream` | Streaming SSE RAG query |
| POST | `/feedback` | Submit user feedback |
| GET | `/docs` | Swagger UI |

## Environment Variables Required

Set these in your Space **Settings → Repository Secrets**:

- `GROQ_API_KEY` — Groq API key for LLM generation
- `HF_API_KEY` — HuggingFace API key (for future model fallback)
