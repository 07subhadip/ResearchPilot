FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache-bust: forces Docker to re-copy source code on every build
# This ensures HuggingFace always gets the latest code from git
ARG CACHEBUST=20260414_2

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY run_api.py .
COPY .env.example ./.env

# Copy data (uploaded via HuggingFace web UI)
# COPY data/qdrant_db/ ./data/qdrant_db/
# COPY data/embeddings/bm25_index.pkl ./data/embeddings/bm25_index.pkl
# COPY data/embeddings/embeddings.npy ./data/embeddings/embeddings.npy
# COPY data/embeddings/chunk_ids.npy ./data/embeddings/chunk_ids.npy
# COPY data/embeddings/embedding_index.json ./data/embeddings/embedding_index.json
# COPY data/chunks/ ./data/chunks/

# Download the 4.4 GB database from the limits-free HF Dataset using git
# This happens during the Docker build so the API starts instantly later
RUN git lfs install && git clone https://huggingface.co/datasets/Subhadip007/researchpilot-data /app/data

# Create remaining data dirs inside the cloned repo
RUN mkdir -p data/raw data/processed logs

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Start the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
