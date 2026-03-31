"""
Test the full retrieval pipeline: hybrid search + re-ranking + diversity.
Compare it against pure dense search to show the improvement.
"""

import time
from src.utils.logger import setup_logger, get_logger
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.vectorstore.qdrant_store import QdrantStore
from src.embeddings.embedding_model import EmbeddingModel

setup_logger()
logger = get_logger(__name__)


def test_pipeline(pipeline: RetrievalPipeline, query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    start = time.time()
    results = pipeline.retrieve(query, top_k_final=5)
    elapsed = time.time() - start

    print(f"Retrieved {len(results)} results in {elapsed:.2f}s\n")

    for i, r in enumerate(results):
        print(f"[{i+1}] CE Score: {r.get('ce_score', 'N/A'):>7} | "
              f"RRF: {r.get('rrf_score', 'N/A'):.4f}")
        print(f"     {r.get('title','')[:65]}...")
        print(f"     {r.get('text','')[:120].replace(chr(10),' ')}...")
        print()


def main():
    logger.info("Initializing full retrieval pipeline...")
    pipeline = RetrievalPipeline()

    # Test 1: Conceptual query
    test_pipeline(
        pipeline,
        "how does self-attention mechanism work in transformers"
    )

    # Test 2: Specific method query - tests BM25 keyword advantage
    test_pipeline(
        pipeline,
        "LoRA low-rank adaptation fine-tuning"
    )

    # Test 3: Comparison query
    test_pipeline(
        pipeline,
        "reinforcement learning reward shaping techniques"
    )

    # Test 4: With year filter
    print(f"\n{'='*60}")
    print("FILTERED: 'graph neural networks' (2026 only)")
    print(f"{'='*60}")

    results = pipeline.retrieve(
        "graph neural networks",
        filter_year_gte = 2026,
        top_k_final = 3
    )

    for i, r in enumerate(results):
        print(
            f"[{i+1}] {r.get('published_date', 'N/A')} | "
            f"CE: {r.get('ce_score','N/A'):>6} | "
            f"{r.get('title','')[:55]}..."
        )

    logger.info("\n✅ Retrieval pipeline test complete")


if __name__ == "__main__":
    main()