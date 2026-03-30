"""
Phase 6: Generate embeddings for all semantic chunks.

Run from project root:
    python run_embedding.py

Input:  data/chunks/*_semantic.json   (15,664 chunks)
Output: data/embeddings/embeddings.npy (shape: 15664 x 768)
        data/embeddings/chunk_ids.npy
        data/embeddings/embedding_index.json
        data/embeddings/chunk_metadata.json
"""

from src.utils.logger import setup_logger, get_logger
from src.embeddings.embedding_pipeline import EmbeddingPipeline

setup_logger()
logger = get_logger(__name__)



def main():
    logger.info("=" * 60)
    logger.info("PHASE 6 — EMBEDDING PIPELINE")
    logger.info("=" * 60)

    pipeline = EmbeddingPipeline()
    stats    = pipeline.run()

    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)



if __name__ == "__main__":
    main()