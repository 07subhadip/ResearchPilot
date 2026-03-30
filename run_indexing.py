"""
Phase 7: Index all embeddings into Qdrant vector database.

Input:  data/embeddings/embeddings.npy
        data/embeddings/chunk_metadata.json
Output: data/qdrant_db/  (local Qdrant database)

Run from project root:
    python run_indexing.py

To force re-index (e.g. after adding more papers):
    python run_indexing.py --recreate
"""

import sys
from src.utils.logger import setup_logger, get_logger
from src.vectorstore.indexer import VectorIndexer

setup_logger()
logger = get_logger(__name__)


def main():
    recreate = "--recreate" in sys.argv

    logger.info("=" * 60)
    logger.info(f"PHASE 7 - VECTOR DATABASE INDEXING")
    logger.info("=" * 60)


    if recreate:
        logger.warning("--recreate flag set: existing index will be deleted")


    indexer = VectorIndexer()
    stats = indexer.run(recreate = recreate)


    logger.info("=" * 60)
    logger.info("INDEXING COMPLETE")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("=" * 60)



if __name__ == "__main__":
    main()