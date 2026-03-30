"""
Orchestrates embedding generation for all chunks.

FLOW:
    1. Load all chunk files from data/chunks/
    2. Check cache - skip already-embedded chunks
    3. Embed remaining chunks in batches
    4. Save to cache
    5. Report statistics
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_cache import EmbeddingCache
from src.utils.logger import get_logger
from config.settings import CHUNKS_DIR, EMBEDDING_BATCH_SIZE


logger = get_logger(__name__)



class EmbeddingPipeline:
    """
    Loads all semantic chunks and generates embeddings for them
    """


    def __init__(self):
        self.model = EmbeddingModel()
        self.cache = EmbeddingCache()


    def load_all_chunks(self) -> tuple[list[str], list[str], list[dict]]:
        """
        Load all chunk texts, IDs, and metadata from disk.

        Returns:
            texts:     List of chunk text strings
            chunk_ids: List of chunk ID strings (same order)
            metadata:  List of chunk metadata dicts (same order)
        """
        chunk_file = list(CHUNKS_DIR.glob("*semantic.json"))
        logger.info(f"Loading chunks from {len(chunk_file)} files...")


        texts     = []
        chunk_ids = [] 
        metadata  = []

        for cf in tqdm(chunk_file, desc = "Loading chunks"):
            with open(cf, "r", encoding = 'utf-8') as f:
                chunks = json.load(f)


            for chunk in chunks:
                texts.append(chunk["text"])
                chunk_ids.append(chunk["chunk_id"])
                metadata.append(
                    {
                        k: v for k, v in chunk.items()
                        if k != "text"      # Don't duplicate text in metadata
                    }
                )


        logger.info(f"Loaded {len(texts):,} chunks total")
        return texts, chunk_ids, metadata


    def run(self) -> dict:
        """
        Main pipeline: embed all chunks and save to cache.

        Returns:
            Statistics dictionary
        """
        # Load all chunks from disk
        texts, chunk_ids, metadata = self.load_all_chunks()

        if not texts:
            logger.error("No chunks found. Run run_chunking.py first.")
            return {}

        
        # Check if we already have a complete cache
        if self.cache.exists():
            self.cache.load()
            if self.cache.size == len(texts):
                logger.info(
                    f"Cache complete: {self.cache.size:,} embeddings already exist."
                    f"Nothing to do."
                ) 

                return {
                    "total": len(texts),
                    "embedded": 0,
                    "from_cache": self.cache.size,
                    "status": "cache_hit"
                }
            else:
                logger.info(
                    f"Partial cache: {self.cache.size:,} / {len(texts):,} "
                    f"Re-embedding all for consistency."
                )

        # Embed all chunks
        logger.info(f"Embedding {len(texts):,} chunks with BGE-base-en-v1.5...")
        logger.info(f"Batch size: {EMBEDDING_BATCH_SIZE}")
        logger.info(
            f"Estimated time: "
            f"{len(texts) / EMBEDDING_BATCH_SIZE * 0.5:.0f} seconds on CPU"
        )


        # embed_documents handles batching internally and shows progress bar
        embeddings = self.model.embed_documents(
            texts,
            batch_size    = EMBEDDING_BATCH_SIZE,
            show_progress = True,
        )


        # Verify shape
        assert embeddings.shape == (len(texts), 768), (
            f"Expected ({len(texts)}, 768), got {embeddings.shape}"
        )

        # Save to disk
        self.cache.save(embeddings, chunk_ids)

        # Also save metadata separately (needed for Qdrant in Phase 7)
        metadata_path = CHUNKS_DIR.parent / "embeddings" / "chunk_metadata.json"
        with open(metadata_path, "w", encoding = 'utf-8') as f:
            json.dump(metadata, f, ensure_ascii = False)
        
        logger.info(f"Metadata saved to {metadata_path}")


        stats = {
            "total_chunks":       len(texts),
            "embedding_shape":    list(embeddings.shape),
            "embedding_dim":      embeddings.shape[1],
            "cache_size_mb":      round(
                embeddings.nbytes / 1024 / 1024, 1
            ),
            "status": "complete"
        }

        logger.info(f"Embedding pipeline completed: {stats}")
        
        return stats