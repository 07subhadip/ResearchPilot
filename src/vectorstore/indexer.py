"""
Loads embeddings + chunks from disk and indexes them into Qdrant.

This is a ONE-TIME operation (or run when new papers are added).
After this, all searches go through Qdrant - not numpy arrays.
"""

import json
import numpy as np
from pathlib import Path

from src.vectorstore.qdrant_store import QdrantStore
from src.embeddings.embedding_cache import EmbeddingCache
from src.utils.logger import get_logger
from config.settings import CHUNKS_DIR, EMBEDDINGS_DIR

logger = get_logger(__name__)



class VectorIndexer:
    """Orchestrates loading embeddings and indexing into Qdrant"""

    def __init__(self):
        self.store = QdrantStore()
        self.cache = EmbeddingCache()


#----------------------------------------------------------------------------------------------------------

    # def load_texts_by_chunk_id(self, chunk_ids: list[str]) -> dict[str, str]:
    #     """
    #     Build a lookup dict: chunk_id → chunk text.

    #     We need this because EmbeddingCache stores embeddings
    #     but not the original texts. We reload texts from chunk files.
    #     """
    #     # Load the metadata file which has all chunk info
    #     metadata_path = EMBEDDINGS_DIR / "chunk_metadata.json"

    #     if metadata_path.exists():
    #         with open(metadata_path, "r", encoding = 'utf-8') as f:
    #             metadata_list = json.load(f)

    #         logger.info(f"Loaded metadata for {len(metadata_list):,} chunks")
    #         return metadata_list

    #     # Fallback: reload from chunk files (slower)
    #     logger.warning("chunk_metadata.json not found, loading from chunk files...")
    #     id_to_text = {}
    #     for cf in CHUNKS_DIR.glob("*_semantic.json"):
    #         with open(cf, 'r', encoding = 'utf-8') as f:
    #             chunks = json.load(f)
    #         for c in chunks:
    #             id_to_text[c['chunk_id']] = c['text']
        
    #     return id_to_text

#----------------------------------------------------------------------------------------------------------



    def load_chunk_from_disk(self) -> tuple[list[str], list[str], list[str]]:
        """
        Load chunk texts and metadata directly from chunk files.
        This is the ground truth source - chunk files have everything.
        
        Returns:
            chunk_ids: list of chunk ID strings
            texts:     list of chunk text strings  
            metadata:  list of metadata dicts (without text)
        """
        chunk_ids = []
        texts     = []
        metadata  = []


        chunk_files = list(CHUNKS_DIR.glob("*_semantic.json"))
        logger.info(f"Loading chunks from {len(chunk_files)} files...")

        for cf in chunk_files: 
            with open(cf, 'r', encoding = "utf-8") as f:
                raw = json.load(f)

            # Handle both formats:
            #   Old local format: [{chunk_id: ..., text: ...}, ...]
            #   New Kaggle format: {"paper_id": "...", "chunks": [...]}
            if isinstance(raw, dict) and "chunks" in raw:
                chunk_list = raw["chunks"]
            elif isinstance(raw, list):
                chunk_list = raw
            else:
                logger.warning(f"Unexpected format in {cf.name}, skipping")
                continue

            for chunk in chunk_list:
                chunk_ids.append(chunk['chunk_id'])
                texts.append(chunk["text"])

                # Everything except text goes into metadata
                metadata.append(
                    {
                        k: v for k, v in chunk.items()
                        if k != "text"
                    }
                )

        logger.info(f"Loaded {len(chunk_ids):,} chunks from disk")
        return chunk_ids, texts, metadata




    def run(self, recreate: bool = False) -> dict:
        """
        Full indexing pipeline.

        Args:
            recreate: Delete existing collection and re-index everything.
                      Set True when you change embedding model or chunking.

        Returns:
            Indexing statistics
        """
        # Check if already exists
        current_size = self.store.get_collection_size()

        if current_size > 0 and not recreate:
            logger.info(
                f"Collection already has {current_size:,} points. "
                f"Run with recreate=True to re-index."
            )

            return {
                "status": "already_indexed",
                "points": current_size,
            }


        # Step 1: Load directly from chunk files - ground truth source
        # (chunk files have text + metadata, and are the source of truth)
        chunk_ids, texts, metadata = self.load_chunk_from_disk()

        # Step 2: Create the Qdrant collection (skips if already exists)
        self.store.create_collection(recreate=recreate)

        # Step 3: Load embeddings from cache and reorder to match chunk order from disk
        # (cache order may differ from disk order, so we align by chunk_id)
        logger.info("Loading embeddings from cache...")
        self.cache.load()
        embeddings_matrix, cached_ids = self.cache.get_all()

        # Build a lookup dict: chunk_id → row index in embedding matrix
        id_to_row = {cid: i for i, cid in enumerate(cached_ids)}

        # Reorder embeddings so they match the chunk_ids order we loaded from disk
        ordered_embeddings = np.array([
            embeddings_matrix[id_to_row[cid]]
            for cid in chunk_ids
            if cid in id_to_row      # only include chunks that have an embedding
        ])

        # Filter chunk_ids, texts, metadata to only those that have a matching embedding
        # (some chunks may have been added after last embedding run)
        valid_indices = [i for i, cid in enumerate(chunk_ids) if cid in id_to_row]
        chunk_ids     = [chunk_ids[i] for i in valid_indices]
        texts         = [texts[i]     for i in valid_indices]
        metadata      = [metadata[i]  for i in valid_indices]

        logger.info(f"Matched {len(chunk_ids):,} chunks with embeddings")

        # Step 4: Index everything into Qdrant
        logger.info(f"Indexing {len(chunk_ids):,} chunks into Qdrant...")
        total = self.store.index_chunks(
            embeddings = ordered_embeddings,
            chunk_ids  = chunk_ids,
            metadata   = metadata,
            texts      = texts,
        )

        stats = {
            "status":          "complete",
            "chunks_indexed":  total,
            "collection_info": self.store.get_collection_info(),
        }

        logger.info(f"Indexing completed: {stats}")
        return stats
