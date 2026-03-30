"""
Disk-based cache for computed embeddings.

PROBLEM WE'RE SOLVING:
    Embedding 15,664 chunks takes ~30-60 minutes on CPU.
    If you restart your pipeline or add 10 new papers,
    you don't want to re-embed the 15,654 unchanged chunks.

SOLUTION:
    Save embeddings to disk as numpy .npy files.
    Build an index that maps chunk_id -> array row index.
    On next run, load from disk instead of recomputing.

STORAGE FORMAT:
    data/embeddings/
    |-- embeddings.npy        <- numpy array, shape (N, 768)
    |-- chunk_ids.npy         <- chunk IDs in same order as rows  
    |-- embedding_index.json  <- metadata + chunk_id -> row mapping

WHY NUMPY .npy OVER JSON:
    Storing 15,664 * 768 floats as JSON = ~90MB of text
    Storing as .npy binary = ~46MB + loads 100x faster
"""

import json
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from config.settings import EMBEDDINGS_DIR, EMBEDDING_DIMENSION

logger = get_logger(__name__)



class EmbeddingCache:
    """
    Manages persistent storage of chunk embeddings
    """


    def  __init__(self):
        self.embedding_file  = EMBEDDINGS_DIR / "embeddings.npy"
        self.chunk_ids_file  = EMBEDDINGS_DIR / "chunk_ids.npy"
        self.index_file      = EMBEDDINGS_DIR / "embedding_index.json"


        # In-memory state
        self._embeddings: np.ndarray = None     # Shape (N, 768)
        self._chunk_ids: list[str]   = None     # length N   
        self._id_to_row:    dict     = None     # chunk_id -> row index


    def exists(self) -> bool:
        """Check if cached embeddings exists on disk"""
        return (
            self.embedding_file.exists() and 
            self.chunk_ids_file.exists() and
            self.index_file.exists()
        )


    def load(self) -> bool:
        """
        Load embeddings from disk into memory

        Returns True if loaded successfully. False if no cache exists
        """
        if not self.exists():
            logger.info("No embedding cache found on disk")
            return False

        logger.info("Loading embeddings from disk cache...")


        # Load numpy arrays - mmap_mode='r' means memory-mapped read
        # WHY mmap: The array is NOT fully loaded into RAM immediately
        # It's read from disk only when specific rows are accessed
        # This is critical for large arrays on machines with limited RAM
        self._embeddings = np.load(
            str(self.embedding_file),
            mmap_mode = 'r'
        )

        # chunk_ids are stored as numpy array of strings
        # We convert back to Python list for easier indexing
        self._chunk_ids = list(
            np.load(str(self.chunk_ids_file), allow_pickle = True)
        )

        # Build the reverse lookup: chunk_id -> row number
        self._id_to_row = {
            chunk_id: idx
            for idx, chunk_id in enumerate(self._chunk_ids)
        }

        logger.info(
            f"Cache loaded: {self._embeddings.shape[0]:,} embeddings"
            f"dimension = {self._embeddings.shape[1]}"
        )

        return True

    
    def save(self, embeddings: np.ndarray, chunk_ids: list[str]):
        """
        Save embeddings and their chunk IDs to disk.

        Args:
            embeddings: numpy array of shape (N, 768)
            chunk_ids:  list of N chunk ID strings (same order as rows)
        """

        assert len(embeddings) == len(chunk_ids), (
            f"Mismatch {len(embeddings)} embeddings vs {len(chunk_ids)} IDs"
        )

        logger.info(f"Saving {len(embeddings):,} embeddings to disk...")

        # Save the embedding matrix
        np.save(str(self.embedding_file), embeddings)  

        # Save chunk IDs as numpy object array (handles strings)
        np.save(str(self.chunk_ids_file), np.array(chunk_ids, dtype = object))

        # Save human-readable index file
        index = {
            "total_embeddings": len(embeddings),
            "embedding_dimension": embeddings.shape[1],
            "model_name": "BAAI/bge-base-en-v1.5",
            "chunk_id_sample": chunk_ids[:5],   # First 5 for verification
        }

        with open(self.index_file, "w", encoding = 'utf-8') as f:
            json.dump(index, f, indent = 2)



        # Update in-memory state
        self._embeddings = embeddings
        self._chunk_ids  = chunk_ids
        self._id_to_row  = {cid: i for i, cid in enumerate(chunk_ids)}


        logger.info(
            f"Saved embeddings: {self.embedding_file}"
            f"({self.embedding_file.stat().st_size / 1024 / 1024:.1f} MB)"
        )


    def get_embeddings(self, chunk_id: str) -> np.ndarray | None:
        """Get the embedding vector for a specific chunk ID."""
        if self._id_to_row is None:
            return None
        
        row = self._id_to_row.get(chunk_id)

        if row is None:
            return None
        
        return self._embeddings[row]



    def get_all(self) -> tuple[np.ndarray, list[str]]:
        """Return all embeddings and their chunk IDs."""
        return self._embeddings, self._chunk_ids

    
    @property
    def size(self) -> int:
        """Number of cached embeddings"""
        if self._chunk_ids is None:
            return 0

        return len(self._chunk_ids)