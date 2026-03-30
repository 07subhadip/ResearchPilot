"""
BGE embedding model wrapper for ResearchPilot.

RESPONSIBILITIES:
    1. Load and cache the BGE-base-en-v1.5 model
    2. Embed document chunks (no prefix)
    3. Embed user queries (with BGE instruction prefix)
    4. Handle batching for large-scale embedding

WHY A WRAPPER CLASS instead of calling SentenceTransformer directly:
    If we decide to swap BGE for a better model tomorrow, we change
    ONE file. Nothing else in the codebase changes. This is called
    the FACADE PATTERN - hide implementation behind a stable interface
"""

import logging
# Suppress noisy sentence-transformers logs
logging.getLogger("sentence-transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
from typing import Union

from src.utils.logger import get_logger
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION



logger = get_logger(__name__)




class EmbeddingModel:
    """
    Wrapper around BGE-base-en-v1.5 for document and query embedding.

    Usage:
        model = EmbeddingModel()
        
        # Embed chunks (documents)
        chunk_vectors = model.embed_documents(["chunk text 1", "chunk text 2"])
        
        # Embed a user query
        query_vector = model.embed_query("what is attention mechanism?")
    """

    # BGE introduction prefix for queries
    # This is specified in the official BGE model card
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model     = None  # Lazy loaded
        logger.info(f"EmbeddingModel wrapper created for: {model_name}")


    @property
    def model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                f"Model loaded. "
                f"Embedding dimension: {self._model.get_sentence_embedding_dimension()}"
            )

        return self._model


    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of document chunks.

        NO prefix applied - BGE embeds documents as-is.

        Args:
            texts:         List of chunk texts to embed
            batch_size:    How many chunks to process at once
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), 768)
            Each row is the embedding for one chunk.

        BATCHING EXPLAINED:
            We cannot embed all 15,664 chunks at once - that would
            require ~15,664 * 768 * 4 bytes = ~48MB just for the
            output array, plus the model's working memory.
            
            Processing in batches of 32-64 keeps memory stable
            while still being fast (model processes the batch
            as a single matrix multiplication).
        """

        if not texts:
            return np.array([])

        
        logger.info(f"Embedding {len(texts)} documents in batches of {batch_size}")


        embeddings = self.model.encode(
            texts,
            batch_size           = batch_size,
            show_progress_bar    = show_progress,
            normalize_embeddings = True,    # L2 normalize -> cosine sim = dot product
            convert_to_numpy = True, 
        )


        logger.info(f"Embedding complete. Shape: {embeddings.shape}")

        return embeddings


    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query WITH the BGE instruction prefix.

        Args:
            query: Raw user question

        Returns:
            numpy array of shape (768,)

        WHY SINGLE QUERY (not batch):
            At query time, we receive one question at a time.
            Batching makes no sense here - we want the answer fast.
        """

        # Apply BGE's instruction prefix for retrieval queries
        prefixed_query = self.QUERY_PREFIX + query

        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings    = True,
            convert_to_numpy        = True,
            show_progress_bar       = False,
        )

        return embedding


    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Embed texts in batches, yielding one batch at a time.

        WHY A GENERATOR:
            For 15,664 chunks, we don't want to hold ALL embeddings
            in memory while also saving them. This generator yields
            one batch at a time - we save each batch, then free memory.

        Usage:
            for batch_embeddings, batch_texts in model.embed_batch(texts):
                save(batch_embeddings)
        """

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings    = True,
                convert_to_numpy        = True,
                show_progress_bar       = False,
            )

            yield embeddings, batch