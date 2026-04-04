"""
Hybrid retriever combining dense (Qdrant) and sparse (BM25) search.

RECIPROCAL RANK FUSION (RRF) EXPLAINED:

Instead of trying to normalize scores across two completely different
scoring systems (cosine similarity vs BM25 score), RRF uses RANKS.

For each result, we compute:
    RRF_score = 1 / (k + rank_in_dense_results)
              + 1 / (k + rank_in_bm25_results)

Where k=60 is a constant that dampens the impact of very high ranks.

Example:
    Chunk A: rank 1 in dense, rank 3 in BM25
        RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

    Chunk B: rank 2 in dense, not in BM25
        RRF = 1/(60+2) + 0 = 0.0161

    Chunk C: rank 5 in dense, rank 1 in BM25
        RRF = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0317

Chunk A wins - it ranked highly in BOTH systems.
Chunk C is second - it was top in BM25 and decent in dense.

WHY RRF OVER SCORE NORMALIZATION:
    BM25 scores range 0-15 typically.
    Cosine similarity scores range 0-1.
    Normalizing these to the same scale requires knowing
    the distribution of each, which changes per query.
    RRF sidesteps this entirely by using ranks.
    
    This is why RRF is the industry standard for hybrid search.
"""

from typing import Optional
import numpy as np


from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.bm25_store import BM25Store
from src.embeddings.embedding_model import EmbeddingModel
from src.utils.logger import get_logger
from config.settings import TOP_K_RETRIEVAL


logger = get_logger(__name__)

# RRF constant - 60 is the standard value from the original paper
RRF_K = 60



class HybridRetriever:
    """
    Combines dense vector search and BM25 keyword search
    using Reciprocal Rank Fusion for score merging.
    """

    def __init__(
        self,
        qdrant_store:    QdrantStore,
        bm25_store:      BM25Store,
        embedding_model: EmbeddingModel,
    ):
        self.qdrant     = qdrant_store
        self.bm25       = bm25_store
        self.embedder   = embedding_model


    def retrieve(
        self,
        query:           str,
        top_k:           int = TOP_K_RETRIEVAL,
        filter_category: Optional[str] = None,
        filter_year_gte: Optional[int] = None,
        dense_weight:    float = 0.7,
        sparse_weight:   float = 0.3,
    ) -> list[dict]:
        """
        Hybrid retrieval with RRF fusion.

        Args:
            query:           User's raw query string
            top_k:           Final number of results to return
            filter_category: ArXiv category filter (e.g. "cs.LG")
            filter_year_gte: Only papers from this year onwards
            dense_weight:    Weight for dense retrieval in fusion (0-1)
            sparse_weight:   Weight for BM25 retrieval in fusion (0-1)

        Returns:
            List of result dicts sorted by RRF score (best first)

        WHY dense_weight = 0.7, sparse_weight = 0.3:
            Research papers use technical language where semantic
            understanding (dense) matters more than exact keyword
            matching (sparse). For a code search system, you'd
            flip these weights.
        """

        # -------------- Step 1: Dense retrieval --------------
        query_vector    = self.embedder.embed_query(query)
        dense_results   = self.qdrant.search(
            query_vector    = query_vector,
            top_k           = top_k * 2,    # Retrieve more for fusion
            filter_category = filter_category,
            filter_year_gte = filter_year_gte,
        )

        # -------------- Step 2: Sparse (BM25) retrieval --------------
        sparse_results = self.bm25.search(query, top_k = top_k * 2)


        # -------------- Step 3: Build chunk_id -> full data lookup --------------
        # Dense results have full payload (text, metadata)
        # Sparse results only have chunk_id and text

        chunk_data = {}
        
        # -------------------------------------------------------
        # for r in dense_results:
        #     if r["chunk_id"] not in chunk_data:
        #         chunk_data[r["chunk_id"]] = {
        #             "chunk_id": r["chunk_id"],
        #             "text":     r["text"],
        #             "score":    0.0,
        #         }
        # -------------------------------------------------------

        for r in dense_results:
            if r["chunk_id"] not in chunk_data:
                chunk_data[r["chunk_id"]] = {**r}


        # -------------- Step 4: Compute RRF score --------------
        RRF_scores = {}

        # Add dense ranks
        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            RRF_scores[cid] = RRF_scores.get(cid, 0.0)
            RRF_scores[cid] += dense_weight * (1.0 / (RRF_K + rank + 1))

        # Add sparse ranks
        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            RRF_scores[cid] = RRF_scores.get(cid, 0.0)
            RRF_scores[cid] += sparse_weight * (1.0 / (RRF_K + rank + 1))

        # -------------- Step 5: Sort by RRF score --------------
        sorted_ids = sorted(RRF_scores, key = RRF_scores.get, reverse = True)


        # -------------- Step 6: Build final results --------------
        final_results = []
        for cid in sorted_ids[:top_k]:
            data = chunk_data.get(cid, {})
            final_results.append(
                {
                    **data,
                    "rrf_score":    round(RRF_scores[cid], 6),
                    "retrieval":    "hybrid",
                }
            )

        logger.debug(
            f"Hybrid retrieval: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse -> {len(final_results)} merged"
        )

        return final_results