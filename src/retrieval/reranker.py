"""
Cross-encoder re-ranking for improved retrieval precision.

THE DIFFERENCE BETWEEN BI-ENCODER AND CROSS-ENCODER:

Bi-encoder (what BGE does):
    embed(query) → vector_q
    embed(chunk) → vector_c
    score = cosine(vector_q, vector_c)
    
    Query and chunk are embedded INDEPENDENTLY.
    Fast (vectors pre-computed), but loses interaction signal.

Cross-encoder (what we use for re-ranking):
    score = model(query + [SEP] + chunk)
    
    Query and chunk are processed TOGETHER by the model.
    The model can see how query tokens relate to chunk tokens.
    Slower (cannot pre-compute), but much more accurate.

THE TWO-STAGE PATTERN:
    Stage 1 (Retrieval):   Bi-encoder -> top-20 candidates (fast, approximate)
    Stage 2 (Re-ranking):  Cross-encoder -> re-score top-20 (slow, accurate)
    
    We only run the expensive cross-encoder on 20 candidates,
    not all 15,664 chunks. This gives us accuracy without
    paying the full cost for every chunk.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage retrieval dataset (500K+ queries)
    - MiniLM architecture: fast on CPU
    - Output: relevance score (-inf to +inf, higher = more relevant)
    - Size: ~80MB
"""

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import CrossEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.
    """


    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model      = None
        self._model_name = model_name
        logger.info(f"CrossEncoderReranker initialized: {model_name}")

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load cross-encoder model.""" 
        if self._model is None:
            logger.info(f"Loading cross-encoder: {self._model_name}")
            self._model = CrossEncoder(
                self._model_name,
                max_length = 512    # Max tokens for query+chunk combined
            )
            logger.info("Cross-encoder loaded")

        return self._model


    def rerank(
        self,
        query:      str,
        results:    list[dict],
        top_k:      int = 5
    ) -> list[dict]:
        """
        Re-rank a list of retrieved chunks using cross-encoder scoring.

        Args:
            query:   Original user query
            results: List of retrieved chunk dicts (from hybrid retriever)
            top_k:   How many top results to return after re-ranking

        Returns:
            Top-k results sorted by cross-encoder relevance score

        WHAT THE CROSS-ENCODER SEES:
            Input: "[CLS] how does attention work? [SEP] The transformer
                    architecture uses scaled dot-product attention where
                    queries, keys and values are computed... [SEP]"
            Output: 8.3  (high relevance)

            vs.

            Input: "[CLS] how does attention work? [SEP] UAV delivery
                    systems require multi-agent coordination... [SEP]"
            Output: -2.1  (low relevance)

        The model learned these relevance patterns from 500K+
        human-labeled query-passage pairs in MS MARCO.
        """

        if not results:
            return []

        # Build (query, chunk_text) pairs for batch scoring
        pairs = [
            (query, r.get("text", ""))
            for r in results
        ]

        # Score all pairs in one batch
        # predict() returns numpy array of relevance scores
        scores = self.model.predict(
            pairs,
            show_progress_bar = False,
            batch_size = 32,
        )

        # Attach cross_encoder score to each result
        for result, score in zip(results, scores):
            result["ce_score"] = round(float(score), 4)

        # Sort by cross-encoder score (descending)
        reranked = sorted(results, key = lambda x: x["ce_score"], reverse = True)

        logger.debug(
            f"Re-ranked {len(results)} -> top-{top_k}. "
            f"Score range: [{reranked[-1]["ce_score"]:.2f}, "
            f"{reranked[0]["ce_score"]:.2f}]"
        )


        return reranked[:top_k]



def diversity_filter(results: list[dict], max_per_paper: int = 2) -> list[dict]:
    """
    Ensure no single paper dominates the results.

    As you saw in test_search.py - the same paper appeared twice
    in top-3. This function limits results to max_per_paper
    chunks from any single paper.

    Args:
        results:       List of result dicts (sorted by relevance)
        max_per_paper: Maximum chunks allowed from the same paper

    Returns:
        Filtered list maintaining original relevance order

    WHY THIS MATTERS FOR USER EXPERIENCE:
        User asks: "how does attention work?"
        Without diversity filter: 3 chunks from same attention paper
        With diversity filter: 1-2 chunks each from 3 different papers

        The second response is richer - multiple perspectives,
        multiple research groups, more comprehensive coverage.
    """

    seen_papers: dict[str, int] = {}
    filtered = []

    for result in results:
        paper_id    = result.get("paper_id", "unknown")
        count       = seen_papers.get(paper_id, 0)

        if count < max_per_paper:
            filtered.append(result)
            seen_papers[paper_id] = count + 1

    
    return filtered