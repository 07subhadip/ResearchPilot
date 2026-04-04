"""
Orchestrates the full retrieval pipeline:
    1. Hybrid retrieval (dense + BM25)
    2. Cross-encoder re-ranking
    3. Diversity filtering

This is the component that the RAG pipeline (Phase 9) will call.
It takes a query string and returns the best chunks.
"""

from typing import Optional


from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, diversity_filter
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.bm25_store import BM25Store
from src.embeddings.embedding_model import EmbeddingModel
from src.utils.logger import get_logger
from config.settings import TOP_K_RETRIEVAL, TOP_K_RERANK


logger = get_logger(__name__)




class RetrievalPipeline:
    """
    Full retrieval pipeline with hybrid search + re-ranking.

    Usage:
        pipeline = RetrievalPipeline()
        results  = pipeline.retrieve("how does LoRA fine-tuning work?")
        for r in results:
            print(r["title"], r["ce_score"], r["text"][:100])
    """

    def __init__(self):
        # Initialize all components
        logger.info("Initializing RetrievalPipeline...")


        qdrant   = QdrantStore()
        embedder = EmbeddingModel()


        # Load or build BM25 index
        bm25 = BM25Store()
        if not bm25.load():
            logger.info("BM25 index not found - building now...") 
            bm25.build_index()

        self.hybrid_retriever = HybridRetriever(
            qdrant_store    = qdrant,
            bm25_store      = bm25,
            embedding_model = embedder,     
        )

        self.reranker = CrossEncoderReranker()

        logger.info("RetrievalPipeline ready")


    def retrieve(
        self,
        query:           str,
        top_k_final:     int = TOP_K_RERANK,
        filter_category: Optional[str] = None,
        filter_year_gte: Optional[int] = None,
    ) -> list[dict]:
        """
        Full retrieval: hybrid search → re-rank → diversity filter.

        Args:
            query:           User's natural language question
            top_k_final:     Number of chunks to return (default 5)
            filter_category: ArXiv category filter
            filter_year_gte: Year filter

        Returns:
            List of top chunks with all metadata and scores
        """
        logger.debug(f"Retrieving for query: '{query[:60]}'")

        # Stage 1: Hybrid retrieval → top-20 candidates
        candidates = self.hybrid_retriever.retrieve(
            query           = query,
            top_k           = TOP_K_RETRIEVAL * 2,   # 40 candidates
            filter_category = filter_category,
            filter_year_gte = filter_year_gte,
        )

        if not candidates:
            logger.warning(f"No candidates found for query: {query}")
            return []

        # Stage 2: Cross-encoder re-ranking -> top-5
        reranked = self.reranker.rerank(
            query   = query,
            results = candidates[:10],
            top_k   = TOP_K_RETRIEVAL * 2,  # Keep extra before diversity filter
        )

        # Stage 3: Diversity filter -> max 2 chunks per paper
        diverse = diversity_filter(reranked, max_per_paper=2)

        # Return top_k_final after diversity filtering
        final = diverse[:top_k_final]

        logger.debug(
            f"Pipeline: {len(candidates)} candidates -> "
            f"{len(reranked)} reranked -> "
            f"{len(final)} final"
        )

        return final