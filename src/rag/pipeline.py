"""
The complete RAG pipeline - orchestrates retrieval + generation.

This is the core of ResearchPilot. Every user query goes through this.

PIPELINE FLOW:
    1. Validate and clean the query
    2. Retrieve top-5 relevant chunks (Phase 8 pipeline)
    3. Build prompt with context
    4. Generate answer via Groq LLM
    5. Parse and structure the response
    6. Return answer + citations + metadata
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.rag.llm_client import LLMClient
from src.rag.prompt_templates import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    build_citation_list,
)
from src.utils.logger import get_logger
from config.settings import TOP_K_RERANK

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """
    Structured response from the RAG pipeline.

    WHY A DATACLASS INSTEAD OF A DICT:
        Dicts can have any keys - you never know what's in them.
        A dataclass defines the exact contract. The FastAPI layer
        (Phase 11) and frontend (Phase 12) can rely on these
        fields always being present.
    """
    # The generated answer
    answer:     str

    # Source papers used to generate the answer
    citations:  list[dict]

    # Raw retrieved chunks (for debugging / evaluation)
    retrieved_chunks: list[dict]

    # Performance metadata
    query:      str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms:  float

    # Whether retrieval found retrieval content
    has_context:    bool


    def to_dict(self) -> dict:
        return {
            "answer":             self.answer,
            "citations":          self.citations,
            "query":              self.query,
            "retrieval_time_ms":  round(self.retrieval_time_ms, 1),
            "generation_time_ms": round(self.generation_time_ms, 1),
            "total_time_ms":      round(self.total_time_ms, 1),
            "has_context":        self.has_context,
            "chunks_used":        len(self.retrieved_chunks),
        }




class RAGPipeline:
    """
    End-to-end RAG pipeline: query -> retrieve -> generate -> respond.

    Usage:
        pipeline = RAGPipeline()
        response = pipeline.query("How does LoRA reduce training parameters?")
        print(response.answer)
        for cite in response.citations:
            print(cite["title"], cite["arxiv_url"])
    """

    def __init__(self):
        logger.info("Initializing RAGPipeline...")

        self.retriever  = RetrievalPipeline()
        self.llm        = LLMClient()

        logger.info("RAGPipeline ready")

    def query(
        self,
        question:        str,
        top_k:           int = TOP_K_RERANK,
        filter_category: Optional[str] = None,
        filter_year_gte: Optional[int] = None,
    ) -> RAGResponse:
        """
        Process a user question through the full RAG pipeline.

        Args:
            question:        User's natural language question
            top_k:           Number of chunks to retrieve
            filter_category: Optional ArXiv category filter
            filter_year_gte: Optional year filter

        Returns:
            RAGResponse with answer, citations, and timing metadata
        """
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        total_start = time.time()

        # ------------ Stage 1: Retrieval ------------
        retrieval_start = time.time()

        chunks = self.retriever.retrieve(
            query           =  question,
            top_k_final     = top_k,
            filter_category = filter_category,
            filter_year_gte = filter_year_gte,
        )

        retrieval_ms = (time.time() - retrieval_start) * 1000

        logger.info(
            f"Retrieved: {len(chunks)} chunks in {retrieval_ms:.0f}ms"
        )

        has_context = len(chunks) > 0

        # ------------ Stage 2: Prompt Construction ------------
        if has_context:
            user_prompt = build_rag_prompt(question, chunks)
        else:
            # Fallback prompt when no relevant context found
            user_prompt = (
                f"The user asked: {question}\n\n"
                f"No relevant research papers were found in the database. "
                f"Politely inform the user and suggest they try rephrasing "
                f"or broadening their query."
            )

        # ------------ Stage 3: LLM Generation ------------
        generation_start = time.time()

        answer = self.llm.generate(
            system_prompt = SYSTEM_PROMPT,
            user_prompt   = user_prompt,
        )

        generation_ms   = (time.time() - generation_start) * 1000
        total_ms        = (time.time() - total_start) * 1000

        logger.info(
            f"Generated answer in {generation_ms:.0f}ms | "
            f"Total: {total_ms:.0f}ms"
        )

        # ------------ Stage 4: Build Citations ------------
        citations = build_citation_list(chunks)

        return RAGResponse(
            answer             = answer,
            citations          = citations,
            retrieved_chunks   = chunks,
            query              = question,
            retrieval_time_ms  = retrieval_ms,
            generation_time_ms = generation_ms,
            total_time_ms      = total_ms,
            has_context        = has_context,
        )