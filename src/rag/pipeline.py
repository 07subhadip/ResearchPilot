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
import json
from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.rag.llm_client import MultiModelClient
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
    answer:     str
    citations:  list[dict]
    retrieved_chunks: list[dict]
    query:      str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms:  float
    has_context:    bool
    model_used:     str = ""


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
            "model_used":         self.model_used,
        }

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAGPipeline...")
        self.retriever  = RetrievalPipeline()
        self.llm        = MultiModelClient()
        logger.info("RAGPipeline ready")

    def query(
        self,
        question:        str,
        top_k:           int = TOP_K_RERANK,
        filter_category: Optional[str] = None,
        filter_year_gte: Optional[int] = None,
    ) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        total_start = time.time()
        retrieval_start = time.time()

        chunks = self.retriever.retrieve(
            query           =  question,
            top_k_final     = top_k,
            filter_category = filter_category,
            filter_year_gte = filter_year_gte,
        )
        retrieval_ms = (time.time() - retrieval_start) * 1000
        has_context = len(chunks) > 0

        if has_context:
            user_prompt = build_rag_prompt(question, chunks)
        else:
            user_prompt = (
                f"The user asked: {question}\n\n"
                f"No relevant research papers were found in the database. "
                f"Politely inform the user and suggest they try rephrasing "
                f"or broadening their query."
            )

        generation_start = time.time()
        answer, model_used = self.llm.generate(
            system_prompt = SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            original_query = question,
            stream=False
        )

        generation_ms   = (time.time() - generation_start) * 1000
        total_ms        = (time.time() - total_start) * 1000
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
            model_used         = model_used
        )

    def stream_query(
        self,
        question:        str,
        top_k:           int = TOP_K_RERANK,
        filter_category: Optional[str] = None,
        filter_year_gte: Optional[int] = None,
    ):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        total_start = time.time()
        retrieval_start = time.time()
        chunks = self.retriever.retrieve(
            query           =  question,
            top_k_final     = top_k,
            filter_category = filter_category,
            filter_year_gte = filter_year_gte,
        )
        retrieval_ms = (time.time() - retrieval_start) * 1000
        has_context = len(chunks) > 0

        if has_context:
            user_prompt = build_rag_prompt(question, chunks)
        else:
            user_prompt = (
                f"The user asked: {question}\n\n"
                f"No relevant research papers were found in the database. "
                f"Politely inform the user and suggest they try rephrasing "
                f"or broadening their query."
            )

        generation_start = time.time()
        generator, model_used = self.llm.generate(
            system_prompt = SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            original_query = question,
            stream=True
        )

        for token in generator:
            yield f"data: {json.dumps({'token': token})}\n\n"

        generation_ms = (time.time() - generation_start) * 1000
        total_ms = (time.time() - total_start) * 1000
        citations = build_citation_list(chunks)

        metadata = {
            "done": True,
            "citations": citations,
            "model_used": model_used,
            "timing": {
                "retrieval_time_ms": round(retrieval_ms, 1),
                "generation_time_ms": round(generation_ms, 1),
                "total_time_ms": round(total_ms, 1),
                "chunks_used": len(chunks)
            }
        }
        yield f"data: {json.dumps(metadata)}\n\n"