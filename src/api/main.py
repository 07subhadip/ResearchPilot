"""
ResearchPilot FastAPI application.

STARTUP BEHAVIOR:
    When the server starts, it loads ALL models into memory:
    - BGE embedding model (~110MB)
    - Cross-encoder re-ranker (~80MB)
    - BM25 index (~40MB)
    - Qdrant connection

    This takes ~15 seconds once, then every request is fast.
    This is called "warm start" - the model is always ready.

    Without this, the first request after server restart
    would take 20+ seconds. Unacceptable for production.

LIFESPAN PATTERN:
    FastAPI's lifespan context manager runs code at startup
    and shutdown. We use it to initialize the RAG pipeline
    once and store it in app.state for all requests to share.
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json
import os

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    CitationSchema,
    HealthResponse,
    ErrorResponse,
)

class FeedbackRequest(BaseModel):
    query: str
    rating: int
    thumbs: str | None = None
    comment: str
    model_used: str
    citations_count: int
    total_time_ms: float
from src.rag.pipeline import RAGPipeline
from src.utils.logger import setup_logger, get_logger


setup_logger()
logger = get_logger(__name__)


# ---------------------------------------------------------
# LIFESPAN - runs at startup and shutdown
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize resources at startup, clean up at shutdown.

    The 'yield' separates startup (before) from shutdown (after).
    Everything before yield runs when server starts.
    Everything after yield runs when server shuts down.
    """

    # --------------- STARTUP ---------------
    logger.info("ResearchPilot API starting up...")
    start = time.time()

    # Initialize RAG pipeline - loads all models into memory
    # We store it on app.state so all request handlers can access it
    app.state.rag_pipeline = RAGPipeline()

    elapsed = time.time() - start
    logger.info(f"API ready in {elapsed:.1f}s")

    yield   # Server is now running and handling requests

    # --------------- SHUTDOWN ---------------
    logger.info("ResearchPilot API shutting down...")


# ---------------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------------

app = FastAPI(
    title       = "ResearchPilot API",
    description = "Production RAG system for ML research paper Q&A",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",    # Swagger UI at http://localhost:8000/docs
    redoc_url   = "/redoc",   # ReDoc at http://localhost:8000/redoc
)

# CORS middleware — allows browser-based frontends to call this API
# Without this, a browser on localhost:3000 cannot call localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # In production, restrict to your domain
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ---------------------------------------------------------
# EXCEPTION HANDLER
# ---------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exception and return a clean JSON error.
    Without this, FastAPI returns a raw 500 error with no detail.
    """
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":  "Internal server error",
            "detail": str(exc),
            "code":   500,
        }
    )


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
    tags           = ["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """
    Returns system health status.
    Used by deployment platforms to verify the service is running.
    Also useful for debugging - shows database sizes.
    """
    pipeline = request.app.state.rag_pipeline

    # Get Qdrant collection size
    qdrant_size = pipeline.retriever.hybrid_retriever.qdrant.get_collection_size()

    # Get BM25 index size
    bm25_size = len(pipeline.retriever.hybrid_retriever.bm25.chunk_ids)

    return HealthResponse(
        status           = "healthy",
        model            = "llama-3.3-70b-versatile",
        vector_db_size   = qdrant_size,
        bm25_index_size  = bm25_size,
        version          = "1.0.0",
    )

@app.post(
    "/query/stream",
    summary        = "Stream query research papers",
    tags           = ["RAG"],
)
async def stream_query_papers(
    request:     Request,
    query_input: QueryRequest,
):
    import asyncio
    pipeline = request.app.state.rag_pipeline

    async def async_generator():
        """
        Wraps the synchronous pipeline.stream_query() generator in an
        async-friendly way using a thread + asyncio.Queue so we never
        block the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def run_sync():
            try:
                for chunk in pipeline.stream_query(
                    question        = query_input.question,
                    top_k           = query_input.top_k,
                    filter_category = query_input.filter_category,
                    filter_year_gte = query_input.filter_year_gte,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        import threading
        thread = threading.Thread(target=run_sync, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            yield item

    return StreamingResponse(
        async_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.post(
    "/feedback",
    summary        = "Submit feedback",
    tags           = ["System"],
)
async def submit_feedback(feedback: FeedbackRequest):
    os.makedirs("logs", exist_ok=True)
    with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback.model_dump()) + "\n")
    return {"status": "ok"}

@app.post(
    "/query",
    response_model = QueryResponse,
    summary        = "Query research papers",
    tags           = ["RAG"],
)
async def query_papers(
    request:     Request,
    query_input: QueryRequest,
) -> QueryResponse:
    """
    Submit a natural language question about ML research.

    The system retrieves relevant paper excerpts and generates
    a grounded answer with citations.

    - **question**: Your research question (3-500 characters)
    - **top_k**: Number of paper chunks to retrieve (1-20, default 5)
    - **filter_category**: Filter by ArXiv category (e.g. cs.LG)
    - **filter_year_gte**: Only include papers from this year onwards
    """
    pipeline = request.app.state.rag_pipeline

    logger.info(
        f"Query received: '{query_input.question[:60]}' "
        f"[top_k={query_input.top_k}]"
    )

    # Run the RAG pipeline in a thread pool
    # WHY asyncio.to_thread:
    #   Our RAG pipeline is CPU-bound (not async).
    #   Running it directly in an async handler would BLOCK
    #   the entire FastAPI event loop - no other requests
    #   could be processed while one query is running.
    #   asyncio.to_thread runs it in a separate thread,
    #   keeping the event loop free for other requests.
    try:
        response = await asyncio.to_thread(
            pipeline.query,
            query_input.question,
            query_input.top_k,
            query_input.filter_category,
            query_input.filter_year_gte,
        )
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Convert RAGResponse dataclass to API schema
    citations = [
        CitationSchema(
            paper_id       = c.get("paper_id", ""),
            title          = c.get("title", ""),
            authors        = c.get("authors", []),
            published_date = c.get("published_date", ""),
            arxiv_url      = c.get("arxiv_url", ""),
        )
        for c in response.citations
    ]

    return QueryResponse(
        answer             = response.answer,
        citations          = citations,
        query              = response.query,
        chunks_used        = len(response.retrieved_chunks),
        retrieval_time_ms  = response.retrieval_time_ms,
        generation_time_ms = response.generation_time_ms,
        total_time_ms      = response.total_time_ms,
        has_context        = response.has_context,
    )


@app.get(
    "/",
    summary = "API root",
    tags    = ["System"],
)
async def root():
    """API root - confirms service is running."""
    return {
        "service": "ResearchPilot API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }