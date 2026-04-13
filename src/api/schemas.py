"""
Pydantic schemas for API request and response validation.

WHY PYDANTIC SCHEMAS IN THE API LAYER:
    FastAPI uses these to:
    1. Validate incoming requests (wrong types -> automatic 422 error)
    2. Serialize outgoing responses (Python objects -> JSON)
    3. Generate automatic API documentation (OpenAPI/Swagger)
    
    You get input validation AND documentation for free.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ConversationTurnSchema(BaseModel):
    role:     str
    content:  str
    citations: list = []



class QueryRequest(BaseModel):
    """
    Schema for POST /query request body.
    
    Field() lets us add validation constraints and documentation.
    """
    question: str = Field(
        ...,                    # ... means required
        min_length      = 3,
        max_length      = 500,
        description     = "Research question to answer",
        examples        = ["How does LoRA reduce trainable parameters?"]
    )
    history: List[ConversationTurnSchema] = Field(
        default=[],
        description="Conversation history for context"
    )
    top_k: int = Field(
        default     = 5,
        ge          = 1,                 # ge = greater than or equal
        le          = 20,
        description = "Number of chunks to retrieve"
    )
    filter_category: Optional[str] = Field(
        default     = None,
        description = "ArXiv category filter, e.g. 'cs.LG'",
        example     = ["cs.LG"]
    )
    filter_year_gte: Optional[int] = Field(
        default     = None,
        ge          = 2020,
        le          = 2030,
        description = "Only include papers from this year onwards",
        example     = [2024]
    )


class CitationSchema(BaseModel):
    """A single cited paper."""
    paper_id:       str
    title:          str
    authors:        list[str]
    published_date: str
    arxiv_url:      str


class QueryResponse(BaseModel):
    """Schema for POST /query response."""
    answer:             str
    citations:          list[CitationSchema]
    query:              str
    chunks_used:        int
    retrieval_time_ms:  float
    generation_time_ms: float
    total_time_ms:      float
    has_context:        bool


class HealthResponse(BaseModel):
    """Schema for GET /health response."""
    status:          str
    model:           str
    vector_db_size:  int
    bm25_index_size: int
    version:         str = "1.0.0"


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error:   str
    detail:  str
    code:    int