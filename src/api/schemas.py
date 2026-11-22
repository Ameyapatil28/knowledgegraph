"""Pydantic request and response schemas for TMDB Knowledge Graph FastAPI endpoints."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    graph_loaded: bool = Field(..., example=True)
    graph_nodes: Optional[int] = Field(None, example=124700)
    graph_edges: Optional[int] = Field(None, example=311242)
    gemini_configured: bool = Field(..., example=True)
    gemini_model: str = Field(..., example="gemini-2.5-flash")
    gemini_mode: str = Field(..., example="LIVE_GEMINI")  # LIVE_GEMINI or MOCK
    pricing_tier: str = Field(..., example="paid_standard")


class AskQuestionRequest(BaseModel):
    """Natural language question request schema."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question about movies, cast, crew, genres, ratings, or comparisons.",
        example="Tell me about The Dark Knight",
    )


class AnswerMetadata(BaseModel):
    """Observability, telemetry, and execution metadata for Q&A."""
    parser_provider: str = Field(..., example="LIVE_GEMINI")
    answer_provider: str = Field(..., example="LIVE_GEMINI")
    model: str = Field(..., example="gemini-2.5-flash")
    pricing_tier: str = Field(..., example="paid_standard")
    total_latency_ms: float = Field(..., example=452.1)
    parser_latency_ms: float = Field(..., example=150.2)
    engine_latency_ms: float = Field(..., example=8.5)
    answer_latency_ms: float = Field(..., example=293.4)
    tokens: Optional[Dict[str, Any]] = Field(None, example={"prompt_tokens": 150, "completion_tokens": 60, "total_tokens": 210})
    cost_usd: Optional[float] = Field(None, example=0.000195)
    grounding_passed: bool = Field(..., example=True)
    grounding_repaired: bool = Field(..., example=False)
    grounding_violations: List[str] = Field(default_factory=list)


class AskQuestionResponse(BaseModel):
    """Natural language answer response schema."""
    status: str = Field(
        ...,
        description="Response status: SUCCESS, NO_RESULTS, CLARIFICATION_REQUIRED, UNSUPPORTED, GENERATION_ERROR, GROUNDING_VIOLATION",
        example="SUCCESS",
    )
    question: str = Field(..., example="Tell me about The Dark Knight")
    answer: str = Field(..., example="The Dark Knight was released in 2008. Directed by Christopher Nolan...")
    operation: Optional[str] = Field(None, example="movie_details")
    results: Optional[Any] = Field(None, description="Raw structured results from the Knowledge Graph query engine.")
    metadata: Optional[AnswerMetadata] = None
    error: Optional[Dict[str, str]] = None


class StructuredQueryRequest(BaseModel):
    """Direct structured query execution request."""
    operation: str = Field(
        ...,
        description="Allowed query operation from SUPPORTED_OPERATIONS",
        example="top_movies",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters strictly matching allowed parameters",
        example={"genre_name": "Science Fiction", "limit": 5, "metric": "vote_average"},
    )


class StructuredQueryResponse(BaseModel):
    """Direct structured query execution response."""
    success: bool = Field(..., example=True)
    operation: str = Field(..., example="top_movies")
    result_count: int = Field(..., example=5)
    results: Any = Field(..., description="Retrieved records from the Knowledge Graph.")
    execution_time_ms: float = Field(..., example=12.4)
    error: Optional[Dict[str, str]] = None


class OperationInfo(BaseModel):
    """Metadata describing a supported graph operation."""
    description: str
    allowed_parameters: List[str]


class OperationsResponse(BaseModel):
    """Documentation and allowlist of all supported operations and metrics."""
    supported_operations: List[str]
    supported_metrics: List[str]
    supported_roles: List[str]
    supported_sort_orders: List[str]
    operations: Dict[str, OperationInfo]


class ErrorDetail(BaseModel):
    """Generic error detail model."""
    error: str = Field(..., example="INVALID_REQUEST")
    message: str = Field(..., example="The request body was malformed.")
    code: str = Field(..., example="BAD_REQUEST")
    details: Optional[Any] = None
