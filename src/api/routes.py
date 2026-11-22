"""FastAPI router endpoints for TMDB 5000 Knowledge Graph Question Answering."""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.config import (
    API_VERSION,
    MAX_QUESTION_LENGTH,
)
from src.api.schemas import (
    HealthResponse,
    AskQuestionRequest,
    AskQuestionResponse,
    AnswerMetadata,
    StructuredQueryRequest,
    StructuredQueryResponse,
    OperationsResponse,
    OperationInfo,
    ErrorDetail,
)
from src.api.dependencies import (
    get_query_engine,
    get_query_parser,
    get_answer_generator,
    get_gemini_client,
)
from src.gemini_client import GeminiClient
from src.query_engine import (
    GraphQueryEngine,
    QuerySpecification,
    SUPPORTED_OPERATIONS,
    SUPPORTED_METRICS,
    SUPPORTED_ROLES,
    SUPPORTED_SORT_ORDERS,
    ALLOWED_PARAMETERS_PER_OPERATION,
)
from src.query_parser import NaturalLanguageQueryParser
from src.answer_generator import NaturalLanguageAnswerGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

# Descriptions for all 15 supported operations
OPERATION_DESCRIPTIONS = {
    "movie_details": "Retrieve full movie details including title, year, rating, budget, revenue, directors, genres, and overview.",
    "search_movies": "Fuzzy search movies by title with optional result limit.",
    "movie_cast": "List top actors and their characters for a specific movie.",
    "movie_crew": "List crew members for a movie, optionally filtered by department or job.",
    "movie_directors": "List directors for a specific movie.",
    "movies_by_person": "Retrieve movies associated with an actor, director, or crew member.",
    "movies_by_genre": "Rank movies within a genre by rating, popularity, revenue, or budget.",
    "movies_by_keyword": "Retrieve and rank movies associated with a specific thematic keyword.",
    "movies_by_company": "Retrieve and rank movies produced by a specific production company.",
    "top_movies": "Global ranking of movies by rating, revenue, popularity, or budget with optional genre filter.",
    "compare_movies": "Side-by-side comparison of budget, revenue, rating, and runtime for multiple movies.",
    "common_cast": "Identify actors who appeared in common across two or more specified movies.",
    "aggregate_by_genre": "Statistical aggregation (count, average, min, max) of metrics grouped by genre.",
    "movies_by_year_range": "Chronological retrieval of movies released within a given year range.",
    "count_by_year": "Count and distribution of movies released per year across a year range.",
}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    description="Check the system status, loaded Knowledge Graph metrics, and Gemini configuration without exposing secrets.",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """Return application health, Knowledge Graph readiness, and Gemini client configuration."""
    engine: Optional[GraphQueryEngine] = getattr(request.app.state, "query_engine", None)
    graph_loaded = False
    graph_nodes = None
    graph_edges = None

    if engine is not None:
        g = getattr(engine, "G", getattr(engine, "graph", None))
        if g is not None:
            graph_loaded = True
            try:
                graph_nodes = g.number_of_nodes()
                graph_edges = g.number_of_edges()
            except Exception:
                pass

    client: Optional[GeminiClient] = getattr(request.app.state, "gemini_client", None)
    if client is None:
        client = GeminiClient()

    is_configured = client.is_configured()
    gemini_mode = "LIVE_GEMINI" if is_configured else "MOCK"

    return HealthResponse(
        status="healthy" if graph_loaded else "degraded",
        version=API_VERSION,
        graph_loaded=graph_loaded,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        gemini_configured=is_configured,
        gemini_model=client.model,
        gemini_mode=gemini_mode,
        pricing_tier=client.pricing_tier,
    )


@router.post(
    "/api/v1/ask",
    response_model=AskQuestionResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Empty question or validation error"},
        422: {"model": ErrorDetail, "description": "Question exceeds length limit"},
        500: {"model": ErrorDetail, "description": "Internal parsing or generation failure"},
    },
    summary="Ask Natural Language Question",
    description=(
        "Translates a natural language question into a structured graph query using Gemini, "
        "executes it against the Knowledge Graph, and generates a conversational grounded answer."
    ),
    tags=["Question Answering"],
)
async def ask_question(
    req: AskQuestionRequest,
    engine: GraphQueryEngine = Depends(get_query_engine),
    parser: NaturalLanguageQueryParser = Depends(get_query_parser),
    generator: NaturalLanguageAnswerGenerator = Depends(get_answer_generator),
    client: GeminiClient = Depends(get_gemini_client),
) -> AskQuestionResponse:
    """End-to-end question answering pipeline."""
    clean_question = req.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "EMPTY_QUESTION", "message": "Question cannot be empty or whitespace."},
        )

    if len(clean_question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "QUESTION_TOO_LONG",
                "message": f"Question exceeds maximum allowed length of {MAX_QUESTION_LENGTH} characters.",
            },
        )

    t_start = time.perf_counter()

    # Step 1: Parse Natural Language Query
    t_p0 = time.perf_counter()
    parsed_resp = parser.parse_query(clean_question)
    parser_latency_ms = (time.perf_counter() - t_p0) * 1000

    # Handle Clarification Required
    if parsed_resp.clarification_required:
        ans_resp = generator.generate_answer(question=clean_question, parser_response=parsed_resp)
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        return AskQuestionResponse(
            status="CLARIFICATION_REQUIRED",
            question=clean_question,
            answer=ans_resp.answer,
            operation=parsed_resp.detected_operation,
            results=None,
            metadata=AnswerMetadata(
                parser_provider=parsed_resp.provider,
                answer_provider="LOCAL_VALIDATION",
                model=client.model,
                pricing_tier=client.pricing_tier,
                total_latency_ms=round(total_latency_ms, 2),
                parser_latency_ms=round(parser_latency_ms, 2),
                engine_latency_ms=0.0,
                answer_latency_ms=0.0,
                grounding_passed=True,
                grounding_repaired=False,
            ),
        )

    # Handle Unsupported Question
    if parsed_resp.is_unsupported:
        ans_resp = generator.generate_answer(question=clean_question, parser_response=parsed_resp)
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        return AskQuestionResponse(
            status="UNSUPPORTED",
            question=clean_question,
            answer=ans_resp.answer,
            operation=parsed_resp.detected_operation,
            results=None,
            metadata=AnswerMetadata(
                parser_provider=parsed_resp.provider,
                answer_provider="LOCAL_VALIDATION",
                model=client.model,
                pricing_tier=client.pricing_tier,
                total_latency_ms=round(total_latency_ms, 2),
                parser_latency_ms=round(parser_latency_ms, 2),
                engine_latency_ms=0.0,
                answer_latency_ms=0.0,
                grounding_passed=True,
                grounding_repaired=False,
            ),
        )

    # Handle Parser API Failure
    if not parsed_resp.success or not parsed_resp.query_specification:
        err_msg = parsed_resp.error.get("message") if parsed_resp.error else "Failed to parse question into a valid query specification."
        err_code = parsed_resp.error.get("code") if parsed_resp.error else "PARSER_ERROR"
        logger.error(f"Parser failure for question '{clean_question}': {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": err_code, "message": err_msg},
        )

    spec = parsed_resp.query_specification

    # Step 2: Execute in Knowledge Graph Query Engine
    t_e0 = time.perf_counter()
    retrieval_result = engine.execute_query(spec)
    engine_latency_ms = (time.perf_counter() - t_e0) * 1000

    # Step 3: Generate Natural Language Answer
    t_a0 = time.perf_counter()
    ans_resp = generator.generate_answer(
        question=clean_question,
        query_specification=spec,
        retrieval_result=retrieval_result,
        parser_response=parsed_resp,
    )
    answer_latency_ms = (time.perf_counter() - t_a0) * 1000
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    # Check for Answer Generation Failure
    if not ans_resp.success and ans_resp.status == "GENERATION_ERROR":
        err_code = ans_resp.error.get("code") if ans_resp.error else "GENERATION_ERROR"
        err_msg = ans_resp.error.get("message") if ans_resp.error else "Answer generator failed."
        logger.error(f"Answer generation error: {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": err_code, "message": err_msg},
        )

    # Extract Token and Cost Telemetry
    tokens_dict = None
    cost_usd = None
    if ans_resp.telemetry:
        if ans_resp.telemetry.total_tokens is not None:
            tokens_dict = {
                "prompt_tokens": ans_resp.telemetry.prompt_tokens,
                "completion_tokens": ans_resp.telemetry.completion_tokens,
                "total_tokens": ans_resp.telemetry.total_tokens,
            }
        if ans_resp.telemetry.cost_available:
            cost_usd = ans_resp.telemetry.actual_cost_usd

    return AskQuestionResponse(
        status=ans_resp.status,
        question=clean_question,
        answer=ans_resp.answer,
        operation=spec.operation,
        results=retrieval_result.get("results"),
        metadata=AnswerMetadata(
            parser_provider=parsed_resp.provider,
            answer_provider=ans_resp.provider,
            model=client.model,
            pricing_tier=client.pricing_tier,
            total_latency_ms=round(total_latency_ms, 2),
            parser_latency_ms=round(parser_latency_ms, 2),
            engine_latency_ms=round(engine_latency_ms, 2),
            answer_latency_ms=round(answer_latency_ms, 2),
            tokens=tokens_dict,
            cost_usd=cost_usd,
            grounding_passed=ans_resp.grounding_passed,
            grounding_repaired=ans_resp.grounding_repaired,
            grounding_violations=ans_resp.grounding_violations,
        ),
        error=ans_resp.error,
    )


@router.post(
    "/api/v1/query",
    response_model=StructuredQueryResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid operation or unexpected parameters"},
        500: {"model": ErrorDetail, "description": "Query execution failure"},
    },
    summary="Execute Structured Query Directly",
    description="Submits a structured QuerySpecification directly to the Knowledge Graph Query Engine without calling Gemini.",
    tags=["Query Engine"],
)
async def execute_structured_query(
    req: StructuredQueryRequest,
    engine: GraphQueryEngine = Depends(get_query_engine),
) -> StructuredQueryResponse:
    """Direct deterministic query execution without LLM dependency."""
    op = req.operation.strip()

    # 1. Validate operation against strict allowlist
    if op not in SUPPORTED_OPERATIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "UNSUPPORTED_OPERATION",
                "message": f"Operation '{op}' is not supported. Supported operations: {sorted(SUPPORTED_OPERATIONS)}",
            },
        )

    # 2. Validate parameters against strict allowlist
    allowed_params = ALLOWED_PARAMETERS_PER_OPERATION.get(op, set())
    provided_params = set(req.parameters.keys())
    unexpected = provided_params - allowed_params
    if unexpected:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "UNEXPECTED_PARAMETER",
                "message": f"Parameters {sorted(unexpected)} are not allowed for operation '{op}'. Allowed: {sorted(allowed_params)}",
            },
        )

    # 3. Create and execute QuerySpecification
    try:
        spec = QuerySpecification(operation=op, parameters=req.parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_SPECIFICATION", "message": str(exc)},
        )

    retrieval_res = engine.execute_query(spec)

    return StructuredQueryResponse(
        success=retrieval_res.get("success", False),
        operation=op,
        result_count=retrieval_res.get("result_count", 0),
        results=retrieval_res.get("results"),
        execution_time_ms=retrieval_res.get("execution_time_ms", 0.0),
        error=retrieval_res.get("error"),
    )


@router.get(
    "/api/v1/operations",
    response_model=OperationsResponse,
    summary="List Supported Operations and Metrics",
    description="Retrieve all supported query operations, parameters, metrics, roles, and sort orders directly from graph definitions.",
    tags=["Query Engine"],
)
async def list_operations() -> OperationsResponse:
    """Return all supported operations, metrics, roles, and allowed parameters."""
    ops_dict: Dict[str, OperationInfo] = {}
    for op in sorted(SUPPORTED_OPERATIONS):
        allowed = sorted(list(ALLOWED_PARAMETERS_PER_OPERATION.get(op, [])))
        desc = OPERATION_DESCRIPTIONS.get(op, f"Execute '{op}' graph query.")
        ops_dict[op] = OperationInfo(description=desc, allowed_parameters=allowed)

    return OperationsResponse(
        supported_operations=sorted(list(SUPPORTED_OPERATIONS)),
        supported_metrics=sorted(list(SUPPORTED_METRICS)),
        supported_roles=sorted(list(SUPPORTED_ROLES)),
        supported_sort_orders=sorted(list(SUPPORTED_SORT_ORDERS)),
        operations=ops_dict,
    )
