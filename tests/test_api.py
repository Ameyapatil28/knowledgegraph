"""Unit and integration tests for Phase 5: FastAPI Backend.

All 16 required test scenarios verify endpoints, error conditions, validation,
CORS policies, lifecycle singleton behavior, and secret protection using TestClient
with mocked Gemini responses.
"""

from __future__ import annotations
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.gemini_client import GeminiClient, GeminiClientError, GeminiUsageMetadata
from src.query_engine import QuerySpecification, load_query_engine
from src.query_parser import ParsedQueryResponse, NaturalLanguageQueryParser
from src.answer_generator import GeneratedAnswerResponse, NaturalLanguageAnswerGenerator


@pytest.fixture(scope="module")
def dummy_telemetry() -> GeminiUsageMetadata:
    """Fixture providing realistic Gemini 2.5 Flash usage telemetry."""
    return GeminiUsageMetadata(
        provider="LIVE_GEMINI",
        model="gemini-2.5-flash",
        pricing_tier="paid_standard",
        cost_available=True,
        input_rate_per_m=0.30,
        output_rate_per_m=2.50,
        prompt_tokens=150,
        completion_tokens=60,
        total_tokens=210,
        input_cost_usd=0.000045,
        output_cost_usd=0.000150,
        estimated_cost_usd=0.000195,
        actual_cost_usd=0.000195,
        latency_ms=185.0,
        retry_count=0,
    )


@pytest.fixture(scope="module")
def api_client():
    """Module-scoped TestClient with pre-initialized application state for fast deterministic testing."""
    engine = load_query_engine()
    client = GeminiClient()
    parser = NaturalLanguageQueryParser(client=client)
    generator = NaturalLanguageAnswerGenerator(client=client)

    app.state.query_engine = engine
    app.state.gemini_client = client
    app.state.query_parser = parser
    app.state.answer_generator = generator

    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client


# ===========================================================================
# 1. Health endpoint
# ===========================================================================
def test_01_health_endpoint(api_client: TestClient):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
    assert "graph_loaded" in data
    assert "gemini_configured" in data
    assert "gemini_model" in data
    assert data["gemini_mode"] in ("LIVE_GEMINI", "MOCK")
    assert "pricing_tier" in data


# ===========================================================================
# 2. Ask endpoint with valid movie question
# ===========================================================================
def test_02_ask_valid_movie_question(api_client: TestClient, dummy_telemetry: GeminiUsageMetadata):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Tell me about The Dark Knight",
        success=True,
        query_specification=QuerySpecification(operation="movie_details", parameters={"title": "The Dark Knight"}),
        detected_operation="movie_details",
        provider="LIVE_GEMINI",
        confidence=1.0,
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GeneratedAnswerResponse(
        success=True,
        status="SUCCESS",
        answer="The Dark Knight was released in 2008. Directed by Christopher Nolan...",
        original_question="Tell me about The Dark Knight",
        detected_operation="movie_details",
        provider="LIVE_GEMINI",
        telemetry=dummy_telemetry,
        grounding_checked=True,
        grounding_passed=True,
    )

    with patch.object(app.state, "query_parser", mock_parser), patch.object(app.state, "answer_generator", mock_gen):
        resp = api_client.post("/api/v1/ask", json={"question": "Tell me about The Dark Knight"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "SUCCESS"
        assert "The Dark Knight" in data["question"]
        assert "Christopher Nolan" in data["answer"]
        assert data["operation"] == "movie_details"
        assert data["metadata"]["parser_provider"] == "LIVE_GEMINI"
        assert data["metadata"]["grounding_passed"] is True
        assert "X-Request-ID" in resp.headers


# ===========================================================================
# 3. Ask endpoint with unsupported question
# ===========================================================================
def test_03_ask_unsupported_question(api_client: TestClient):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Will Avatar 3 make two billion dollars?",
        success=True,
        is_unsupported=True,
        unsupported_reason="Future box office predictions are not supported by the TMDB dataset.",
        provider="LOCAL_VALIDATION",
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GeneratedAnswerResponse(
        success=True,
        status="UNSUPPORTED",
        answer="Future box office predictions are not supported by the TMDB dataset.",
        original_question="Will Avatar 3 make two billion dollars?",
        provider="LOCAL_VALIDATION",
    )

    with patch.object(app.state, "query_parser", mock_parser), patch.object(app.state, "answer_generator", mock_gen):
        resp = api_client.post("/api/v1/ask", json={"question": "Will Avatar 3 make two billion dollars?"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "UNSUPPORTED"
        assert "not supported" in data["answer"].lower()
        assert data["metadata"]["parser_provider"] == "LOCAL_VALIDATION"


# ===========================================================================
# 4. Ask endpoint with clarification-required question
# ===========================================================================
def test_04_ask_clarification_required(api_client: TestClient):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Tell me about Batman",
        success=True,
        clarification_required=True,
        clarification_question="Multiple movies match 'Batman'. Did you mean Batman (1989) or Batman Begins (2005)?",
        provider="LOCAL_VALIDATION",
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GeneratedAnswerResponse(
        success=True,
        status="CLARIFICATION_REQUIRED",
        answer="Multiple movies match 'Batman'. Did you mean Batman (1989) or Batman Begins (2005)?",
        original_question="Tell me about Batman",
        provider="LOCAL_VALIDATION",
    )

    with patch.object(app.state, "query_parser", mock_parser), patch.object(app.state, "answer_generator", mock_gen):
        resp = api_client.post("/api/v1/ask", json={"question": "Tell me about Batman"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "CLARIFICATION_REQUIRED"
        assert "Did you mean Batman" in data["answer"]


# ===========================================================================
# 5. Ask endpoint with no results
# ===========================================================================
def test_05_ask_no_results(api_client: TestClient, dummy_telemetry: GeminiUsageMetadata):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Movies directed by NonexistentPersonXYZ",
        success=True,
        query_specification=QuerySpecification(
            operation="movies_by_person",
            parameters={"person_name": "NonexistentPersonXYZ", "role": "director"},
        ),
        detected_operation="movies_by_person",
        provider="LIVE_GEMINI",
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GeneratedAnswerResponse(
        success=True,
        status="NO_RESULTS",
        answer="No matching movies were found in the TMDB 5000 dataset for NonexistentPersonXYZ.",
        original_question="Movies directed by NonexistentPersonXYZ",
        detected_operation="movies_by_person",
        provider="LOCAL_VALIDATION",
        telemetry=dummy_telemetry,
        grounding_checked=True,
        grounding_passed=True,
    )

    with patch.object(app.state, "query_parser", mock_parser), patch.object(app.state, "answer_generator", mock_gen):
        resp = api_client.post("/api/v1/ask", json={"question": "Movies directed by NonexistentPersonXYZ"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "NO_RESULTS"
        assert "No matching movies" in data["answer"]


# ===========================================================================
# 6. Direct structured query
# ===========================================================================
def test_06_direct_structured_query(api_client: TestClient):
    body = {
        "operation": "movie_details",
        "parameters": {"title": "The Dark Knight"},
    }
    resp = api_client.post("/api/v1/query", json=body)
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["operation"] == "movie_details"
    assert data["result_count"] >= 1
    assert data["results"]["title"] == "The Dark Knight"
    assert data["execution_time_ms"] > 0


# ===========================================================================
# 7. Invalid structured query (unknown operation or invalid parameter)
# ===========================================================================
def test_07_invalid_structured_query(api_client: TestClient):
    # Case A: Unsupported operation
    resp_op = api_client.post("/api/v1/query", json={"operation": "drop_tables", "parameters": {}})
    assert resp_op.status_code == 400
    assert resp_op.json()["detail"]["error"] == "UNSUPPORTED_OPERATION"

    # Case B: Unexpected parameter for operation
    resp_param = api_client.post(
        "/api/v1/query",
        json={"operation": "movie_details", "parameters": {"title": "Avatar", "sql_injection": "DROP TABLE"}},
    )
    assert resp_param.status_code == 400
    assert resp_param.json()["detail"]["error"] == "UNEXPECTED_PARAMETER"


# ===========================================================================
# 8. Empty question
# ===========================================================================
def test_08_empty_question(api_client: TestClient):
    resp = api_client.post("/api/v1/ask", json={"question": "   "})
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"]["error"] == "EMPTY_QUESTION"


# ===========================================================================
# 9. Excessively long question
# ===========================================================================
def test_09_excessively_long_question(api_client: TestClient):
    long_question = "Tell me about movies " + ("very long " * 80)
    assert len(long_question) > 500

    resp = api_client.post("/api/v1/ask", json={"question": long_question})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data


# ===========================================================================
# 10. Gemini API failure
# ===========================================================================
def test_10_gemini_api_failure(api_client: TestClient):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Tell me about Inception",
        success=False,
        provider="LIVE_GEMINI",
        error={"code": "RATE_LIMIT_EXCEEDED", "message": "Gemini rate limit exceeded."},
    )

    with patch.object(app.state, "query_parser", mock_parser):
        resp = api_client.post("/api/v1/ask", json={"question": "Tell me about Inception"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"]["error"] == "RATE_LIMIT_EXCEEDED"


# ===========================================================================
# 11. Answer generation failure
# ===========================================================================
def test_11_answer_generation_failure(api_client: TestClient):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Tell me about Titanic",
        success=True,
        query_specification=QuerySpecification(operation="movie_details", parameters={"title": "Titanic"}),
        detected_operation="movie_details",
        provider="LIVE_GEMINI",
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GeneratedAnswerResponse(
        success=False,
        status="GENERATION_ERROR",
        answer="Generation error occurred.",
        original_question="Tell me about Titanic",
        error={"code": "SERVICE_UNAVAILABLE", "message": "Gemini answer generation service unavailable."},
    )

    with patch.object(app.state, "query_parser", mock_parser), patch.object(app.state, "answer_generator", mock_gen):
        resp = api_client.post("/api/v1/ask", json={"question": "Tell me about Titanic"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"]["error"] == "SERVICE_UNAVAILABLE"


# ===========================================================================
# 12. Knowledge Graph startup failure
# ===========================================================================
def test_12_graph_startup_failure(api_client: TestClient):
    # Simulate app state where query_engine failed to load
    original_engine = app.state.query_engine
    try:
        app.state.query_engine = None
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_loaded"] is False
        assert data["status"] == "degraded"
    finally:
        app.state.query_engine = original_engine


# ===========================================================================
# 13. CORS configuration
# ===========================================================================
def test_13_cors_configuration(api_client: TestClient):
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    }
    resp = api_client.options("/api/v1/ask", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ===========================================================================
# 14. Multiple requests without reloading the graph
# ===========================================================================
def test_14_singleton_graph_reuse(api_client: TestClient):
    engine_id_1 = id(app.state.query_engine)

    api_client.get("/health")
    api_client.post("/api/v1/query", json={"operation": "movie_details", "parameters": {"title": "Avatar"}})
    api_client.get("/api/v1/operations")

    engine_id_2 = id(app.state.query_engine)
    assert engine_id_1 == engine_id_2, "GraphQueryEngine instance must be preserved across requests"


# ===========================================================================
# 15. Provider metadata accuracy
# ===========================================================================
def test_15_provider_metadata_accuracy(api_client: TestClient, dummy_telemetry: GeminiUsageMetadata):
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = ParsedQueryResponse(
        original_question="Top sci-fi movies",
        success=True,
        query_specification=QuerySpecification(operation="top_movies", parameters={"genre_name": "Science Fiction", "limit": 2}),
        detected_operation="top_movies",
        provider="MOCK",
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = GeneratedAnswerResponse(
        success=True,
        status="SUCCESS",
        answer="Top sci-fi movies list...",
        original_question="Top sci-fi movies",
        detected_operation="top_movies",
        provider="MOCK",
        telemetry=dummy_telemetry,
        grounding_checked=True,
        grounding_passed=True,
    )

    with patch.object(app.state, "query_parser", mock_parser), patch.object(app.state, "answer_generator", mock_gen):
        resp = api_client.post("/api/v1/ask", json={"question": "Top sci-fi movies"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["metadata"]["parser_provider"] == "MOCK"
        assert data["metadata"]["answer_provider"] == "MOCK"


# ===========================================================================
# 16. No secret leakage in API responses
# ===========================================================================
def test_16_no_secret_leakage(api_client: TestClient):
    fake_secret = "AIzaSySecretApiKeyDoNotLeak12345"

    with patch.dict(os.environ, {"GEMINI_API_KEY": fake_secret}):
        # Check /health
        resp_h = api_client.get("/health")
        assert fake_secret not in resp_h.text
        for val in resp_h.headers.values():
            assert fake_secret not in val

        # Check /api/v1/operations
        resp_ops = api_client.get("/api/v1/operations")
        assert fake_secret not in resp_ops.text

        # Check /api/v1/query
        resp_q = api_client.post(
            "/api/v1/query",
            json={"operation": "movie_details", "parameters": {"title": "The Dark Knight"}},
        )
        assert fake_secret not in resp_q.text
