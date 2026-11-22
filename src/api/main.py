"""FastAPI application entry point for TMDB 5000 Knowledge Graph QA backend."""

from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
    API_HOST,
    API_PORT,
)
from src.gemini_client import GeminiClient
from src.query_engine import load_query_engine
from src.query_parser import NaturalLanguageQueryParser
from src.answer_generator import NaturalLanguageAnswerGenerator
from src.api.routes import router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (req:%(request_id)s): %(message)s"
    if "request_id" in logging._levelToName
    else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tmdb_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: Load Knowledge Graph and initialize pipeline singletons once."""
    logger.info("Initializing Knowledge Graph QA API backend...")
    t0 = time.perf_counter()

    # 1. Initialize Knowledge Graph Query Engine
    try:
        engine = load_query_engine()
        app.state.query_engine = engine
        g = getattr(engine, "G", getattr(engine, "graph", None))
        node_count = g.number_of_nodes() if g is not None else 0
        edge_count = g.number_of_edges() if g is not None else 0
        logger.info(f"Knowledge Graph successfully loaded into memory: {node_count:,} nodes, {edge_count:,} edges.")
    except Exception as exc:
        logger.error(f"Critical error loading Knowledge Graph: {exc}", exc_info=False)
        app.state.query_engine = None

    # 2. Initialize Gemini Client, Parser, and Answer Generator
    client = GeminiClient()
    app.state.gemini_client = client

    parser = NaturalLanguageQueryParser(client=client, allow_mock_fallback=True)
    app.state.query_parser = parser

    generator = NaturalLanguageAnswerGenerator(client=client)
    app.state.answer_generator = generator

    provider_mode = "LIVE_GEMINI" if client.is_configured() else "MOCK"
    startup_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"API initialization complete in {startup_ms:.1f}ms. "
        f"Model: '{client.model}' | Provider Mode: {provider_mode} | Tier: '{client.pricing_tier}'"
    )

    yield

    logger.info("Shutting down Knowledge Graph QA API backend...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS Middleware
# Explicitly support localhost and 127.0.0.1 on all development ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestIdAndLoggingMiddleware:
    """Pure ASGI middleware for unique request IDs, latency tracking, and structured logging without BaseHTTPMiddleware deadlocks."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        for name, value in scope.get("headers", []):
            if name.lower() == b"x-request-id":
                try:
                    request_id = value.decode("latin1")
                except Exception:
                    pass
                break

        # Store in scope state
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        start_time = time.perf_counter()
        status_code = [200]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            path = scope.get("path", "")
            method = scope.get("method", "")
            logger.info(
                f"[{request_id[:8]}] {method} {path} -> "
                f"status={status_code[0]} in {duration_ms:.1f}ms"
            )


# Add Request ID & Logging ASGI Middleware
# app.add_middleware(RequestIdAndLoggingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for request validation errors (Pydantic / FastAPI)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    msg = first_error.get("msg", "Invalid request body")
    loc = " -> ".join(str(l) for l in first_error.get("loc", []))

    logger.warning(f"[{request_id[:8]}] Validation error on {request.url.path}: {loc}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": f"Validation error at {loc}: {msg}",
            "details": errors,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


# Register API Routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
