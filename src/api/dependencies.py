"""Dependency injection providers for FastAPI endpoints."""

from __future__ import annotations
import logging
from typing import Optional

from fastapi import Request, HTTPException, status

from src.gemini_client import GeminiClient
from src.query_engine import GraphQueryEngine, load_query_engine
from src.query_parser import NaturalLanguageQueryParser
from src.answer_generator import NaturalLanguageAnswerGenerator

logger = logging.getLogger(__name__)


def get_query_engine(request: Request) -> GraphQueryEngine:
    """Retrieve the singleton GraphQueryEngine from application state."""
    engine: Optional[GraphQueryEngine] = getattr(request.app.state, "query_engine", None)
    if engine is None:
        logger.warning("GraphQueryEngine not found in app.state, initializing fallback...")
        try:
            engine = load_query_engine()
            request.app.state.query_engine = engine
        except Exception as exc:
            logger.error(f"Failed to initialize GraphQueryEngine: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "GRAPH_UNAVAILABLE", "message": "Knowledge Graph could not be loaded."},
            )
    return engine


def get_gemini_client(request: Request) -> GeminiClient:
    """Retrieve the singleton GeminiClient from application state."""
    client: Optional[GeminiClient] = getattr(request.app.state, "gemini_client", None)
    if client is None:
        client = GeminiClient()
        request.app.state.gemini_client = client
    return client


def get_query_parser(request: Request) -> NaturalLanguageQueryParser:
    """Retrieve the singleton NaturalLanguageQueryParser from application state."""
    parser: Optional[NaturalLanguageQueryParser] = getattr(request.app.state, "query_parser", None)
    if parser is None:
        client = get_gemini_client(request)
        parser = NaturalLanguageQueryParser(client=client)
        request.app.state.query_parser = parser
    return parser


def get_answer_generator(request: Request) -> NaturalLanguageAnswerGenerator:
    """Retrieve the singleton NaturalLanguageAnswerGenerator from application state."""
    generator: Optional[NaturalLanguageAnswerGenerator] = getattr(request.app.state, "answer_generator", None)
    if generator is None:
        client = get_gemini_client(request)
        generator = NaturalLanguageAnswerGenerator(client=client)
        request.app.state.answer_generator = generator
    return generator
