"""Google Gemini API client wrapper using the official google-genai SDK.

Handles authentication, timeouts, retries, usage telemetry, and safe error handling
without exposing credentials or executing arbitrary code.
"""

from __future__ import annotations
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    GEMINI_MAX_RETRIES,
    GEMINI_PRICING_TIER,
    DEFAULT_MODEL_PRICING,
    PRICING_SOURCE,
    PRICING_VERIFIED_DATE,
)

logger = logging.getLogger(__name__)


class GeminiUsageMetadata(BaseModel):
    """Telemetry tracking model usage, latency, and costs."""
    provider: str = "LIVE_GEMINI"  # "LIVE_GEMINI" or "MOCK"
    model: str = ""
    pricing_tier: str = "paid_standard"  # "paid_standard" or "free"
    pricing_source: str = PRICING_SOURCE
    pricing_verified_date: str = PRICING_VERIFIED_DATE
    cost_available: bool = False
    cost_status_reason: Optional[str] = None  # Reason why cost is unavailable
    input_rate_per_m: Optional[float] = None
    output_rate_per_m: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    estimated_cost_usd: Optional[float] = None  # Standard paid-tier equivalent cost
    actual_cost_usd: Optional[float] = None     # $0.00 for free tier, equal to estimated for paid
    latency_ms: float = 0.0
    retry_count: int = 0


class GeminiClientError(Exception):
    """Safe exception for Gemini API failures without leaking sensitive credentials."""
    def __init__(self, code: str, message: str):
        # Sanitize any accidental API key occurrences from error message
        sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", str(message))
        super().__init__(sanitized)
        self.code = code
        self.message = sanitized


def _sanitize_message(msg: str) -> str:
    """Strip any API key pattern from error strings."""
    return re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", str(msg))


class GeminiClient:
    """Wrapper around official google-genai SDK Client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        pricing_tier: Optional[str] = None,
        pricing_config: Optional[Dict[str, Any]] = None,
    ):
        raw_key = api_key if api_key is not None else GEMINI_API_KEY
        self.api_key = raw_key.strip() if raw_key else ""
        self.model = (model or GEMINI_MODEL).strip()
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else GEMINI_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else GEMINI_MAX_RETRIES
        self.pricing_tier = (pricing_tier or GEMINI_PRICING_TIER).strip().lower()
        self.pricing_config = pricing_config or DEFAULT_MODEL_PRICING

        self._client: Optional[genai.Client] = None
        if self.is_configured():
            try:
                http_opts = types.HttpOptions(timeout=float(self.timeout_seconds))
                self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
            except Exception as exc:
                logger.error(f"Failed to initialize google-genai Client: {_sanitize_message(str(exc))}")
                self._client = None

    def is_configured(self) -> bool:
        """Check whether a non-empty, non-placeholder API key is set."""
        return bool(
            self.api_key
            and self.api_key != "your_gemini_api_key_here"
            and self.api_key != "your_api_key_here"
        )

    def calculate_cost(
        self,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        is_mock: bool = False,
    ) -> Dict[str, Any]:
        """Calculate token costs separately for input and output according to pricing tier.
        
        Official source: https://ai.google.dev/gemini-api/docs/pricing
        Verified date: 2025-05-01
        """
        if is_mock:
            return {
                "cost_available": False,
                "cost_status_reason": "Cost calculation is disabled for mock requests.",
                "input_rate_per_m": None,
                "output_rate_per_m": None,
                "input_cost_usd": None,
                "output_cost_usd": None,
                "estimated_cost_usd": None,
                "actual_cost_usd": None,
            }

        if prompt_tokens is None or completion_tokens is None:
            return {
                "cost_available": False,
                "cost_status_reason": "Actual token usage metadata was not returned by Gemini API.",
                "input_rate_per_m": None,
                "output_rate_per_m": None,
                "input_cost_usd": None,
                "output_cost_usd": None,
                "estimated_cost_usd": None,
                "actual_cost_usd": None,
            }

        model_pricing = self.pricing_config.get(self.model, {})
        tier_data = model_pricing.get(self.pricing_tier)
        if not tier_data:
            tier_data = model_pricing.get("paid_standard", {})

        if not tier_data:
            return {
                "cost_available": False,
                "input_rate_per_m": None,
                "output_rate_per_m": None,
                "input_cost_usd": None,
                "output_cost_usd": None,
                "estimated_cost_usd": None,
                "actual_cost_usd": None,
            }

        if self.pricing_tier == "free":
            in_rate = tier_data.get("equivalent_paid_input", 0.30)
            out_rate = tier_data.get("equivalent_paid_output", 2.50)
            in_cost = round((prompt_tokens / 1_000_000.0) * in_rate, 6)
            out_cost = round((completion_tokens / 1_000_000.0) * out_rate, 6)
            est_cost = round(in_cost + out_cost, 6)
            return {
                "cost_available": True,
                "input_rate_per_m": in_rate,
                "output_rate_per_m": out_rate,
                "input_cost_usd": in_cost,
                "output_cost_usd": out_cost,
                "estimated_cost_usd": est_cost,
                "actual_cost_usd": 0.0,
            }
        else:
            in_rate = tier_data.get("input", 0.30)
            out_rate = tier_data.get("output", 2.50)
            in_cost = round((prompt_tokens / 1_000_000.0) * in_rate, 6)
            out_cost = round((completion_tokens / 1_000_000.0) * out_rate, 6)
            tot_cost = round(in_cost + out_cost, 6)
            return {
                "cost_available": True,
                "input_rate_per_m": in_rate,
                "output_rate_per_m": out_rate,
                "input_cost_usd": in_cost,
                "output_cost_usd": out_cost,
                "estimated_cost_usd": tot_cost,
                "actual_cost_usd": tot_cost,
            }

    def calculate_estimated_cost(
        self, prompt_tokens: Optional[int], completion_tokens: Optional[int]
    ) -> Optional[float]:
        """Backward-compatible helper returning total estimated USD cost."""
        return self.calculate_cost(prompt_tokens, completion_tokens).get("estimated_cost_usd")

    def _execute_generate(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> Tuple[str, GeminiUsageMetadata]:
        """Execute model content generation with retries, timeout, and telemetry tracking."""
        if not self.is_configured():
            raise GeminiClientError(
                code="MISSING_API_KEY",
                message="GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment or .env file.",
            )

        if not self._client:
            try:
                http_opts = types.HttpOptions(timeout=float(self.timeout_seconds))
                self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
            except Exception as exc:
                raise GeminiClientError(
                    code="INITIALIZATION_ERROR",
                    message=f"Failed to initialize Gemini Client: {_sanitize_message(str(exc))}",
                )

        attempts = 0
        last_error = None
        start_time = time.perf_counter()

        while attempts <= self.max_retries:
            attempts += 1
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                if not response or not response.text:
                    raise GeminiClientError(
                        code="EMPTY_RESPONSE",
                        message="Gemini returned an empty response.",
                    )

                # Extract telemetry metadata
                prompt_toks = None
                comp_toks = None
                tot_toks = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    prompt_toks = getattr(um, "prompt_token_count", None)
                    comp_toks = getattr(um, "candidates_token_count", None)
                    tot_toks = getattr(um, "total_token_count", None)

                cost_info = self.calculate_cost(prompt_toks, comp_toks, is_mock=False)

                usage = GeminiUsageMetadata(
                    provider="LIVE_GEMINI",
                    model=self.model,
                    pricing_tier=self.pricing_tier,
                    cost_available=cost_info["cost_available"],
                    cost_status_reason=cost_info.get("cost_status_reason"),
                    input_rate_per_m=cost_info["input_rate_per_m"],
                    output_rate_per_m=cost_info["output_rate_per_m"],
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    total_tokens=tot_toks,
                    input_cost_usd=cost_info["input_cost_usd"],
                    output_cost_usd=cost_info["output_cost_usd"],
                    estimated_cost_usd=cost_info["estimated_cost_usd"],
                    actual_cost_usd=cost_info["actual_cost_usd"],
                    latency_ms=round(latency_ms, 2),
                    retry_count=attempts - 1,
                )

                return response.text.strip(), usage

            except APIError as api_err:
                err_str = _sanitize_message(str(api_err))
                err_code = getattr(api_err, "code", None)

                # Map specific API errors
                if err_code in (401, 403) or "API_KEY_INVALID" in err_str or "unregistered" in err_str.lower():
                    raise GeminiClientError(code="INVALID_API_KEY", message="Provided Gemini API key is invalid or unauthorized.")
                elif err_code == 404 or "not found" in err_str.lower():
                    raise GeminiClientError(code="INVALID_MODEL", message=f"Configured model '{self.model}' was not found.")
                elif err_code == 429 or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    if attempts <= self.max_retries:
                        time.sleep(1.0 * attempts)
                        last_error = api_err
                        continue
                    raise GeminiClientError(code="RATE_LIMIT_EXCEEDED", message="Gemini rate limit or quota exceeded. Please try again shortly.")
                elif err_code and err_code >= 500:
                    if attempts <= self.max_retries:
                        time.sleep(0.5 * attempts)
                        last_error = api_err
                        continue
                    raise GeminiClientError(code="SERVICE_UNAVAILABLE", message=f"Gemini service error ({err_code}): {err_str}")
                else:
                    raise GeminiClientError(code="API_ERROR", message=f"Gemini API error: {err_str}")

            except (TimeoutError, Exception) as exc:
                err_str = _sanitize_message(str(exc))
                exc_name = type(exc).__name__.lower()

                if "timeout" in exc_name or "timed out" in err_str.lower():
                    if attempts <= self.max_retries:
                        time.sleep(0.5 * attempts)
                        last_error = exc
                        continue
                    raise GeminiClientError(code="TIMEOUT_ERROR", message=f"Gemini API request timed out after {self.timeout_seconds}s.")
                elif "connect" in exc_name or "network" in err_str.lower() or "dns" in err_str.lower():
                    if attempts <= self.max_retries:
                        time.sleep(0.5 * attempts)
                        last_error = exc
                        continue
                    raise GeminiClientError(code="NETWORK_ERROR", message=f"Network error connecting to Gemini API: {err_str}")
                else:
                    raise GeminiClientError(code="INTERNAL_ERROR", message=f"Unexpected error calling Gemini API: {err_str}")

        raise GeminiClientError(
            code="MAX_RETRIES_EXCEEDED",
            message=f"Failed to generate content after {self.max_retries} retries. Last error: {_sanitize_message(str(last_error))}",
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Tuple[str, GeminiUsageMetadata]:
        """Generate a strictly structured JSON response from Gemini."""
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )
        return self._execute_generate(prompt=prompt, config=config)

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Tuple[str, GeminiUsageMetadata]:
        """Generate conversational natural-language text response from Gemini."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        return self._execute_generate(prompt=prompt, config=config)
