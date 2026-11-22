from pathlib import Path
import os
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset paths
MOVIES_RAW_PATH = RAW_DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_RAW_PATH = RAW_DATA_DIR / "tmdb_5000_credits.csv"

# Processed paths
PROCESSED_MOVIES_JSONL = PROCESSED_DATA_DIR / "movies_processed.jsonl"
PROCESSED_MOVIES_PARQUET = PROCESSED_DATA_DIR / "movies_processed.parquet"
GRAPH_PICKLE_PATH = PROCESSED_DATA_DIR / "tmdb_knowledge_graph.pickle"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Gemini API and Pricing Configuration
# Official pricing source: https://ai.google.dev/gemini-api/docs/pricing
# Verified date: 2025-05-01
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_VERIFIED_DATE = "2025-05-01"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_PRICING_TIER = os.getenv("GEMINI_PRICING_TIER", "paid_standard").strip().lower()

# Base pricing table (USD per 1M tokens)
# Gemini 2.5 Flash Standard (Paid Tier): $0.30 / 1M input tokens, $2.50 / 1M output tokens
DEFAULT_MODEL_PRICING = {
    "gemini-2.5-flash": {
        "paid_standard": {"input": 0.30, "output": 2.50},
        "free": {"input": 0.0, "output": 0.0, "equivalent_paid_input": 0.30, "equivalent_paid_output": 2.50},
    },
    "gemini-2.0-flash": {
        "paid_standard": {"input": 0.10, "output": 0.40},
        "free": {"input": 0.0, "output": 0.0, "equivalent_paid_input": 0.10, "equivalent_paid_output": 0.40},
    },
    "gemini-1.5-flash": {
        "paid_standard": {"input": 0.075, "output": 0.30},
        "free": {"input": 0.0, "output": 0.0, "equivalent_paid_input": 0.075, "equivalent_paid_output": 0.30},
    },
    "gemini-1.5-pro": {
        "paid_standard": {"input": 1.25, "output": 5.00},
        "free": {"input": 0.0, "output": 0.0, "equivalent_paid_input": 1.25, "equivalent_paid_output": 5.00},
    },
}

# Optional environment variable overrides for custom model pricing
_env_input_price = os.getenv("GEMINI_INPUT_PRICE_PER_M")
_env_output_price = os.getenv("GEMINI_OUTPUT_PRICE_PER_M")
if _env_input_price and _env_output_price and GEMINI_MODEL in DEFAULT_MODEL_PRICING:
    try:
        DEFAULT_MODEL_PRICING[GEMINI_MODEL]["paid_standard"]["input"] = float(_env_input_price)
        DEFAULT_MODEL_PRICING[GEMINI_MODEL]["paid_standard"]["output"] = float(_env_output_price)
    except ValueError:
        pass

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "TMDB Knowledge Graph QA API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "FastAPI backend exposing TMDB 5000 movie knowledge graph question answering powered by Google Gemini."
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if origin.strip()
]
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "500"))
