# TMDB 5000 Knowledge Graph Question Answering System

A production-grade Knowledge Graph Question Answering (QA) application built on the TMDB 5000 Movies & Credits datasets. The system combines an in-memory NetworkX `MultiDiGraph`, the official Google Gemini API (`google-genai`), Pydantic v2 schemas, a FastAPI REST service, and a modern React + TypeScript + Tailwind CSS chat interface.

---

## Complete Architecture Overview

```
+-----------------------------------------------------------------------------------------+
|                                    Frontend Client                                      |
|  - React 18 + TypeScript + Vite + Tailwind CSS + Lucide React                           |
|  - Conversational Chat UI with Markdown & Badges (Clarification, Unsupported, Error)   |
|  - Collapsible Technical Telemetry Panel (Model, Tokens, Cost, Latency breakdown)       |
|  - Raw Knowledge Graph JSON Result Viewer                                               |
|  - Knowledge Graph Insights Dashboard (6 Metric Cards, Relations, 15 Operations)        |
+-----------------------------------------------------------------------------------------+
                                             |
                                 REST API (HTTP / JSON)
                                             v
+-----------------------------------------------------------------------------------------+
|                                  FastAPI Backend Service                                |
|  - GET  /health              (Graph stats, Gemini config, pricing tier)                 |
|  - POST /api/v1/ask          (Full NL QA Pipeline: Parse -> Query -> Grounded Answer)   |
|  - POST /api/v1/query        (Direct Structured Graph Query Execution)                  |
|  - GET  /api/v1/operations   (Operation Catalog, Parameters, Allowed Metrics)           |
|  - CORS Middleware, Pure ASGI Request ID & Logging Middleware, OpenAPI Docs (/docs)     |
|  - Lifespan Singleton Management: Pre-loaded NetworkX Graph, Parser, and Generator      |
+-----------------------------------------------------------------------------------------+
                                             |
        +------------------------------------+-----------------------------------+
        |                                                                        |
        v                                                                        v
+------------------------------------+                 +---------------------------------+
|     NaturalLanguageQueryParser     |                 |   NaturalLanguageAnswerGenerator|
|  - Prompt Injection Defense        |                 |  - Grounded Conversational Synth|
|  - google-genai Gemini Client      |                 |  - Strict Result Grounding Check|
|  - Strict Structured JSON Schema   |                 |  - Deterministic Zero-Shot      |
|  - Domain & Ambiguity Guardrails   |                 |    Fallback on Hallucination    |
+------------------------------------+                 +---------------------------------+
        |                                                                        ^
        v                                                                        |
+--------------------------------------------------------------------------------+--------+
|                          GraphQueryEngine (NetworkX MultiDiGraph)                       |
|  - 124,700 Nodes & 311,242 Directed Edges                                               |
|  - In-Memory Inverted Index Lookups ($O(1)$ titles, names, genres, studios, keywords)   |
|  - Deterministic Multi-Metric Traversal & Secondary Tie-Breaking Sorting                |
|  - 15 Core Query & Analytical Operations                                                |
+-----------------------------------------------------------------------------------------+
```

---

## Technology Stack

### Backend
- **Language**: Python 3.10.8
- **Data Engineering**: Pandas, NumPy, PyArrow (Parquet)
- **Knowledge Graph Database**: NetworkX `MultiDiGraph` (persisted via Pickle)
- **LLM SDK**: Official Google Gemini SDK (`google-genai` with `gemini-2.5-flash`)
- **Schema Validation & Modeling**: Pydantic v2
- **REST Framework**: FastAPI & Uvicorn
- **Testing**: pytest

### Frontend
- **Framework**: React 18 & TypeScript
- **Build Tooling**: Vite
- **Styling**: Tailwind CSS (Dark theme: `#0B0F19`, `#111827`, `#1F2937`)
- **Icons**: Lucide React
- **Markdown & Rendering**: react-markdown
- **Testing**: Vitest

---

## Directory Structure

```
knowledge_graog/
├── data/
│   ├── raw/
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/
│       ├── movies_processed.parquet         # Cleaned tabular dataset
│       ├── movies_processed.jsonl           # Cleaned JSONL dataset
│       └── tmdb_knowledge_graph.pickle      # Serialized NetworkX MultiDiGraph
├── src/
│   ├── config.py                            # Global paths, parameters, API settings
│   ├── preprocessing.py                     # Phase 1: Cleaning, key validation, JSON parsing
│   ├── graph_builder.py                     # Phase 2: NetworkX MultiDiGraph construction
│   ├── query_engine.py                      # Phase 3: Deterministic query engine (15 operations)
│   ├── gemini_client.py                     # Phase 4A: Google Gemini API client wrapper & telemetry
│   ├── query_parser.py                      # Phase 4A: Natural Language to QuerySpecification parser
│   ├── answer_generator.py                  # Phase 4B: Grounded natural-language answer synthesis
│   ├── api/                                 # Phase 5: FastAPI Backend REST API
│   │   ├── __init__.py                      # Package exports app
│   │   ├── main.py                          # FastAPI app, lifespan, CORS, middleware, errors
│   │   ├── routes.py                        # /health, /api/v1/ask, /api/v1/query, /api/v1/operations
│   │   ├── schemas.py                       # Pydantic request & response models
│   │   └── dependencies.py                  # Dependency injection for singletons
│   ├── components/                          # Phase 6: React Frontend Components
│   │   ├── Sidebar.tsx                      # Collapsible navigation, history, status, tab switch
│   │   ├── WelcomeScreen.tsx                # Hero screen with suggested question cards
│   │   ├── ChatInput.tsx                    # Auto-resizing textarea with keyboard shortcuts
│   │   ├── MessageItem.tsx                  # Markdown answers, status badges, copy button
│   │   ├── MetadataPanel.tsx                # Collapsible telemetry panel (tokens, cost, latency)
│   │   ├── JsonViewer.tsx                   # Expandable NetworkX query results viewer
│   │   └── InsightsDashboard.tsx            # Graph metrics, schema relations, supported operations
│   ├── hooks/
│   │   └── useChat.ts                       # Custom React hook for conversation state & API calls
│   ├── services/
│   │   └── api.ts                           # Frontend API service connecting to backend
│   ├── types/
│   │   └── api.ts                           # TypeScript interfaces matching backend models
│   ├── __tests__/
│   │   └── frontend.test.ts                 # Vitest frontend unit tests (12 tests)
│   ├── App.tsx                              # Main application layout coordinator
│   ├── main.tsx                             # React entry point
│   └── index.css                            # Tailwind CSS styling & animations
├── tests/
│   ├── test_preprocessing.py                # Phase 1 unit tests (13 tests)
│   ├── test_graph.py                        # Phase 2 unit tests (16 tests)
│   ├── test_query_engine.py                 # Phase 3 unit tests (33 tests)
│   ├── test_query_parser.py                 # Phase 4A unit tests (27 tests)
│   ├── test_answer_generator.py             # Phase 4B unit tests (20 tests)
│   └── test_api.py                          # Phase 5 FastAPI backend unit tests (16 tests)
├── .env.example                             # Environment configuration template
├── package.json                             # Frontend dependencies & npm scripts
├── tsconfig.json                            # TypeScript configuration
├── vite.config.ts                           # Vite configuration with proxy
├── tailwind.config.js                       # Tailwind CSS styling configuration
├── requirements.txt                         # Pinned Python dependencies
├── run_preprocessing.py                     # Preprocessing execution script
├── run_graph_builder.py                     # Graph construction script
├── run_query_demo.py                        # 15 query engine operations demonstration
├── run_gemini_parser_demo.py                # Phase 4A NL query parser demonstration
├── run_full_qa_demo.py                      # Phase 4B end-to-end QA pipeline demonstration
├── run_demo.bat / run_demo.ps1              # Phase 4A demo runner scripts
└── run_full_qa.bat / run_full_qa.ps1        # Phase 4B demo runner scripts
```

---

## Implementation Phases

### Phase 1: Data Validation and Preprocessing
- Validated primary and foreign key integrity across movies and credits datasets (`id` == `movie_id`).
- Safely parsed all seven JSON-encoded columns (`genres`, `keywords`, `production_companies`, `production_countries`, `spoken_languages`, `cast`, `crew`).
- Preserved unrecorded/unknown budgets and revenues as missing (`NaN`) rather than misleading zeros.
- Disambiguated duplicate movie titles with release years (e.g., *Batman (1989)* vs. *Batman (1966)*).
- Extracted directors and key crew positions with clean normalization.
- Serialized cleaned dataset to Parquet (10.49 MB) and JSONL (43.34 MB).

### Phase 2: Knowledge Graph Construction
- Constructed a NetworkX `MultiDiGraph` with 124,700 nodes and 311,242 directed edges.
- **7 Node Types**: `Movie`, `Person`, `Genre`, `Keyword`, `ProductionCompany`, `ProductionCountry`, `Language`.
- **8 Directed Relationship Types**: `HAS_CAST`, `HAS_CREW_MEMBER`, `DIRECTED_BY`, `HAS_GENRE`, `HAS_KEYWORD`, `PRODUCED_BY`, `PRODUCED_IN`, `HAS_LANGUAGE`.
- Single person node identity maintained for multi-role contributors (e.g. director + actor).
- Edge attributes preserve cast `order`, `character`, crew `job`, `department`, and `credit_id`.

### Phase 3: Graph Query Engine
- Deterministic, high-performance query execution without LLM or vector database dependencies.
- In-memory inverted indexes for fast $O(1)$ lookups on movies, titles, people, genres, keywords, and studios.
- Supported 15 core operations:
  1. `movie_details`
  2. `search_movies`
  3. `movie_cast`
  4. `movie_crew`
  5. `movie_directors`
  6. `movies_by_person`
  7. `movies_by_genre` (with deterministic metric tie-breaking)
  8. `movies_by_keyword`
  9. `movies_by_company`
  10. `top_movies`
  11. `compare_movies`
  12. `common_cast`
  13. `aggregate_by_genre` (with `min_vote_count` filtering unrated zero-vote outliers)
  14. `movies_by_year_range` (chronological sorting order configurable: ascending or descending)
  15. `count_by_year`

### Phase 4A: Gemini Natural Language Query Understanding
- Translates natural language questions into strictly validated Pydantic `QuerySpecification` objects.
- Uses official `google-genai` SDK (`gemini-2.5-flash`).
- **Security & Sandboxing**:
  - Prompt injection detection blocks malicious override attempts upfront.
  - Strict allowlist enforcement ensures only approved operations and metrics are executed.
  - Model cannot generate or execute arbitrary Python, Cypher, SQL, or shell code.
- **Ambiguity & Domain Boundaries**:
  - Automatically identifies underspecified questions (e.g., "Compare movies") and requests user clarification without executing queries.
  - Detects out-of-domain questions (e.g., weather, general trivia) and cleanly returns explanatory rejections.
- **Telemetry & Cost Tracking**:
  - Distinguishes free-tier usage ($0.00 actual) from standard paid-tier equivalent cost.
  - Correct official pricing configured for `gemini-2.5-flash` ($0.30/1M input, $2.50/1M output).
  - Tracks API latency vs. Knowledge Graph query execution latency separately.

### Phase 4B: Natural Language Answer Generation
- Implemented `NaturalLanguageAnswerGenerator` (`src/answer_generator.py`) to convert structured graph results into clear, accurate, conversational markdown responses.
- **Strict Grounding & Hallucination Prevention**:
  - Model is instructed to base answers *strictly* on returned Knowledge Graph records.
  - Automated grounding check validates that movie titles and people cited in the answer are present in the structured graph results.
  - If a grounding violation or hallucination is detected, the generator automatically falls back to deterministic zero-shot template formatting.
- **Comprehensive Handling**:
  - Clarification, Unsupported, and No Results responses handled without unnecessary LLM calls.
  - Complete telemetry aggregation combining query parsing tokens/latency with answer generation tokens/latency.

### Phase 5: FastAPI Backend Development
- RESTful service exposing the complete Question Answering pipeline.
- **Endpoints**:
  - `GET /health`: Returns service status (`healthy`/`degraded`), version, graph loaded state, node/edge counts, Gemini mode (`LIVE_GEMINI` vs `MOCK`), model name, and pricing tier.
  - `POST /api/v1/ask`: Accepts natural language questions, runs the complete pipeline (parse -> query -> generate), and returns answers with metadata, execution metrics, and raw graph results.
  - `POST /api/v1/query`: Direct structured query execution for API consumers and testing.
  - `GET /api/v1/operations`: Returns the catalog of 15 supported operations with required/optional parameters and allowed metrics.
- **Architecture Highlights**:
  - Lifespan singleton management: NetworkX graph, Gemini client, query parser, and answer generator are pre-loaded once at startup.
  - Pure ASGI `RequestIdAndLoggingMiddleware` providing unique request IDs and request timing.
  - Full CORS configuration allowing cross-origin requests from frontend dev servers.
  - Interactive OpenAPI documentation available at `/docs` and ReDoc at `/redoc`.

### Phase 6: Frontend Development (MovieGraph AI Chat)
- Modern dark-themed UI built with React 18, TypeScript, Vite, and Tailwind CSS.
- **Key Features**:
  - **Conversational Chat Interface**: Smooth auto-scrolling chat history with formatted markdown rendering, copy answer buttons, and badge indicators for Clarification, Unsupported, and No Results.
  - **Collapsible Technical Telemetry Panel**: Detailed per-message inspection of detected operation, execution model, token usage, latency breakdown (parser, graph query, answer generation, total), estimated cost, and provider badge (`LIVE_GEMINI` vs `MOCK`).
  - **Raw Graph Results Viewer**: Expandable JSON inspector displaying the exact records retrieved from NetworkX.
  - **Knowledge Graph Insights Tab**: Live dashboard displaying 6 verified graph metrics (4,803 Movies, 102,159 People, 20 Genres, 9,561 Keywords, 5,051 Companies, 311,242 Directed Edges), directed relationship schema table, and 15 supported operations catalog.
  - **Sidebar & Session History**: Collapsible sidebar with New Chat button, session conversation management, quick suggested query chips, and backend health status indicator.

---

## Getting Started

### 1. Prerequisites
- **Python**: 3.10.8+
- **Node.js**: 18.0.0+ (and npm)

### 2. Environment Configuration
Copy the template and configure your Gemini API Key:
```bash
cp .env.example .env
```
Edit `.env` and set:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
*(Note: If no API key is set, tests run fully with mocks, and the demonstration scripts automatically run in Smart Mock Demo mode.)*

### 3. Backend Setup & Launch

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI backend server:

```powershell
# Using the PowerShell runner script (Recommended):
.\run_backend.ps1

# Or using the Batch runner script:
.\run_backend.bat

# Or invoking via Python module directly:
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:
- **API Base**: `http://127.0.0.1:8000`
- **Health Check**: `http://127.0.0.1:8000/health`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

### 4. Frontend Setup & Launch

Install Node packages:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```

Open your browser to:
```
http://localhost:5173
```

To create a production build of the frontend:
```bash
npm run build
```

---

## Pipelines & CLI Demos

### Run Data Preprocessing (Phase 1)
```bash
python run_preprocessing.py
```

### Build Knowledge Graph (Phase 2)
```bash
python run_graph_builder.py
```

### Run Query Engine Demo (Phase 3)
```bash
python run_query_demo.py
```

### Run Gemini Natural Language Parser Demo (Phase 4A)
```powershell
# Using the PowerShell runner script:
.\run_demo.ps1 --interactive

# Or using the Batch runner script:
.\run_demo.bat --interactive

# Or invoking Python directly:
python run_gemini_parser_demo.py --interactive
```

### Run Full End-to-End QA Pipeline Demo (Phase 4B)
```powershell
# Using the PowerShell runner script:
.\run_full_qa.ps1

# Or using the Batch runner script:
.\run_full_qa.bat

# Or invoking Python directly:
python run_full_qa_demo.py
```

---

## Example API Requests

### 1. Ask a Natural Language Question
```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about The Dark Knight"}'
```

### 2. Direct Structured Graph Query
```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "movies_by_genre",
    "parameters": {
      "genre": "Science Fiction",
      "metric": "vote_average",
      "limit": 5
    }
  }'
```

### 3. Check Operations Catalog
```bash
curl -X GET http://127.0.0.1:8000/api/v1/operations
```

---

## Automated Test Suites

The project includes 121 automated tests across the backend and frontend:

### Backend Unit Tests (109 Tests)
Run all backend tests with pytest:
```bash
pytest -v tests/test_preprocessing.py tests/test_graph.py tests/test_query_engine.py tests/test_query_parser.py tests/test_answer_generator.py
```

Test breakdown:
- `tests/test_preprocessing.py`: **13 passed** (data cleaning, validation, missing value preservation)
- `tests/test_graph.py`: **16 passed** (graph topology, node types, edge attributes, multi-role nodes)
- `tests/test_query_engine.py`: **33 passed** (15 operations, deterministic tie-breaking, sorting)
- `tests/test_query_parser.py`: **27 passed** (prompt injection defense, schema validation, allowlists)
- `tests/test_answer_generator.py`: **20 passed** (grounding validation, zero-shot fallback, telemetry)
- **Total Backend**: **109 passed, 0 failed**

### Frontend Unit Tests (12 Tests)
Run frontend tests with Vitest:
```bash
npx vitest run
```

Test breakdown:
- `src/__tests__/frontend.test.ts`: **12 passed** (formatting helpers, currency formatting, badge labels, markdown escaping, mock API contracts)
