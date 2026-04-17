# CLAUDE.md

# Atlas Property Intelligence – Claude Context

## Overview

Atlas is a UK-based property intelligence SaaS platform designed for investors.

The system takes a property address or postcode and returns a full investment analysis including:

* investment score
* deal analysis
* rental yield
* growth forecasts
* risk analysis
* development potential
* AI-generated summary

The frontend is built in Lovable.
The backend is built with FastAPI and deployed on Railway.

---

## Core Architecture

### Backend

* Framework: FastAPI
* Main file: dashboard_main.py
* Deployment: Railway
* API base URL: production Railway URL
* Main endpoint: POST /analyse-property

### Frontend

* Built in Lovable
* Uses a single API call
* Expects ALL data from one endpoint

---

## CRITICAL RULES (DO NOT BREAK THESE)

### 1. NO DATABASE

* This system is stateless
* DO NOT use:

  * SQLAlchemy
  * SessionLocal
  * database connections
* Any portfolio features must return mock responses:

  * GET /portfolio → []
  * POST /portfolio/add → {"status": "success"}

---

### 2. SINGLE SOURCE OF TRUTH

* ALL frontend widgets depend on ONE API call:
  POST /analyse-property
* DO NOT create multiple dependent endpoints for core data
* Response must contain ALL required fields

---

### 3. INPUT FORMAT

The API must accept:

{
"address": "string"
}

OR

{
"postcode": "string"
}

Internally normalize:
input_location = address or postcode

---

### 4. AI SYSTEM

* Use Hugging Face (NOT OpenAI, NOT Groq)
* Model:
  mistralai/Mistral-7B-Instruct-v0.2
* API key from environment:
  HUGGINGFACE_API_KEY

Requirements:

* Always return ai_summary
* If AI fails → return fallback summary
* NEVER crash API due to AI

---

### 5. ERROR HANDLING

* Wrap ALL external API calls in try/except
* If any data source fails:

  * return fallback/default values
  * DO NOT crash

---

### 6. RESPONSE STRUCTURE

The /analyse-property endpoint MUST always return:

* property (address, postcode, type, floor area)
* scores (investment, deal, risk, liquidity)
* financials (yield, rent, value, cashflow)
* growth (1yr, 3yr, 5yr)
* development (loft, extension, ROI)
* neighbourhood (crime, schools, transport)
* ai_summary

No missing keys. No mock London data.

---

### 7. FRONTEND COMPATIBILITY

* Lovable uses a global PropertyContext
* ALL widgets read from ONE response
* Do NOT rename fields randomly
* Do NOT change structure without mapping

---

## CURRENT GOALS

Claude must prioritise:

1. Fix backend errors (no 500 errors)
2. Ensure /analyse-property works reliably
3. Ensure response is complete and consistent
4. Ensure compatibility with Lovable frontend
5. Remove ALL hardcoded/mock data
6. Ensure ALL widgets can update dynamically

---

## NON-GOALS (DO NOT DO)

* Do NOT add authentication
* Do NOT add payments
* Do NOT add database
* Do NOT restructure frontend
* Do NOT introduce unnecessary complexity

---

## SUCCESS DEFINITION

The system is complete when:

* API returns 200 consistently
* No crashes on Railway
* Lovable dashboard updates ALL widgets from ONE search
* No mock data remains
* AI summary works or gracefully falls back

---

## INSTRUCTIONS FOR CLAUDE

When modifying code:

* Make minimal, precise changes
* Do not break existing working features
* Always explain what was changed
* Always prioritise stability over new features

If unsure:

* default to simpler implementation
* avoid adding dependencies

---

End of file.

This repo has two separate FastAPI apps that coexist:

| File | DB | AI | Status |
|------|----|----|--------|
| `dashboard_main.py` | None (stateless) | HuggingFace Inference API (`mistralai/Mistral-7B-Instruct-v0.2`) | Production entry point |
| `app/main.py` | PostgreSQL via SQLAlchemy + asyncpg | OpenAI | Modular/extensible version |

**Default for deployment is `dashboard_main.py`** — it's zero-dependency (no DB), self-contained, and built for live UK government API calls. The `app/` directory is a more modular alternative that requires Postgres.

## Running the app

```bash
# Stateless version (no DB required)
uvicorn dashboard_main:app --reload --port 8000

# Modular version (requires Postgres)
docker-compose up -d postgres
uvicorn app.main:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

## Running tests

```bash
pytest tests/ -v

# Include full AI integration test (uses OpenAI credits, hits all live APIs)
TEST_WITH_AI=1 pytest tests/ -v

# Run a single test
pytest tests/test_endpoints.py::test_health -v
```

Tests in `tests/test_endpoints.py` hit real external APIs — they require internet access.

## Environment variables

```
HUGGINGFACE_API_KEY   # dashboard_main.py AI (free, huggingface.co/settings/tokens)
EPC_API_KEY           # EPC energy cert data (epc.opendatacommunities.org)
EPC_API_EMAIL         # email registered with EPC API
OPENAI_API_KEY        # app/ module AI (platform.openai.com)
DATABASE_URL          # app/ module only (or use docker-compose postgres)
```

EPC is the only key that affects `dashboard_main.py` beyond AI — without it, EPC data returns empty and property details fall back to defaults.

## Architecture: `dashboard_main.py`

All logic lives in a single file. The main endpoint `POST /analyse-property` orchestrates the flow:

1. **Geocode** postcode via Nominatim → lat/lng + region
2. **Fan out** via `asyncio.gather` to 6 data fetchers: `_fetch_sales`, `_fetch_epc`, `_fetch_crime`, `_fetch_demographics`, `_fetch_flood`, `_fetch_transport`
3. **Calculate** scores and financials using pure functions (`_calc_value`, `_investment_score`, `_risk_score`, etc.)
4. **AI summary** via `_run_ai` → `generate_ai_summary` → HuggingFace, with a rule-based `_hf_fallback` when the key is absent
5. **Return** a large JSON response covering property, financials, scores, growth, risk, neighbourhood, comparables, deals, HMO, renovation, development

Static lookup tables (`VOA_RENTS`, `ONS_GROWTH`) contain hardcoded 2024 UK regional data — update these when refreshing market data.

Portfolio is **in-memory only** (`portfolio_store: list[dict]`) — it resets on server restart.

## Architecture: `app/` module

Follows a layered structure: `api/endpoints/` → `services/ai_analysis/report_builder.py` → `services/data_fetchers/` → DB via `db/session.py`. Reports are cached in Postgres with a TTL; `force_refresh=True` bypasses the cache.

**To add a new data source to `app/`:**
1. Create `app/services/data_fetchers/my_source.py` with `async def fetch(...) -> dict`
2. Call it inside `gather_all_data()` in `app/services/ai_analysis/report_builder.py`
3. Add a prompt in `app/services/ai_analysis/prompts.py`

## External APIs (all free, no key needed)

- **Nominatim** (OpenStreetMap) — geocoding
- **postcodes.io** — region, ward, IMD decile lookup
- **HM Land Registry SPARQL** (`landregistry.data.gov.uk`) — price paid data
- **data.police.uk** — street-level crime
- **Environment Agency** (`environment.data.gov.uk`) — flood warnings
- **Overpass API** — transport/stations via OSM tags
