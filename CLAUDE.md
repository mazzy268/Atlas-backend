# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Two coexisting apps — always know which one you're touching

| File | DB | AI | Tests target it? |
|------|----|----|-----------------|
| `dashboard_main.py` | None (stateless) | HuggingFace `mistralai/Mistral-7B-Instruct-v0.2` | No |
| `app/main.py` | PostgreSQL via SQLAlchemy + asyncpg | OpenAI | Yes |

**Production entry point is `dashboard_main.py`.** The `app/` directory is a modular alternative that requires Postgres and is what `tests/test_endpoints.py` imports. Changes to one do not affect the other.

---

## Commands

```bash
# Run production app (no DB)
uvicorn dashboard_main:app --reload --port 8000

# Run modular app (requires Postgres)
docker-compose up -d postgres
uvicorn app.main:app --reload --port 8000

# Tests (hit real external APIs — need internet)
pytest tests/ -v
pytest tests/test_endpoints.py::test_health -v   # single test

# Full AI test (uses OpenAI credits)
TEST_WITH_AI=1 pytest tests/ -v

# Refresh static market data
python scripts/daily_update.py
```

Swagger UI: `http://localhost:8000/docs`

---

## Environment variables

```
HUGGINGFACE_API_KEY   # required for AI summary in dashboard_main.py
EPC_API_KEY           # required for EPC data; without it, property details fall back to defaults
EPC_API_EMAIL         # email registered with EPC API
OPENAI_API_KEY        # app/ module only
DATABASE_URL          # app/ module only
```

---

## Architecture: `dashboard_main.py` (production)

All logic is in one ~3 000-line file. `POST /analyse-property` is the only endpoint Lovable calls.

**Request → response flow:**

1. Parse body — accepts `{"address": "..."}` or `{"postcode": "..."}` (also tolerates bare strings and malformed JSON)
2. `_geocode()` → Nominatim → lat/lng/region
3. `asyncio.gather` fans out to 8 fetchers: `_fetch_sales`, `_fetch_epc`, `_fetch_crime`, `_fetch_demographics`, `_fetch_flood`, `_fetch_transport`, `_fetch_planning_data`, `_fetch_schools`
4. `_fetch_ukhpi_data()` called sequentially after geocoding (needs district name from demographics)
5. Pure calculation functions derive all scores and financials
6. `_run_ai()` → HuggingFace with `_hf_fallback()` on failure
7. Return one large JSON dict

**Static lookup tables** (update these when refreshing market data):
- `VOA_RENTS` — median monthly rents by region × bedroom count (VOA 2023-24)
- `ONS_GROWTH` — annual HPI % by region (ONS November 2024)
- `REGIONAL_YIELDS` — gross yield benchmarks by region
- `PROP_TYPE_MULTIPLIER` — rent multiplier by property type

**Score functions** (all pure, deterministic):
- `_investment_score`, `_risk_score`, `_liquidity_score`, `_deal_score_calc`, `_rental_demand_score`, `_street_score`

**Bedroom inference:** majority-vote via `_consensus_bedrooms()` across all EPC records for postcode; `_infer_bedrooms()` for a single targeted EPC record. Falls back to `3` (UK median) if no EPC data.

---

## `/analyse-property` response structure

Top-level keys in order:

```
postcode, display_address, latitude, longitude, generated_at
property          — bedrooms, floor_area, type, EPC details, tenure, construction era
financials        — value, rent, yield, cashflow, mortgage, deposit, SDLT, mortgage_scenarios
scores            — investment, risk, liquidity, street, deal, rental_demand
growth            — 1/3/5yr projections, annual_growth_rate_pct, is_projection: true
market            — UKHPI district/type averages, trend, PSM
ai_analysis       — best_strategy, all_strategies, key_positives, key_risks, summary
renovation        — light/medium/heavy scenarios + EPC upgrade
renovation_scenarios, best_scenario, renovation_summary
development       — loft, extension, HMO viability and costs
risk              — flood, crime, IMD, red_flags
neighbourhood     — transport, schools, demographics, desirability
planning          — Article 4, conservation area, listed building, HMO block
comparables       — last 5 sales from Land Registry
deals             — deal score, potential deals, best deal
hmo_analysis      — room count, yield, cashflow vs BTL
tax_analysis, brrrr_analysis, exit_analysis, ten_year_model
confidence        — per-source confidence levels + overall score
data_validation   — warnings, sanity flags
data_freshness    — source dates per field
data_age          — freshness label per source
deal_breakdown, investor_decision, area_ranking, exit_strategy
data_sources      — boolean flags for each live source + active_count
disclaimer        — not_financial_advice: true, analysis_limitations[], confidence_explanation, regulatory_notice
```

**Do not rename or remove existing keys** — Lovable's PropertyContext reads all of them. Adding new keys is safe.

---

## Other endpoints in `dashboard_main.py`

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/analyse-property` | Main — full analysis |
| `POST` | `/compare-properties` | Returns field list for client-side comparison; does not call analyse internally |
| `GET`  | `/market-heatmap` | Opportunity score for a location |
| `GET`  | `/deal-scanner` | Deal score for a postcode |
| `GET`  | `/risk-analysis` | Flood + crime risk only |
| `GET`  | `/true-value` | Valuation with confidence range |
| `GET`  | `/liquidity-score` | Time-to-sell estimate |
| `GET`  | `/development-potential` | Loft/extension/HMO viability |
| `GET`  | `/portfolio` | Returns `{"total": 0, "properties": []}` — in-memory stub |
| `POST` | `/portfolio/add` | Returns `{"status": "success"}` — stub |
| `DELETE` | `/portfolio/{id}` | Returns `{"status": "removed"}` — stub |

Portfolio is permanently a stub. Do not add a database.

---

## Architecture: `app/` module (modular/DB-backed)

Layered: `app/api/endpoints/` → `app/services/ai_analysis/report_builder.py` → `app/services/data_fetchers/` → `app/db/session.py`

Reports are cached in Postgres with a TTL; pass `force_refresh=True` to bypass. To add a new data source: create `app/services/data_fetchers/my_source.py` with `async def fetch(...) -> dict`, call it in `gather_all_data()` in `report_builder.py`, add a prompt in `prompts.py`.

Tests import from `app.main` — they will not test `dashboard_main.py` behaviour.

---

## Critical rules

- **No database in `dashboard_main.py`** — no SQLAlchemy, no sessions, no connections
- **Single endpoint** — all Lovable widgets depend on one call to `POST /analyse-property`
- **AI must never crash the API** — always return a fallback summary
- **All external calls wrapped in try/except** — return defaults, never propagate exceptions to 500
- **Disclaimer block is required** — `disclaimer.not_financial_advice` must remain `true`; do not present projected values as facts
- **Do not add**: authentication, payments, database, or new external dependencies

---

## External APIs (free, no key)

- **Nominatim** — geocoding
- **postcodes.io** — region/ward/IMD lookup
- **HM Land Registry SPARQL** — price paid data
- **data.police.uk** — crime
- **Environment Agency** — flood warnings
- **Overpass API** — transport/stations via OSM
- **GOV.UK Planning Data API** — Article 4, conservation, listed buildings
