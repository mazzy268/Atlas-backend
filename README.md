# Atlas Property Intelligence API

A modular FastAPI backend that analyses any UK property and returns a full AI-powered investment intelligence report.

## Features

- Address → coordinates via OpenStreetMap Nominatim
- 8 live data sources (HMLR, EPC, Police, ONS, EA Flood, Planning, Schools, Transport)
- 12 AI-powered report features (Investment Score, Strategy Detector, ROI, etc.)
- PostgreSQL / Supabase for persistent storage
- Daily data refresh scripts
- Fully modular — add new data sources or report features with minimal changes

## Project Structure

```
atlas-property-api/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   └── endpoints/
│   │       ├── property.py      # POST /analyse-property
│   │       ├── sales.py         # GET /property-sales
│   │       ├── crime.py         # GET /crime-data
│   │       ├── demographics.py  # GET /demographics
│   │       └── flood.py         # GET /flood-risk
│   ├── core/
│   │   ├── config.py            # Settings & env vars
│   │   └── logging.py           # Structured logging
│   ├── models/
│   │   └── database.py          # SQLAlchemy ORM models
│   ├── schemas/
│   │   └── property.py          # Pydantic request/response schemas
│   ├── services/
│   │   ├── geocoder.py          # OSM Nominatim geocoding
│   │   ├── data_fetchers/
│   │   │   ├── land_registry.py
│   │   │   ├── epc.py
│   │   │   ├── crime.py
│   │   │   ├── demographics.py
│   │   │   ├── flood_risk.py
│   │   │   ├── planning.py
│   │   │   ├── schools.py
│   │   │   └── transport.py
│   │   └── ai_analysis/
│   │       ├── report_builder.py  # Orchestrates all 12 features
│   │       ├── prompts.py         # All AI prompts
│   │       └── openai_client.py   # LLM wrapper
│   └── db/
│       └── session.py             # DB connection
├── scripts/
│   ├── daily_update.py            # Cron job for daily data refresh
│   └── init_db.py                 # Create tables
├── tests/
│   └── test_endpoints.py
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Quickstart

### 1. Clone and install

```bash
git clone <repo>
cd atlas-property-api
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your API keys (see below)
```

### 3. Start the database

```bash
docker-compose up -d postgres
```

Or point `DATABASE_URL` at your Supabase connection string.

### 4. Initialise the database

```bash
python scripts/init_db.py
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

## API Keys Required

| Service | Where to get it | Env var |
|---------|----------------|---------|
| OpenAI | platform.openai.com | `OPENAI_API_KEY` |
| EPC / DLUHC | epc.opendatacommunities.org | `EPC_API_KEY` |
| OS Places (optional) | osdatahub.os.uk | `OS_PLACES_API_KEY` |

All other sources (HMLR, Police API, ONS, EA Flood, DfE Schools, TfL) are **free and require no key**.

## Example Request

```bash
curl -X POST http://localhost:8000/analyse-property \
  -H "Content-Type: application/json" \
  -d '{"address": "10 Downing Street, London, SW1A 2AA"}'
```

## Daily Data Refresh

```bash
python scripts/daily_update.py
```

Or add to cron:
```
0 2 * * * /path/to/venv/bin/python /path/to/scripts/daily_update.py
```

## Adding a New Data Source

1. Create `app/services/data_fetchers/my_source.py` implementing `async def fetch(lat, lng, **kwargs) -> dict`
2. Import and call it in `app/services/ai_analysis/report_builder.py` inside `gather_all_data()`
3. Add a prompt for it in `app/services/ai_analysis/prompts.py`
4. Done — no other files need changing.
