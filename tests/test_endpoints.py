"""
tests/test_endpoints.py
Integration tests for the Atlas API endpoints.
Run with: pytest tests/ -v

These tests use real external APIs where possible.
Set TEST_WITH_AI=1 to also test AI features (uses OpenAI credits).
"""
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Load .env for test environment
from dotenv import load_dotenv
load_dotenv()

from app.main import app

TEST_WITH_AI = os.getenv("TEST_WITH_AI", "0") == "1"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Health check ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Data endpoints ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_property_sales(client):
    resp = await client.get("/property-sales", params={"postcode": "SW1A 2AA", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "sales" in data
    assert isinstance(data["sales"], list)


@pytest.mark.asyncio
async def test_crime_data(client):
    # Westminster
    resp = await client.get("/crime-data", params={"latitude": 51.5014, "longitude": -0.1419})
    assert resp.status_code == 200
    data = resp.json()
    assert "total_crimes" in data
    assert "by_category" in data


@pytest.mark.asyncio
async def test_demographics(client):
    resp = await client.get("/demographics", params={"postcode": "SW1A 2AA"})
    assert resp.status_code == 200
    data = resp.json()
    assert "postcode" in data
    assert "local_authority" in data


@pytest.mark.asyncio
async def test_flood_risk(client):
    # Non-flood area (Central London)
    resp = await client.get("/flood-risk", params={"latitude": 51.5014, "longitude": -0.1419})
    assert resp.status_code == 200
    data = resp.json()
    assert "flood_zone" in data
    assert "risk_level" in data


# ── Geocoding unit tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_geocode():
    from app.services.geocoder import geocode_address
    result = await geocode_address("10 Downing Street, London, SW1A 2AA")
    assert result["latitude"] == pytest.approx(51.503, abs=0.01)
    assert result["longitude"] == pytest.approx(-0.127, abs=0.01)
    assert result["postcode"] is not None


# ── Data fetcher unit tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_land_registry_fetch():
    from app.services.data_fetchers.land_registry import fetch
    result = await fetch("SW1A 2AA", limit=5)
    assert "sales" in result
    # HMLR may return 0 results for this postcode but should not error
    assert isinstance(result["sales"], list)


@pytest.mark.asyncio
async def test_flood_fetcher():
    from app.services.data_fetchers.flood_risk import fetch
    result = await fetch(51.5014, -0.1419)
    assert "flood_zone" in result
    assert "risk_level" in result


@pytest.mark.asyncio
async def test_transport_fetcher():
    from app.services.data_fetchers.transport import fetch
    result = await fetch(51.5014, -0.1419, radius_m=500)
    assert "transport_score" in result
    assert 0 <= result["transport_score"] <= 10


@pytest.mark.asyncio
async def test_crime_fetcher():
    from app.services.data_fetchers.crime import fetch
    result = await fetch(51.5014, -0.1419)
    assert "total_crimes" in result
    assert "by_category" in result


# ── Full report integration test (requires OpenAI key + TEST_WITH_AI=1) ────────

@pytest.mark.asyncio
@pytest.mark.skipif(not TEST_WITH_AI, reason="Set TEST_WITH_AI=1 to run full report test")
async def test_full_report(client):
    resp = await client.post(
        "/analyse-property",
        json={"address": "1 Whitehall, London, SW1A 2AA", "force_refresh": True},
        timeout=120.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    report = data["report"]

    # Check all 12 features present
    assert "investment_score" in report
    assert "strategy_detector" in report
    assert "renovation_predictor" in report
    assert "floorplan_analysis" in report
    assert "neighbourhood_intelligence" in report
    assert "rental_demand_score" in report
    assert "planning_scanner" in report
    assert "deal_finder" in report
    assert "price_growth_predictor" in report
    assert "rental_yield_simulator" in report
    assert "ai_summary" in report

    # Investment score should be 0-100
    score = report["investment_score"].get("score")
    assert score is not None
    assert 0 <= score <= 100

    print(f"\n✓ Investment Score: {score}/100")
    print(f"✓ Strategy: {report['strategy_detector'].get('primary_strategy')}")
    print(f"✓ Summary: {report['ai_summary'][:200]}")
