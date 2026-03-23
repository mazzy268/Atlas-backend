"""
UK Police Crime Data
Source: data.police.uk — free, no API key required.
Docs: https://data.police.uk/docs/
"""
import httpx
from collections import Counter
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

POLICE_BASE = "https://data.police.uk/api"


async def fetch(lat: float, lng: float, radius_m: int | None = None) -> dict:
    """
    Fetch crimes within radius of lat/lng.
    Returns aggregated counts by category plus raw incidents.
    """
    radius_m = radius_m or settings.crime_radius_m

    # police API uses 'radius' in miles but accepts lat/lng + date
    url = f"{POLICE_BASE}/crimes-street/all-crime"
    params = {"lat": lat, "lng": lng}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        if resp.status_code == 503:
            return {"error": "Police API temporarily unavailable", "crimes": []}
        resp.raise_for_status()
        crimes = resp.json()

    if not isinstance(crimes, list):
        return {"error": "Unexpected response format", "crimes": []}

    category_counts = Counter(c.get("category", "unknown") for c in crimes)
    by_category = [
        {"category": cat, "count": count}
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
    ]

    # Get date range from results
    dates = [c.get("month", "") for c in crimes if c.get("month")]
    period = f"{min(dates)} to {max(dates)}" if dates else "unknown"

    log.info("crime_data_fetched", lat=lat, lng=lng, total=len(crimes))
    return {
        "total_crimes": len(crimes),
        "by_category": by_category,
        "period": period,
        "crimes": crimes[:50],       # cap raw output
    }


async def fetch_categories() -> list[dict]:
    """Return all available crime categories."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{POLICE_BASE}/crime-categories")
        resp.raise_for_status()
        return resp.json()
