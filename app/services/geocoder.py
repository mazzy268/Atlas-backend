"""
Geocoding service — converts UK addresses to lat/lng using OpenStreetMap Nominatim.
No API key required. Respects OSM's 1 req/sec rate limit via tenacity.
"""
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def geocode_address(address: str) -> dict:
    """
    Returns {"latitude": float, "longitude": float, "display_name": str}
    Raises ValueError if address cannot be resolved.
    """
    params = {
        "q": address,
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
        "countrycodes": "gb",
    }
    headers = {"User-Agent": settings.nominatim_user_agent}

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(NOMINATIM_URL, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()

    if not results:
        raise ValueError(f"Could not geocode address: {address!r}")

    best = results[0]
    log.info("geocoded", address=address, lat=best["lat"], lon=best["lon"])

    return {
        "latitude": float(best["lat"]),
        "longitude": float(best["lon"]),
        "display_name": best.get("display_name", address),
        "postcode": _extract_postcode(best.get("address", {})),
    }


def _extract_postcode(address_detail: dict) -> str | None:
    return address_detail.get("postcode")
