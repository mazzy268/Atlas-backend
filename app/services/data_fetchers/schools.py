"""
Schools data
Source: DfE Get Information About Schools (GIAS) + Ofsted ratings
Free, no API key required.
Docs: https://get-information-schools.service.gov.uk/
"""
import httpx
import math
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

GIAS_BASE = "https://ea-edubase-api-prod.azurewebsites.net/edubase/downloads/public"
OFSTED_API = "https://api.ofsted.gov.uk/v1"

# We use Edubase's postcode-based search endpoint
SCHOOLS_SEARCH = "https://www.get-information-schools.service.gov.uk/api/schools"


async def fetch(lat: float, lng: float, radius_m: int | None = None) -> dict:
    """Fetch nearby schools with Ofsted ratings."""
    radius_m = radius_m or settings.schools_radius_m

    schools = await _fetch_nearby_schools(lat, lng, radius_m)

    log.info("schools_fetched", lat=lat, lng=lng, count=len(schools))
    return {
        "total_schools": len(schools),
        "schools": schools,
        "radius_m": radius_m,
        "best_school": _best_school(schools),
    }


async def _fetch_nearby_schools(lat: float, lng: float, radius_m: int) -> list[dict]:
    """
    Use postcodes.io to get nearby postcodes, then query GIAS.
    GIAS public API doesn't support direct lat/lng radius queries,
    so we use a public search endpoint.
    """
    try:
        # Get nearby postcodes
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.postcodes.io/postcodes",
                params={"lon": lng, "lat": lat, "radius": min(radius_m, 2000), "limit": 10},
            )
            resp.raise_for_status()
            nearby_postcodes = [p["postcode"] for p in resp.json().get("result", [])]

        schools = []
        seen_urns = set()

        for pc in nearby_postcodes[:3]:    # limit API calls
            pc_schools = await _schools_for_postcode(pc)
            for s in pc_schools:
                urn = s.get("urn")
                if urn and urn not in seen_urns:
                    seen_urns.add(urn)
                    # Calculate distance
                    s["distance_m"] = _haversine(lat, lng, s.get("lat", lat), s.get("lng", lng))
                    schools.append(s)

        # Sort by distance, cap at 10
        schools.sort(key=lambda x: x.get("distance_m", 9999))
        return schools[:10]

    except Exception as e:
        log.warning("schools_fetch_failed", error=str(e))
        return []


async def _schools_for_postcode(postcode: str) -> list[dict]:
    """Query GIAS for schools near a postcode."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.get-information-schools.service.gov.uk/search/results-json",
                params={
                    "SearchType": "0",
                    "Postcode": postcode,
                    "Distance": "1",
                    "SelectedTab": "All",
                },
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [_parse_school(s) for s in data.get("Establishments", [])[:10]]
    except Exception:
        return []


def _parse_school(s: dict) -> dict:
    return {
        "urn": str(s.get("Urn", "")),
        "name": s.get("EstablishmentName", ""),
        "type": s.get("TypeOfEstablishment", {}).get("DisplayName", ""),
        "phase": s.get("PhaseOfEducation", {}).get("DisplayName", ""),
        "ofsted_rating": s.get("OfstedRating", {}).get("DisplayName", "Not inspected"),
        "ofsted_date": s.get("OfstedLastInsp", ""),
        "postcode": s.get("Postcode", ""),
        "lat": float(s.get("Latitude", 0) or 0),
        "lng": float(s.get("Longitude", 0) or 0),
        "open_date": s.get("OpenDate", ""),
        "pupils": s.get("NumberOfPupils"),
    }


def _best_school(schools: list[dict]) -> dict | None:
    priority = {"Outstanding": 1, "Good": 2, "Requires improvement": 3, "Inadequate": 4}
    rated = [s for s in schools if s.get("ofsted_rating") in priority]
    if not rated:
        return schools[0] if schools else None
    return min(rated, key=lambda s: priority[s["ofsted_rating"]])


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
