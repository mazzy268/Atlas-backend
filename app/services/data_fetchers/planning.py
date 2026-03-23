"""
Planning Applications
Source: PlanningAlerts.org.uk API (free) + individual council APIs where available.
PlanningAlerts aggregates applications from 400+ UK councils.
Docs: https://www.planningalerts.org.uk/api/howto
"""
import httpx
from app.core.logging import get_logger

log = get_logger(__name__)

PLANNING_ALERTS_BASE = "https://www.planningalerts.org.au/api/v2"
# Note: PlanningAlerts UK mirror:
UK_PLANNING_BASE = "https://api.planningalerts.org.uk/applications.js"


async def fetch(lat: float, lng: float, radius_m: int | None = None) -> dict:
    """
    Fetch recent planning applications near a lat/lng.
    Uses PlanningAlerts UK API — free tier allows ~100 req/day.
    """
    radius_m = radius_m or 500

    try:
        applications = await _fetch_planning_alerts(lat, lng, radius_m)
    except Exception as e:
        log.warning("planning_alerts_failed", error=str(e))
        applications = []

    # Classify applications
    classified = [_classify_application(a) for a in applications]

    risk_level = _assess_risk(classified)

    log.info("planning_fetched", lat=lat, lng=lng, count=len(classified))
    return {
        "total_applications": len(classified),
        "applications": classified,
        "risk_level": risk_level,
        "radius_m": radius_m,
    }


async def _fetch_planning_alerts(lat: float, lng: float, radius_m: int) -> list[dict]:
    """
    PlanningAlerts API — returns applications within radius.
    radius is in metres; API uses 'lng' not 'long'.
    """
    params = {
        "lat": lat,
        "lng": lng,
        "radius": radius_m / 1000,   # convert to km for some endpoints
        "count": 20,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(UK_PLANNING_BASE, params=params)
        if resp.status_code in (404, 403, 500):
            return _fallback_open_planning(lat, lng)
        resp.raise_for_status()
        data = resp.json()
        return data.get("applications", data if isinstance(data, list) else [])


def _fallback_open_planning(lat: float, lng: float) -> list[dict]:
    """Return empty list — planning data unavailable for this area."""
    return []


def _classify_application(app: dict) -> dict:
    desc = (app.get("description") or app.get("info") or "").lower()
    app_type = "Other"

    keywords = {
        "HMO / licensing": ["hmo", "house in multiple", "sui generis"],
        "Extension": ["extension", "conservatory", "outbuilding", "garage conversion"],
        "Loft conversion": ["loft", "dormer", "roof"],
        "New dwelling": ["new dwelling", "new house", "residential development"],
        "Change of use": ["change of use", "convert", "conversion"],
        "Commercial": ["commercial", "office", "retail", "mixed use"],
        "Demolition": ["demolition", "demolish"],
        "Advertisement": ["advertisement", "signage"],
    }
    for label, words in keywords.items():
        if any(w in desc for w in words):
            app_type = label
            break

    return {
        "reference": app.get("council_reference") or app.get("uid", ""),
        "description": app.get("description") or app.get("info", "")[:300],
        "status": app.get("status") or app.get("decision", "Pending"),
        "decision_date": app.get("decision_date") or app.get("date_validated"),
        "address": app.get("address", ""),
        "application_type": app_type,
        "distance_m": app.get("distance"),
        "url": app.get("url", ""),
    }


def _assess_risk(applications: list[dict]) -> str:
    """Assess planning risk level based on nearby applications."""
    if not applications:
        return "low"
    high_risk_types = {"New dwelling", "Commercial", "Demolition", "HMO / licensing"}
    count_risky = sum(1 for a in applications if a.get("application_type") in high_risk_types)
    if count_risky >= 3:
        return "high"
    if count_risky >= 1:
        return "medium"
    return "low"
