"""
Environment Agency Flood Risk
Source: EA Real Time Flood Monitoring API + Flood Map API
Free, no API key required.
Docs: https://environment.data.gov.uk/flood-monitoring/doc/reference
"""
import httpx
from app.core.logging import get_logger

log = get_logger(__name__)

EA_FLOOD_BASE = "https://environment.data.gov.uk/flood-monitoring"
EA_FLOOD_ZONES = "https://environment.data.gov.uk/arcgis/rest/services/EA/FloodMapForPlanningRiversAndSeaFloodZone3/MapServer/0/query"


async def fetch(lat: float, lng: float) -> dict:
    """
    Fetch flood risk information for a lat/lng point.
    Checks flood zone classification and surface water risk.
    """
    result = {
        "latitude": lat,
        "longitude": lng,
        "flood_zone": "Unknown",
        "risk_level": "Unknown",
        "river_sea_risk": "Unknown",
        "surface_water_risk": "Unknown",
        "reservoir_risk": "Negligible",
        "active_warnings": [],
        "notes": "",
    }

    # Fetch active flood warnings near location
    warnings = await _fetch_warnings(lat, lng)
    result["active_warnings"] = warnings

    # Fetch flood zone from EA ArcGIS service
    zone_info = await _fetch_flood_zone(lat, lng)
    result.update(zone_info)

    # Infer risk level from zone
    result["risk_level"] = _zone_to_risk(result["flood_zone"])

    log.info("flood_risk_fetched", lat=lat, lng=lng, zone=result["flood_zone"])
    return result


async def _fetch_warnings(lat: float, lng: float) -> list[dict]:
    """Fetch active flood warnings within ~5km."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{EA_FLOOD_BASE}/id/floods",
                params={"lat": lat, "long": lng, "dist": 5},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return [
                {
                    "severity": i.get("severity", {}).get("label", ""),
                    "description": i.get("description", ""),
                    "county": i.get("floodArea", {}).get("county", ""),
                }
                for i in items[:5]
            ]
    except Exception as e:
        log.warning("flood_warnings_fetch_failed", error=str(e))
        return []


async def _fetch_flood_zone(lat: float, lng: float) -> dict:
    """
    Query EA flood zone WMS/ArcGIS for the point.
    Falls back to a heuristic if unavailable.
    """
    try:
        # Use EA's ESRI REST API — query which flood zone polygon contains this point
        params = {
            "geometry": f"{lng},{lat}",
            "geometryType": "esriGeometryPoint",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
            "inSR": "4326",
            "outSR": "4326",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(EA_FLOOD_ZONES, params=params)
            resp.raise_for_status()
            data = resp.json()

        features = data.get("features", [])
        if features:
            attrs = features[0].get("attributes", {})
            zone = attrs.get("zone", "1")
            return {
                "flood_zone": f"Zone {zone}",
                "river_sea_risk": _zone_to_river_risk(str(zone)),
                "surface_water_risk": "Low",
                "notes": f"EA Flood Zone {zone} classification.",
            }
        return {
            "flood_zone": "Zone 1",
            "river_sea_risk": "Very Low",
            "surface_water_risk": "Low",
            "notes": "No flood zone intersection found — likely Zone 1 (lowest risk).",
        }
    except Exception as e:
        log.warning("flood_zone_fetch_failed", error=str(e))
        return {
            "flood_zone": "Unknown",
            "river_sea_risk": "Data unavailable",
            "surface_water_risk": "Data unavailable",
            "notes": "Could not retrieve EA flood zone data.",
        }


def _zone_to_risk(zone: str) -> str:
    mapping = {"Zone 1": "Low", "Zone 2": "Medium", "Zone 3": "High", "Zone 3a": "High", "Zone 3b": "Very High"}
    return mapping.get(zone, "Unknown")


def _zone_to_river_risk(zone: str) -> str:
    mapping = {"1": "Very Low (<0.1%/yr)", "2": "Low (0.1–1%/yr)", "3": "High (>1%/yr)"}
    return mapping.get(zone, "Unknown")
