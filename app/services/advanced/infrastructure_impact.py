"""
Infrastructure Impact Detector
Analyses nearby infrastructure signals and predicts value impact.
"""
import httpx
import asyncio
from app.core.logging import get_logger

log = get_logger(__name__)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def analyse_infrastructure(lat: float, lng: float, postcode: str) -> dict:
    transport, regen, geo = await asyncio.gather(
        _fetch_planned_transport(lat, lng),
        _fetch_regeneration_signals(lat, lng),
        _fetch_geo(postcode),
        return_exceptions=True,
    )

    transport_data = transport if not isinstance(transport, Exception) else []
    regen_data = regen if not isinstance(regen, Exception) else []
    geo_data = geo if not isinstance(geo, Exception) else {}

    transport_boost = _calc_transport_boost(transport_data)
    regen_boost = _calc_regen_boost(regen_data, geo_data)
    total_boost = min(transport_boost + regen_boost, 25.0)

    return {
        "latitude": lat,
        "longitude": lng,
        "postcode": postcode,
        "infrastructure_growth_boost": round(total_boost, 1),
        "boost_label": _boost_label(total_boost),
        "signals": {
            "transport": {
                "nearby_stations": len(transport_data),
                "boost_contribution_pct": round(transport_boost, 1),
                "items": transport_data[:5],
            },
            "regeneration": {
                "signals_detected": len(regen_data),
                "boost_contribution_pct": round(regen_boost, 1),
                "items": regen_data[:5],
            },
        },
        "investment_implication": _investment_implication(total_boost),
        "confidence": "medium",
        "note": "Infrastructure boost is an indicative forward-looking estimate based on current signals.",
    }


async def _fetch_planned_transport(lat: float, lng: float) -> list:
    query = f"""
    [out:json][timeout:10];
    (
      node["railway"~"station|halt|tram_stop"](around:2000,{lat},{lng});
      node["public_transport"="station"](around:2000,{lat},{lng});
    );
    out body;
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            return [
                {
                    "name": e.get("tags", {}).get("name", "Unnamed"),
                    "type": e.get("tags", {}).get("railway", "station"),
                    "distance_m": _approx_distance(lat, lng, e.get("lat", lat), e.get("lon", lng)),
                }
                for e in elements
            ]
    except Exception:
        return []


async def _fetch_regeneration_signals(lat: float, lng: float) -> list:
    query = f"""
    [out:json][timeout:10];
    (
      node["landuse"~"construction|brownfield"](around:1000,{lat},{lng});
      way["landuse"~"construction|brownfield"](around:1000,{lat},{lng});
    );
    out body;
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            return [
                {
                    "type": e.get("tags", {}).get("landuse", "development"),
                    "name": e.get("tags", {}).get("name", "Construction site"),
                }
                for e in elements[:10]
            ]
    except Exception:
        return []


async def _fetch_geo(postcode: str) -> dict:
    try:
        pc = postcode.replace(" ", "")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://api.postcodes.io/postcodes/{pc}")
            if resp.status_code == 200:
                return resp.json().get("result", {})
    except Exception:
        pass
    return {}


def _calc_transport_boost(stations: list) -> float:
    if not stations:
        return 0.0
    close = sum(1 for s in stations if s.get("distance_m", 9999) < 500)
    medium = sum(1 for s in stations if 500 <= s.get("distance_m", 9999) < 1500)
    return min(close * 3.0 + medium * 1.5, 12.0)


def _calc_regen_boost(regen: list, geo: dict) -> float:
    base = min(len(regen) * 2.0, 8.0)
    region = (geo.get("region") or "").lower()
    if "north" in region or "midlands" in region:
        base += 2.0
    return base


def _boost_label(boost: float) -> str:
    if boost >= 15:
        return "Very High — significant infrastructure uplift expected"
    if boost >= 8:
        return "High — notable infrastructure investment nearby"
    if boost >= 4:
        return "Moderate — some infrastructure signals present"
    return "Low — limited infrastructure catalysts detected"


def _investment_implication(boost: float) -> str:
    if boost >= 15:
        return "Buy now — infrastructure catalysts likely to drive above-average capital growth"
    if boost >= 8:
        return "Strong buy signal — regeneration/transport investment supports long-term growth"
    if boost >= 4:
        return "Positive — some infrastructure tailwinds, monitor for further announcements"
    return "Neutral — no significant infrastructure catalysts detected in immediate area"


def _approx_distance(lat1, lon1, lat2, lon2) -> int:
    import math
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return int(2 * R * math.asin(math.sqrt(a)))
