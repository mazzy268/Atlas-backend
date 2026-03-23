"""
Transport / commute data
Sources:
  - TfL Unified API (London only, free, no key required)
  - Transport API (national, key required for premium)
  - OpenStreetMap Overpass API (station locations, free)
"""
import httpx
import math
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

TFL_BASE = "https://api.tfl.gov.uk"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def fetch(lat: float, lng: float, radius_m: int | None = None) -> dict:
    """Fetch transport infrastructure near lat/lng."""
    radius_m = radius_m or settings.transport_radius_m

    stations = await _fetch_stations_overpass(lat, lng, radius_m)
    bus_stops = await _fetch_bus_stops_overpass(lat, lng, radius_m)

    # Try TfL if we're in London
    tfl_data = {}
    if _is_london(lat, lng):
        tfl_data = await _fetch_tfl_stoppoints(lat, lng, radius_m)

    transport_score = _calculate_transport_score(stations, bus_stops, tfl_data)

    log.info("transport_fetched", lat=lat, lng=lng, stations=len(stations), bus_stops=len(bus_stops))
    return {
        "transport_score": transport_score,
        "nearest_stations": stations[:5],
        "bus_stop_count": len(bus_stops),
        "tfl_data": tfl_data,
        "is_london": _is_london(lat, lng),
        "radius_m": radius_m,
    }


async def _fetch_stations_overpass(lat: float, lng: float, radius: int) -> list[dict]:
    """Query OSM Overpass for train/tube/tram stations."""
    query = f"""
    [out:json][timeout:10];
    (
      node["railway"="station"](around:{radius},{lat},{lng});
      node["railway"="subway_entrance"](around:{radius},{lat},{lng});
      node["public_transport"="station"](around:{radius},{lat},{lng});
    );
    out body;
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])

        stations = []
        for e in elements:
            tags = e.get("tags", {})
            slat, slng = e.get("lat", lat), e.get("lon", lng)
            dist = _haversine(lat, lng, slat, slng)
            stations.append({
                "name": tags.get("name", "Unnamed station"),
                "type": tags.get("railway") or tags.get("public_transport", "station"),
                "distance_m": round(dist),
                "latitude": slat,
                "longitude": slng,
            })

        stations.sort(key=lambda s: s["distance_m"])
        return stations
    except Exception as e:
        log.warning("overpass_stations_failed", error=str(e))
        return []


async def _fetch_bus_stops_overpass(lat: float, lng: float, radius: int) -> list[dict]:
    """Query OSM for bus stops."""
    query = f"""
    [out:json][timeout:10];
    node["highway"="bus_stop"](around:{radius},{lat},{lng});
    out count;
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            return resp.json().get("elements", [])
    except Exception:
        return []


async def _fetch_tfl_stoppoints(lat: float, lng: float, radius: int) -> dict:
    """Fetch TfL stop points (London only)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{TFL_BASE}/StopPoint",
                params={"lat": lat, "lon": lng, "stopTypes": "NaptanMetroStation,NaptanRailStation", "radius": radius},
            )
            resp.raise_for_status()
            data = resp.json()
            stops = data.get("stopPoints", [])
            return {
                "nearby_tube_rail": len(stops),
                "stops": [{"name": s.get("commonName"), "distance": s.get("distance")} for s in stops[:5]],
            }
    except Exception:
        return {}


def _calculate_transport_score(stations: list, bus_stops: list, tfl: dict) -> int:
    """Score 0–10 based on public transport accessibility."""
    score = 0
    if stations:
        nearest = stations[0]["distance_m"]
        if nearest < 300:
            score += 5
        elif nearest < 600:
            score += 4
        elif nearest < 1000:
            score += 3
        elif nearest < 1500:
            score += 2
        else:
            score += 1
    bus_count = len(bus_stops)
    if bus_count > 10:
        score += 3
    elif bus_count > 5:
        score += 2
    elif bus_count > 0:
        score += 1
    if tfl.get("nearby_tube_rail", 0) > 0:
        score += 2
    return min(score, 10)


def _is_london(lat: float, lng: float) -> bool:
    return 51.28 <= lat <= 51.69 and -0.51 <= lng <= 0.33


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
