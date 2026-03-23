"""
ONS Demographics data
Source: ONS API + Postcode.io for LSOA lookup
Free, no API key required.
"""
import httpx
from app.core.logging import get_logger

log = get_logger(__name__)

POSTCODES_IO = "https://api.postcodes.io/postcodes"
ONS_API = "https://api.beta.ons.gov.uk/v1"


async def fetch(postcode: str) -> dict:
    """Fetch ONS demographics for a postcode area."""
    postcode_clean = postcode.strip().upper().replace(" ", "")

    # Step 1: Resolve postcode to LSOA/MSOA via postcodes.io
    geo = await _resolve_postcode(postcode_clean)
    if not geo:
        return {"postcode": postcode, "error": "Postcode not found"}

    result = {
        "postcode": postcode,
        "area_name": geo.get("admin_district", ""),
        "lsoa_code": geo.get("lsoa"),
        "msoa_code": geo.get("msoa"),
        "ward": geo.get("admin_ward"),
        "local_authority": geo.get("admin_district"),
        "region": geo.get("region"),
        "country": geo.get("country"),
        "imd_decile": geo.get("imd"),          # Index of Multiple Deprivation
        "rural_urban": geo.get("rural_urban"),
    }

    # Step 2: Fetch ONS Census 2021 stats for MSOA
    msoa = geo.get("msoa")
    if msoa:
        census = await _fetch_census_stats(msoa)
        result.update(census)

    log.info("demographics_fetched", postcode=postcode, lsoa=geo.get("lsoa"))
    return result


async def _resolve_postcode(postcode: str) -> dict | None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{POSTCODES_IO}/{postcode}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {})


async def _fetch_census_stats(msoa_code: str) -> dict:
    """
    Fetch 2021 Census data for an MSOA from the ONS API.
    Falls back gracefully if unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Population density
            resp = await client.get(
                f"{ONS_API}/dataset/TS006/area/{msoa_code}/",
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                pop_data = resp.json()
                return _parse_ons_response(pop_data)
    except Exception as e:
        log.warning("ons_census_fetch_failed", error=str(e), msoa=msoa_code)

    return {}


def _parse_ons_response(data: dict) -> dict:
    """Parse ONS API response — structure varies by dataset."""
    result = {}
    observations = data.get("observations", [])
    for obs in observations:
        dim = obs.get("dimensions", {})
        val = obs.get("observation")
        if "population" in str(dim).lower():
            result["population_estimate"] = val
    return result
