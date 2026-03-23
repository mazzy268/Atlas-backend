"""
EPC (Energy Performance Certificate) data
Source: DLUHC Open Data Communities
API key required — free registration at https://epc.opendatacommunities.org/
"""
import httpx
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

EPC_BASE = "https://epc.opendatacommunities.org/api/v1"


async def fetch(postcode: str, address: str | None = None) -> dict:
    """Fetch EPC rating(s) for a postcode, optionally filtered by address."""
    if not settings.epc_api_key:
        log.warning("epc_api_key_missing — returning empty EPC data")
        return {"ratings": [], "note": "EPC API key not configured"}

    postcode_clean = postcode.strip().upper().replace(" ", "%20")
    url = f"{EPC_BASE}/domestic/search"
    params = {"postcode": postcode.strip().upper(), "size": 10}

    headers = {
        "Accept": "application/json",
        "Authorization": f"Basic {_basic_auth()}",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code == 401:
            return {"ratings": [], "error": "Invalid EPC credentials"}
        resp.raise_for_status()
        data = resp.json()

    rows = data.get("rows", [])

    # If address provided, try to find best match
    if address and rows:
        rows = _best_match(rows, address) or rows

    ratings = [_parse_row(r) for r in rows[:5]]
    log.info("epc_fetched", postcode=postcode, count=len(ratings))
    return {"postcode": postcode, "ratings": ratings}


def _basic_auth() -> str:
    import base64
    creds = f"{settings.epc_api_email}:{settings.epc_api_key}"
    return base64.b64encode(creds.encode()).decode()


def _parse_row(r: dict) -> dict:
    return {
        "lmk_key": r.get("lmk-key"),
        "address": r.get("address"),
        "current_energy_rating": r.get("current-energy-rating"),
        "current_energy_efficiency": r.get("current-energy-efficiency"),
        "potential_energy_rating": r.get("potential-energy-rating"),
        "potential_energy_efficiency": r.get("potential-energy-efficiency"),
        "property_type": r.get("property-type"),
        "built_form": r.get("built-form"),
        "floor_area_sqm": r.get("total-floor-area"),
        "lodgement_date": r.get("lodgement-date"),
        "walls_description": r.get("walls-description"),
        "roof_description": r.get("roof-description"),
        "heating_description": r.get("main-heat-description"),
        "windows_description": r.get("windows-description"),
        "number_habitable_rooms": r.get("number-habitable-rooms"),
        "floor_description": r.get("floor-description"),
    }


def _best_match(rows: list, address: str) -> list:
    address_lower = address.lower()
    for row in rows:
        if address_lower[:8] in row.get("address", "").lower():
            return [row]
    return []
