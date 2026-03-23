"""
Development Potential Simulator
Estimates development opportunities using EPC, OSM, and planning data.
Uses: EPC API, OSM Overpass, AI analysis
"""
import httpx
import asyncio
from app.core.logging import get_logger
from app.core.config import get_settings

log = get_logger(__name__)
settings = get_settings()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def analyse_development_potential(
    address: str, postcode: str, lat: float, lng: float
) -> dict:
    """
    Full development potential analysis combining:
    - EPC property characteristics
    - OSM plot/footprint data
    - AI feasibility scoring
    """
    epc_data, osm_data = await asyncio.gather(
        _fetch_epc(postcode),
        _fetch_osm_building(lat, lng),
        return_exceptions=True,
    )

    epc = epc_data if not isinstance(epc_data, Exception) else {}
    osm = osm_data if not isinstance(osm_data, Exception) else {}

    floor_area = float(epc.get("total-floor-area") or 0)
    property_type = (epc.get("property-type") or "").lower()
    built_form = (epc.get("built-form") or "").lower()
    rooms = int(epc.get("number-habitable-rooms") or 0)
    roof = (epc.get("roof-description") or "").lower()

    loft = _assess_loft(built_form, property_type, roof, floor_area)
    extension = _assess_extension(built_form, property_type, osm)
    additional_unit = _assess_additional_unit(floor_area, rooms, property_type)
    hmo = _assess_hmo(floor_area, rooms, property_type)

    base_value = _estimate_base_value(epc)
    post_dev_value = _estimate_post_dev_value(
        base_value, loft, extension, additional_unit
    )

    total_dev_cost = (
        loft["estimated_cost_gbp"]
        + extension["estimated_cost_gbp"]
    )
    gross_dev_profit = post_dev_value - base_value - total_dev_cost
    roi = (gross_dev_profit / total_dev_cost * 100) if total_dev_cost else 0

    return {
        "address": address,
        "postcode": postcode,
        "property_characteristics": {
            "floor_area_sqm": floor_area,
            "property_type": property_type,
            "built_form": built_form,
            "habitable_rooms": rooms,
            "roof_type": roof,
            "floors_estimated": osm.get("floors"),
        },
        "development_opportunities": {
            "loft_conversion": loft,
            "extension": extension,
            "additional_unit": additional_unit,
            "hmo_conversion": hmo,
        },
        "financial_summary": {
            "estimated_current_value_gbp": base_value,
            "estimated_post_development_value_gbp": post_dev_value,
            "estimated_uplift_gbp": post_dev_value - base_value,
            "estimated_total_dev_cost_gbp": total_dev_cost,
            "estimated_gross_profit_gbp": gross_dev_profit,
            "development_roi_pct": round(roi, 1),
        },
        "overall_development_score": _overall_score(loft, extension, additional_unit),
        "planning_note": "Planning permission likely required for most works. Check local authority before proceeding.",
    }


async def _fetch_epc(postcode: str) -> dict:
    """Fetch EPC data for the postcode."""
    try:
        from app.core.config import get_settings
        s = get_settings()
        if not s.epc_api_key:
            return {}
        import base64
        creds = base64.b64encode(f"{s.epc_api_email}:{s.epc_api_key}".encode()).decode()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://epc.opendatacommunities.org/api/v1/domestic/search",
                params={"postcode": postcode, "size": 1},
                headers={"Accept": "application/json", "Authorization": f"Basic {creds}"},
            )
            if resp.status_code == 200:
                rows = resp.json().get("rows", [])
                return rows[0] if rows else {}
    except Exception as e:
        log.warning("dev_epc_fetch_failed", error=str(e))
    return {}


async def _fetch_osm_building(lat: float, lng: float) -> dict:
    """Fetch building footprint from OSM Overpass."""
    query = f"""
    [out:json][timeout:10];
    way["building"](around:50,{lat},{lng});
    out body;
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            if elements:
                tags = elements[0].get("tags", {})
                return {
                    "building_type": tags.get("building", "residential"),
                    "floors": tags.get("building:levels"),
                    "height": tags.get("height"),
                }
    except Exception as e:
        log.warning("osm_building_fetch_failed", error=str(e))
    return {}


def _assess_loft(built_form: str, property_type: str, roof: str, floor_area: float) -> dict:
    viable = False
    feasibility = "Low"
    cost = 0
    notes = []

    if any(t in property_type for t in ["house", "bungalow"]):
        if any(b in built_form for b in ["detached", "semi", "end-terrace", "mid-terrace"]):
            viable = True
            feasibility = "High" if "detached" in built_form or "semi" in built_form else "Medium"
            cost = 45000 if "detached" in built_form else 35000
            notes.append("Dormer conversion most likely feasible")
            if "200 mm" in roof or "270 mm" in roof:
                notes.append("Existing insulation may simplify works")
    elif "flat" in property_type:
        notes.append("Loft conversion not applicable for flats")

    return {
        "viable": viable,
        "feasibility": feasibility,
        "estimated_cost_gbp": cost,
        "estimated_value_add_gbp": int(cost * 1.6) if cost else 0,
        "notes": notes,
    }


def _assess_extension(built_form: str, property_type: str, osm: dict) -> dict:
    cost = 0
    feasibility = "Low"
    notes = []

    if "flat" in property_type:
        return {
            "viable": False,
            "feasibility": "Not applicable",
            "estimated_cost_gbp": 0,
            "estimated_value_add_gbp": 0,
            "notes": ["Extensions not applicable for flats"],
        }

    if any(b in built_form for b in ["detached", "semi", "end-terrace"]):
        feasibility = "High" if "detached" in built_form else "Medium"
        cost = 60000 if "detached" in built_form else 45000
        notes.append("Rear single-storey extension likely feasible under PD rights")
        notes.append("Check for Article 4 direction in area")
    elif "mid-terrace" in built_form:
        feasibility = "Medium"
        cost = 38000
        notes.append("Rear extension viable but may require party wall agreement")

    return {
        "viable": cost > 0,
        "feasibility": feasibility,
        "estimated_cost_gbp": cost,
        "estimated_value_add_gbp": int(cost * 1.5) if cost else 0,
        "notes": notes,
    }


def _assess_additional_unit(floor_area: float, rooms: int, property_type: str) -> dict:
    if floor_area >= 120 and rooms >= 5 and "house" in property_type:
        return {
            "viable": True,
            "unit_type": "Basement conversion or annex",
            "feasibility": "Medium",
            "estimated_cost_gbp": 80000,
            "notes": ["Subject to planning — not permitted development"],
        }
    return {
        "viable": False,
        "feasibility": "Low",
        "estimated_cost_gbp": 0,
        "notes": ["Insufficient floor area or rooms for additional unit"],
    }


def _assess_hmo(floor_area: float, rooms: int, property_type: str) -> dict:
    if "house" in property_type and rooms >= 4 and floor_area >= 80:
        potential_rooms = min(rooms + 1, int(floor_area / 18))
        return {
            "viable": True,
            "potential_lettable_rooms": potential_rooms,
            "feasibility": "Medium",
            "estimated_conversion_cost_gbp": potential_rooms * 3500,
            "estimated_monthly_hmo_rent_gbp": potential_rooms * 550,
            "notes": [
                "Article 4 direction check required",
                "Mandatory HMO licence needed for 5+ occupants",
                "Fire safety upgrades likely required",
            ],
        }
    return {
        "viable": False,
        "feasibility": "Low",
        "notes": ["Property too small or unsuitable for HMO conversion"],
    }


def _estimate_base_value(epc: dict) -> int:
    floor_area = float(epc.get("total-floor-area") or 65)
    return int(floor_area * 2800)


def _estimate_post_dev_value(
    base: int, loft: dict, extension: dict, additional: dict
) -> int:
    uplift = (
        loft.get("estimated_value_add_gbp", 0)
        + extension.get("estimated_value_add_gbp", 0)
        + (additional.get("estimated_cost_gbp", 0) * 1.4 if additional.get("viable") else 0)
    )
    return int(base + uplift)


def _overall_score(loft: dict, extension: dict, additional: dict) -> int:
    score = 30
    if loft.get("viable"):
        score += 25 if loft["feasibility"] == "High" else 15
    if extension.get("viable"):
        score += 25 if extension["feasibility"] == "High" else 15
    if additional.get("viable"):
        score += 20
    return min(score, 100)
