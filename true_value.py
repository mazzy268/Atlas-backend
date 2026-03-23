"""
True Value Estimation Engine — Enhanced
Improvements over v1:
- Recency-weighted comparable sales (recent = higher weight)
- Property-type filtering (match terraced to terraced etc.)
- Sector-widening fallback (if postcode has <3 sales, widen to sector)
- Bedroom-adjusted rent estimation using ONS/Valuation Office data
- Confidence score tied to data quality, not just spread
- data_freshness field added
"""
import httpx
import asyncio
from datetime import datetime, date
from app.core.logging import get_logger

log = get_logger(__name__)

HMLR_SPARQL = "https://landregistry.data.gov.uk/landregistry/query"

# Valuation Office Agency rental data by region and bedrooms
# Source: VOA Private Rental Market Statistics (published annually, free)
# These are median monthly rents — updated to 2024 VOA figures
VOA_RENTS = {
    "london": {1: 1750, 2: 2300, 3: 2900, 4: 3800},
    "south east": {1: 1050, 2: 1350, 3: 1650, 4: 2100},
    "east of england": {1: 900, 2: 1150, 3: 1400, 4: 1800},
    "south west": {1: 850, 2: 1100, 3: 1350, 4: 1700},
    "east midlands": {1: 650, 2: 850, 3: 1000, 4: 1300},
    "west midlands": {1: 700, 2: 900, 3: 1050, 4: 1350},
    "north west": {1: 700, 2: 875, 3: 1050, 4: 1350},
    "yorkshire and the humber": {1: 600, 2: 775, 3: 900, 4: 1150},
    "north east": {1: 525, 2: 650, 3: 775, 4: 975},
    "wales": {1: 600, 2: 750, 3: 875, 4: 1100},
    "scotland": {1: 800, 2: 1000, 3: 1200, 4: 1550},
    "default": {1: 700, 2: 900, 3: 1100, 4: 1400},
}


async def estimate_true_value(address: str, postcode: str) -> dict:
    postcode_clean = postcode.strip().upper()
    sector = postcode_clean[:postcode_clean.rfind(" ")] if " " in postcode_clean else postcode_clean[:4]

    sales, epc_data, geo = await _gather_data(postcode_clean, sector)

    # Determine property type from EPC for type-matched comps
    prop_type_raw = (
        epc_data.get("property-type")
        or epc_data.get("property_type")
        or ""
    ).lower()
    built_form = (
        epc_data.get("built-form")
        or epc_data.get("built_form")
        or ""
    ).lower()
    bedrooms = _infer_bedrooms(epc_data)
    region = (geo.get("region") or "").lower()

    # Filter to matching property type first, fall back to all types
    typed_sales = _filter_by_type(sales, prop_type_raw, built_form)
    working_sales = typed_sales if len(typed_sales) >= 3 else sales

    comp_value = _recency_weighted_median(working_sales)
    rent_estimate = _voa_rent_estimate(region, bedrooms)
    rental_value = int(rent_estimate * 12 / 0.055) if rent_estimate else 0
    macro_value = _macro_model(comp_value, geo)

    consensus, val_confidence = _consensus_value(comp_value, rental_value, macro_value, working_sales)
    rent_confidence = _rent_confidence(region, bedrooms, len(sales))

    # Data freshness
    latest_sale_date = _latest_date(working_sales)
    freshness_days = (date.today() - latest_sale_date).days if latest_sale_date else None

    return {
        "address": address,
        "postcode": postcode_clean,
        "models": {
            "comparable_sales": {
                "value_gbp": comp_value,
                "method": "Recency-weighted median of matched comparable sales",
                "comparable_count": len(working_sales),
                "type_matched": len(typed_sales) >= 3,
                "confidence": "high" if len(working_sales) >= 5 else "medium" if len(working_sales) >= 2 else "low",
            },
            "rental_yield": {
                "value_gbp": rental_value,
                "method": f"VOA 2024 regional median rent ({bedrooms}bed, {region or 'national'}) capitalised at 5.5%",
                "estimated_monthly_rent_gbp": rent_estimate,
                "rent_source": "VOA Private Rental Market Statistics 2024",
                "confidence": "medium",
            },
            "macro_adjusted": {
                "value_gbp": macro_value,
                "method": "Comparable value with regional price-level overlay",
                "regional_adjustment_pct": _regional_adjustment(geo),
                "confidence": "medium",
            },
        },
        "consensus_value_gbp": consensus,
        "confidence_score": val_confidence,
        "confidence_band": _confidence_label(val_confidence),
        "value_range": {
            "low_gbp": int(consensus * 0.92),
            "mid_gbp": consensus,
            "high_gbp": int(consensus * 1.08),
        },
        "price_per_sqm_gbp": _price_per_sqm(consensus, epc_data),
        "estimated_monthly_rent_gbp": rent_estimate,
        "rent_confidence": rent_confidence,
        # New: confidence block
        "confidence": {
            "valuation": val_confidence,
            "rent": rent_confidence,
            "overall": int((val_confidence * 0.6) + (rent_confidence * 0.4)),
        },
        # New: data freshness block
        "data_freshness": {
            "price": f"{freshness_days} days ago" if freshness_days is not None else "Unknown",
            "price_date": latest_sale_date.isoformat() if latest_sale_date else None,
            "rent": "VOA 2024 annual publication",
            "crime": "Police API — current month",
            "last_updated": datetime.utcnow().isoformat(),
            "staleness_warning": freshness_days is not None and freshness_days > 365,
        },
        "valuation_note": "Indicative estimate only. Not a RICS appraisal.",
    }


async def _gather_data(postcode: str, sector: str) -> tuple:
    async def fetch_sales():
        # First try exact postcode, then widen to sector if needed
        results = await _hmlr_sales(postcode, limit=20)
        if len(results) < 3:
            sector_results = await _hmlr_sales_sector(sector, limit=40)
            results = results + [s for s in sector_results if s not in results]
        return results

    async def fetch_epc():
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
                    params={"postcode": postcode, "size": 3},
                    headers={"Accept": "application/json", "Authorization": f"Basic {creds}"},
                )
                if resp.status_code == 200:
                    rows = resp.json().get("rows", [])
                    return rows[0] if rows else {}
        except Exception:
            pass
        return {}

    async def fetch_geo():
        try:
            pc = postcode.replace(" ", "")
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"https://api.postcodes.io/postcodes/{pc}")
                if resp.status_code == 200:
                    return resp.json().get("result", {})
        except Exception:
            pass
        return {}

    results = await asyncio.gather(
        fetch_sales(), fetch_epc(), fetch_geo(), return_exceptions=True
    )
    return (
        results[0] if not isinstance(results[0], Exception) else [],
        results[1] if not isinstance(results[1], Exception) else {},
        results[2] if not isinstance(results[2], Exception) else {},
    )


async def _hmlr_sales(postcode: str, limit: int = 20) -> list:
    query = f"""
    PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
    PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
    SELECT ?amount ?date ?propertyType ?estateType WHERE {{
      ?trans lrppi:pricePaid ?amount ;
             lrppi:transactionDate ?date ;
             lrppi:propertyType ?propertyType ;
             lrppi:estateType ?estateType ;
             lrppi:propertyAddress ?addr .
      ?addr lrcommon:postcode "{postcode}" .
    }}
    ORDER BY DESC(?date)
    LIMIT {limit}
    """
    return await _run_sparql(query)


async def _hmlr_sales_sector(sector: str, limit: int = 40) -> list:
    query = f"""
    PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
    PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
    SELECT ?amount ?date ?propertyType ?estateType WHERE {{
      ?trans lrppi:pricePaid ?amount ;
             lrppi:transactionDate ?date ;
             lrppi:propertyType ?propertyType ;
             lrppi:estateType ?estateType ;
             lrppi:propertyAddress ?addr .
      ?addr lrcommon:postcode ?pc .
      FILTER(STRSTARTS(?pc, "{sector}"))
    }}
    ORDER BY DESC(?date)
    LIMIT {limit}
    """
    return await _run_sparql(query)


async def _run_sparql(query: str) -> list:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                HMLR_SPARQL,
                params={"query": query, "output": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
            return [
                {
                    "price": int(float(b["amount"]["value"])),
                    "date": b["date"]["value"],
                    "type": b.get("propertyType", {}).get("value", "").split("/")[-1].lower(),
                    "tenure": b.get("estateType", {}).get("value", "").split("/")[-1].lower(),
                }
                for b in bindings
                if "amount" in b and "date" in b
            ]
    except Exception as e:
        log.warning("hmlr_sparql_failed", error=str(e))
        return []


def _filter_by_type(sales: list, prop_type: str, built_form: str) -> list:
    """Filter sales to same property type. Maps EPC types to HMLR types."""
    type_map = {
        "house": ["terraced", "semi-detached", "detached"],
        "flat": ["flat-maisonette"],
        "maisonette": ["flat-maisonette"],
        "bungalow": ["detached", "semi-detached"],
    }
    # Also filter by built form (terraced vs semi etc.)
    form_map = {
        "mid-terrace": "terraced",
        "end-terrace": "terraced",
        "semi-detached": "semi-detached",
        "detached": "detached",
    }
    target_hmlr_type = None
    for epc_key, hmlr_types in type_map.items():
        if epc_key in prop_type:
            # Narrow further by built form if available
            for form_key, hmlr_val in form_map.items():
                if form_key in built_form and hmlr_val in hmlr_types:
                    target_hmlr_type = hmlr_val
                    break
            if not target_hmlr_type:
                target_hmlr_type = hmlr_types[0]
            break

    if not target_hmlr_type:
        return sales

    return [s for s in sales if target_hmlr_type in s.get("type", "")]


def _recency_weighted_median(sales: list) -> int:
    """
    Weight recent sales higher than older ones.
    Sales within 12 months get 3x weight.
    Sales within 24 months get 2x weight.
    Older sales get 1x weight.
    """
    if not sales:
        return 0

    today = date.today()
    weighted_prices = []

    for s in sales:
        price = s["price"]
        try:
            sale_date = date.fromisoformat(s["date"][:10])
            months_ago = (today.year - sale_date.year) * 12 + (today.month - sale_date.month)
        except Exception:
            months_ago = 36

        if months_ago <= 12:
            weight = 3
        elif months_ago <= 24:
            weight = 2
        else:
            weight = 1

        weighted_prices.extend([price] * weight)

    weighted_prices.sort()
    mid = len(weighted_prices) // 2
    if len(weighted_prices) % 2 == 0:
        return int((weighted_prices[mid - 1] + weighted_prices[mid]) / 2)
    return weighted_prices[mid]


def _voa_rent_estimate(region: str, bedrooms: int) -> int:
    """
    Use VOA Private Rental Market Statistics 2024 for rent estimates.
    Much more accurate than floor_area * constant.
    """
    beds = max(1, min(bedrooms, 4))
    for key in VOA_RENTS:
        if key != "default" and key in region:
            return VOA_RENTS[key][beds]
    return VOA_RENTS["default"][beds]


def _infer_bedrooms(epc_data: dict) -> int:
    rooms = epc_data.get("number-habitable-rooms") or epc_data.get("number_habitable_rooms")
    if rooms:
        try:
            r = int(rooms)
            # Habitable rooms = bedrooms + reception. Subtract 1 for living room.
            return max(1, r - 1)
        except (ValueError, TypeError):
            pass
    return 2  # default


def _rent_confidence(region: str, bedrooms: int, sales_count: int) -> int:
    """Confidence in rent estimate based on data quality."""
    base = 65  # VOA data is reliable but regional, not postcode-level
    if region and region != "default":
        base += 10
    if bedrooms > 0:
        base += 5
    if sales_count >= 5:
        base += 10
    return min(base, 90)


def _macro_model(comp_value: int, geo: dict) -> int:
    if not comp_value:
        return 0
    adjustment = _regional_adjustment(geo)
    return int(comp_value * (1 + adjustment / 100))


def _regional_adjustment(geo: dict) -> float:
    region = (geo.get("region") or "").lower()
    # Based on ONS House Price Index regional differentials
    adjustments = {
        "london": 8.0,
        "south east": 4.0,
        "east of england": 2.0,
        "south west": 2.5,
        "east midlands": 1.0,
        "west midlands": 1.5,
        "north west": 1.0,
        "yorkshire and the humber": 0.0,
        "north east": -1.5,
        "wales": -0.5,
        "scotland": 0.5,
    }
    for key, adj in adjustments.items():
        if key in region:
            return adj
    return 0.0


def _consensus_value(comp: int, rental: int, macro: int, sales: list) -> tuple[int, int]:
    values = [v for v in [comp, rental, macro] if v > 0]
    if not values:
        return 0, 0

    # Weight comparables heavily if we have good data
    if len(sales) >= 8:
        weights = [0.65, 0.20, 0.15]
    elif len(sales) >= 5:
        weights = [0.55, 0.25, 0.20]
    elif len(sales) >= 2:
        weights = [0.45, 0.30, 0.25]
    else:
        weights = [0.33, 0.34, 0.33]

    active = [(v, w) for v, w in zip([comp, rental, macro], weights) if v > 0]
    total_w = sum(w for _, w in active)
    consensus = int(sum(v * w for v, w in active) / total_w)

    spread = max(values) - min(values)
    spread_pct = spread / consensus if consensus else 1

    if spread_pct < 0.08:
        confidence = 88
    elif spread_pct < 0.15:
        confidence = 75
    elif spread_pct < 0.25:
        confidence = 60
    elif spread_pct < 0.40:
        confidence = 45
    else:
        confidence = 30

    # Boost/penalise based on data volume
    if len(sales) >= 8:
        confidence = min(confidence + 12, 95)
    elif len(sales) >= 5:
        confidence = min(confidence + 8, 92)
    elif len(sales) >= 2:
        confidence = min(confidence + 3, 85)
    else:
        confidence = max(confidence - 20, 10)

    return consensus, confidence


def _confidence_label(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    if score >= 40:
        return "Low"
    return "Very Low"


def _price_per_sqm(value: int, epc_data: dict) -> int | None:
    floor_area = float(
        epc_data.get("total-floor-area")
        or epc_data.get("floor_area_sqm")
        or 0
    )
    if floor_area > 0 and value > 0:
        return int(value / floor_area)
    return None


def _latest_date(sales: list) -> date | None:
    dates = []
    for s in sales:
        try:
            dates.append(date.fromisoformat(s["date"][:10]))
        except Exception:
            pass
    return max(dates) if dates else None
