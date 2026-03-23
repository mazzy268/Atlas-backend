"""
Market Risk Engine
Comprehensive risk scoring across flood, crime, economic, overvaluation, and liquidity dimensions.
"""
import httpx
import asyncio
from app.core.logging import get_logger

log = get_logger(__name__)


async def analyse_risk(
    address: str, postcode: str, lat: float, lng: float
) -> dict:
    """Full risk profile for a property location."""
    flood, crime, geo, sales = await asyncio.gather(
        _fetch_flood(lat, lng),
        _fetch_crime(lat, lng),
        _fetch_geo(postcode),
        _fetch_sales(postcode),
        return_exceptions=True,
    )

    flood_data = flood if not isinstance(flood, Exception) else {}
    crime_data = crime if not isinstance(crime, Exception) else {}
    geo_data = geo if not isinstance(geo, Exception) else {}
    sales_data = sales if not isinstance(sales, Exception) else []

    flood_risk = _score_flood_risk(flood_data)
    crime_risk = _score_crime_risk(crime_data)
    economic_risk = _score_economic_risk(geo_data)
    overval_risk = _score_overvaluation_risk(sales_data, geo_data)
    liquidity_risk = _score_liquidity_risk(sales_data)

    overall = int(
        flood_risk["score"] * 0.20
        + crime_risk["score"] * 0.25
        + economic_risk["score"] * 0.20
        + overval_risk["score"] * 0.20
        + liquidity_risk["score"] * 0.15
    )

    return {
        "address": address,
        "postcode": postcode,
        "investment_risk_score": overall,
        "risk_band": _risk_band(overall),
        "risk_breakdown": {
            "flood_risk": flood_risk,
            "crime_risk": crime_risk,
            "economic_vulnerability": economic_risk,
            "overvaluation_risk": overval_risk,
            "liquidity_risk": liquidity_risk,
        },
        "red_flags": _identify_red_flags(flood_risk, crime_risk, economic_risk, overval_risk),
        "risk_mitigation": _mitigation_advice(flood_risk, crime_risk, economic_risk),
        "suitable_for": _suitable_investor_profile(overall),
    }


async def _fetch_flood(lat: float, lng: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://environment.data.gov.uk/flood-monitoring/id/floods",
                params={"lat": lat, "long": lng, "dist": 2},
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                return {"warning_count": len(items), "warnings": items[:3]}
    except Exception:
        pass
    return {"warning_count": 0, "warnings": []}


async def _fetch_crime(lat: float, lng: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://data.police.uk/api/crimes-street/all-crime",
                params={"lat": lat, "lng": lng},
            )
            if resp.status_code == 200:
                crimes = resp.json()
                if isinstance(crimes, list):
                    return {"total": len(crimes)}
    except Exception:
        pass
    return {"total": 0}


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


async def _fetch_sales(postcode: str) -> list:
    query = f"""
    PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
    PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
    SELECT ?amount ?date WHERE {{
      ?trans lrppi:pricePaid ?amount ;
             lrppi:transactionDate ?date ;
             lrppi:propertyAddress ?addr .
      ?addr lrcommon:postcode "{postcode.strip().upper()}" .
    }}
    ORDER BY DESC(?date)
    LIMIT 20
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://landregistry.data.gov.uk/landregistry/query",
                params={"query": query, "output": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
            return [
                {"price": int(float(b["amount"]["value"])), "date": b["date"]["value"]}
                for b in bindings if "amount" in b
            ]
    except Exception:
        return []


def _score_flood_risk(flood_data: dict) -> dict:
    warnings = flood_data.get("warning_count", 0)
    if warnings > 2:
        score = 80
        level = "High"
    elif warnings > 0:
        score = 50
        level = "Medium"
    else:
        score = 15
        level = "Low"
    return {"score": score, "level": level, "active_warnings": warnings}


def _score_crime_risk(crime_data: dict) -> dict:
    total = crime_data.get("total", 0)
    if total > 200:
        return {"score": 85, "level": "Very High", "crime_count": total}
    if total > 100:
        return {"score": 65, "level": "High", "crime_count": total}
    if total > 50:
        return {"score": 45, "level": "Medium", "crime_count": total}
    return {"score": 20, "level": "Low", "crime_count": total}


def _score_economic_risk(geo_data: dict) -> dict:
    imd = geo_data.get("imd") or 5
    region = (geo_data.get("region") or "").lower()
    score = 50
    if imd <= 2:
        score = 75
    elif imd <= 4:
        score = 55
    elif imd >= 8:
        score = 25
    vulnerable_regions = ["north east", "wales", "yorkshire"]
    if any(r in region for r in vulnerable_regions):
        score += 10
    resilient_regions = ["london", "south east"]
    if any(r in region for r in resilient_regions):
        score -= 15
    score = max(0, min(100, score))
    return {"score": score, "imd_decile": imd, "region": region or "Unknown"}


def _score_overvaluation_risk(sales: list, geo_data: dict) -> dict:
    if len(sales) < 3:
        return {"score": 40, "level": "Unknown — insufficient data"}
    prices = [s["price"] for s in sales]
    recent = sum(prices[:3]) / 3
    older = sum(prices[-3:]) / 3
    if older == 0:
        return {"score": 40, "level": "Unknown"}
    growth = (recent - older) / older
    if growth > 0.25:
        return {"score": 70, "level": "High — rapid price growth detected", "growth_pct": round(growth * 100, 1)}
    if growth > 0.12:
        return {"score": 45, "level": "Medium", "growth_pct": round(growth * 100, 1)}
    return {"score": 20, "level": "Low", "growth_pct": round(growth * 100, 1)}


def _score_liquidity_risk(sales: list) -> dict:
    count = len(sales)
    if count < 3:
        return {"score": 70, "level": "High — very few transactions"}
    if count < 8:
        return {"score": 45, "level": "Medium"}
    return {"score": 20, "level": "Low — active market"}


def _risk_band(score: int) -> str:
    if score >= 70:
        return "High Risk"
    if score >= 50:
        return "Medium-High Risk"
    if score >= 35:
        return "Medium Risk"
    if score >= 20:
        return "Low-Medium Risk"
    return "Low Risk"


def _identify_red_flags(flood, crime, economic, overval) -> list:
    flags = []
    if flood["score"] >= 60:
        flags.append("Active flood warnings in area")
    if crime["score"] >= 65:
        flags.append("High crime rate — above national average")
    if economic["score"] >= 65:
        flags.append("Area in high deprivation decile")
    if overval.get("score", 0) >= 65:
        flags.append("Rapid recent price growth — overvaluation risk")
    return flags or ["No major red flags identified"]


def _mitigation_advice(flood, crime, economic) -> list:
    advice = []
    if flood["score"] >= 50:
        advice.append("Obtain specialist flood insurance before purchase")
    if crime["score"] >= 55:
        advice.append("Factor higher insurance premiums and security costs into yield calculations")
    if economic["score"] >= 60:
        advice.append("Focus on net yield rather than capital growth in this area")
    return advice or ["Standard due diligence applies"]


def _suitable_investor_profile(score: int) -> str:
    if score < 25:
        return "Suitable for all investor types including first-time investors"
    if score < 45:
        return "Suitable for experienced investors comfortable with moderate risk"
    if score < 65:
        return "Suitable for experienced investors only — significant risk factors present"
    return "High risk — institutional or specialist investors only"
