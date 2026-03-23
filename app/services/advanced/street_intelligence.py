"""
Street Level Intelligence Engine
Evaluates investment quality at street level rather than postcode.
Uses: HMLR, Police API, OSM, postcodes.io
"""
import httpx
import asyncio
import re
from app.core.logging import get_logger

log = get_logger(__name__)


async def analyse_street(address: str, lat: float, lng: float) -> dict:
    """Street-level investment scoring."""
    street = _extract_street(address)

    sales, crimes, density = await asyncio.gather(
        _fetch_street_sales(street, lat, lng),
        _fetch_street_crimes(lat, lng),
        _fetch_housing_density(lat, lng),
        return_exceptions=True,
    )

    sales_data = sales if not isinstance(sales, Exception) else []
    crime_data = crimes if not isinstance(crimes, Exception) else {}
    density_data = density if not isinstance(density, Exception) else {}

    transaction_score = _score_transactions(sales_data)
    crime_score = _score_crime(crime_data)
    density_score = _score_density(density_data)
    price_trend_score = _score_price_trend(sales_data)

    street_score = int(
        transaction_score * 0.30
        + crime_score * 0.35
        + density_score * 0.15
        + price_trend_score * 0.20
    )

    return {
        "address": address,
        "street": street,
        "street_investment_score": street_score,
        "street_grade": _grade(street_score),
        "metrics": {
            "transaction_activity": {
                "score": transaction_score,
                "recent_sales": len(sales_data),
                "avg_price_gbp": int(sum(s["price"] for s in sales_data) / len(sales_data)) if sales_data else 0,
            },
            "crime_environment": {
                "score": crime_score,
                "total_crimes_nearby": crime_data.get("total", 0),
                "dominant_category": crime_data.get("dominant_category", "Unknown"),
            },
            "housing_density": {
                "score": density_score,
                "building_count_nearby": density_data.get("building_count", 0),
                "character": density_data.get("character", "Unknown"),
            },
            "price_momentum": {
                "score": price_trend_score,
                "trend": _price_trend_label(sales_data),
            },
        },
        "street_character": _street_character(street_score, crime_score, transaction_score),
        "investor_verdict": _investor_verdict(street_score),
    }


def _extract_street(address: str) -> str:
    parts = address.split(",")
    if len(parts) >= 2:
        first = parts[0].strip()
        numbers = re.sub(r"^\d+\s*", "", first)
        return numbers if numbers else first
    return address


async def _fetch_street_sales(street: str, lat: float, lng: float) -> list:
    query = f"""
    PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
    PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
    SELECT ?amount ?date WHERE {{
      ?trans lrppi:pricePaid ?amount ;
             lrppi:transactionDate ?date ;
             lrppi:propertyAddress ?addr .
      ?addr lrcommon:street ?street .
      FILTER(CONTAINS(LCASE(?street), "{street.lower()[:20]}"))
    }}
    ORDER BY DESC(?date)
    LIMIT 15
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


async def _fetch_street_crimes(lat: float, lng: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://data.police.uk/api/crimes-street/all-crime",
                params={"lat": lat, "lng": lng},
            )
            if resp.status_code == 200:
                crimes = resp.json()
                if isinstance(crimes, list):
                    from collections import Counter
                    cats = Counter(c.get("category", "") for c in crimes)
                    dominant = cats.most_common(1)[0][0] if cats else "unknown"
                    return {"total": len(crimes), "dominant_category": dominant}
    except Exception:
        pass
    return {"total": 0, "dominant_category": "Unknown"}


async def _fetch_housing_density(lat: float, lng: float) -> dict:
    query = f"""
    [out:json][timeout:10];
    way["building"](around:200,{lat},{lng});
    out count;
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post("https://overpass-api.de/api/interpreter", data={"data": query})
            resp.raise_for_status()
            count = len(resp.json().get("elements", []))
            character = "dense urban" if count > 30 else "suburban" if count > 10 else "sparse"
            return {"building_count": count, "character": character}
    except Exception:
        return {"building_count": 0, "character": "Unknown"}


def _score_transactions(sales: list) -> int:
    count = len(sales)
    if count >= 10:
        return 85
    if count >= 5:
        return 70
    if count >= 2:
        return 50
    return 25


def _score_crime(crime_data: dict) -> int:
    total = crime_data.get("total", 0)
    if total == 0:
        return 75
    if total < 20:
        return 65
    if total < 50:
        return 50
    if total < 100:
        return 35
    return 15


def _score_density(density_data: dict) -> int:
    char = density_data.get("character", "")
    if "suburban" in char:
        return 75
    if "dense" in char:
        return 60
    if "sparse" in char:
        return 40
    return 55


def _score_price_trend(sales: list) -> int:
    if len(sales) < 3:
        return 50
    prices = [s["price"] for s in sales]
    recent = sum(prices[:3]) / 3
    older = sum(prices[-3:]) / 3
    if older == 0:
        return 50
    trend = (recent - older) / older
    if trend > 0.1:
        return 90
    if trend > 0.04:
        return 75
    if trend > 0:
        return 60
    return 35


def _price_trend_label(sales: list) -> str:
    if len(sales) < 3:
        return "Insufficient data"
    prices = [s["price"] for s in sales]
    recent = sum(prices[:3]) / 3
    older = sum(prices[-3:]) / 3
    if older == 0:
        return "Unknown"
    trend = (recent - older) / older
    if trend > 0.08:
        return "Strong growth"
    if trend > 0.03:
        return "Moderate growth"
    if trend > -0.03:
        return "Flat"
    return "Declining"


def _grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    if score >= 35:
        return "D"
    return "F"


def _street_character(score: int, crime: int, transaction: int) -> str:
    if score >= 75 and crime >= 60:
        return "Prime residential — low crime, active market"
    if score >= 60:
        return "Solid residential — good fundamentals"
    if score >= 45:
        return "Mixed — some positive signals but risks present"
    if crime < 40:
        return "High crime environment — investor caution advised"
    return "Challenged street — below average investment conditions"


def _investor_verdict(score: int) -> str:
    if score >= 75:
        return "Strong buy — above average street quality"
    if score >= 60:
        return "Buy — solid street fundamentals"
    if score >= 45:
        return "Neutral — investigate further before committing"
    return "Caution — below average street quality, price accordingly"
