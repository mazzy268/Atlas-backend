"""
Liquidity Engine
Estimates how quickly a property can be sold.
Uses: HMLR transaction frequency, postcodes.io, ONS
"""
import httpx
from app.core.logging import get_logger

log = get_logger(__name__)

HMLR_SPARQL = "https://landregistry.data.gov.uk/landregistry/query"
POSTCODES_IO = "https://api.postcodes.io/postcodes"


async def calculate_liquidity(postcode: str) -> dict:
    """
    Returns a liquidity score 0-100 for a postcode.
    High score = easy to sell quickly.
    Low score = likely to sit on market.
    """
    postcode_clean = postcode.strip().upper()

    sales, geo = await _fetch_data(postcode_clean)

    transaction_score = _score_transactions(sales)
    location_score = _score_location(geo)
    price_band_score = _score_price_band(sales)

    weighted = (
        transaction_score * 0.45
        + location_score * 0.35
        + price_band_score * 0.20
    )
    liquidity_score = max(0, min(100, int(weighted)))

    return {
        "postcode": postcode_clean,
        "liquidity_score": liquidity_score,
        "liquidity_band": _liquidity_band(liquidity_score),
        "estimated_time_to_sell_weeks": _time_to_sell(liquidity_score),
        "transaction_frequency": len(sales),
        "avg_price_gbp": int(sum(s["price"] for s in sales) / len(sales)) if sales else 0,
        "price_band": _price_band(sales),
        "location_type": geo.get("rural_urban", "Unknown"),
        "region": geo.get("region", "Unknown"),
        "score_breakdown": {
            "transaction_frequency_score": transaction_score,
            "location_desirability_score": location_score,
            "price_band_score": price_band_score,
        },
        "recommendation": _liquidity_recommendation(liquidity_score),
    }


async def _fetch_data(postcode: str) -> tuple:
    """Fetch HMLR sales and postcodes.io geo data in parallel."""
    import asyncio

    async def fetch_sales():
        query = f"""
        PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
        PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
        SELECT ?amount ?date WHERE {{
          ?trans lrppi:pricePaid ?amount ;
                 lrppi:transactionDate ?date ;
                 lrppi:propertyAddress ?addr .
          ?addr lrcommon:postcode "{postcode}" .
        }}
        ORDER BY DESC(?date)
        LIMIT 30
        """
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
                    }
                    for b in bindings
                    if "amount" in b and "date" in b
                ]
        except Exception:
            return []

    async def fetch_geo():
        try:
            pc = postcode.replace(" ", "")
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{POSTCODES_IO}/{pc}")
                if resp.status_code == 200:
                    return resp.json().get("result", {})
        except Exception:
            pass
        return {}

    results = await asyncio.gather(fetch_sales(), fetch_geo(), return_exceptions=True)
    sales = results[0] if not isinstance(results[0], Exception) else []
    geo = results[1] if not isinstance(results[1], Exception) else {}
    return sales, geo


def _score_transactions(sales: list) -> float:
    count = len(sales)
    if count >= 20:
        return 90
    if count >= 10:
        return 75
    if count >= 5:
        return 55
    if count >= 2:
        return 35
    return 15


def _score_location(geo: dict) -> float:
    region = (geo.get("region") or "").lower()
    rural = (geo.get("rural_urban") or "").lower()
    imd = geo.get("imd") or 5

    score = 50
    high_demand = ["london", "south east", "east of england", "south west"]
    if any(r in region for r in high_demand):
        score += 25
    elif "midlands" in region or "north west" in region:
        score += 10

    if "urban" in rural:
        score += 15
    elif "rural" in rural:
        score -= 15

    if imd < 3:
        score -= 10
    elif imd > 7:
        score += 10

    return max(0, min(100, score))


def _score_price_band(sales: list) -> float:
    if not sales:
        return 50
    avg = sum(s["price"] for s in sales) / len(sales)
    if avg < 100000:
        return 45
    if avg < 250000:
        return 70
    if avg < 500000:
        return 80
    if avg < 1000000:
        return 65
    return 45


def _liquidity_band(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    if score >= 25:
        return "Low"
    return "Very Low"


def _time_to_sell(score: int) -> str:
    if score >= 75:
        return "4-8 weeks"
    if score >= 50:
        return "8-16 weeks"
    if score >= 25:
        return "16-26 weeks"
    return "26+ weeks"


def _price_band(sales: list) -> str:
    if not sales:
        return "Unknown"
    avg = sum(s["price"] for s in sales) / len(sales)
    if avg < 100000:
        return "Under £100k"
    if avg < 250000:
        return "£100k-£250k"
    if avg < 500000:
        return "£250k-£500k"
    if avg < 1000000:
        return "£500k-£1m"
    return "£1m+"


def _liquidity_recommendation(score: int) -> str:
    if score >= 75:
        return "Highly liquid market — easy exit strategy, low void risk"
    if score >= 50:
        return "Moderate liquidity — allow 3-4 months to sell if needed"
    if score >= 25:
        return "Low liquidity — plan for 6+ months exit, consider price reductions"
    return "Illiquid market — difficult to exit, high risk for short-term investors"
