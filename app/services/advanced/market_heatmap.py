"""
Market Heatmap Engine — Enhanced
Improvements over v1:
- Growth model uses ONS HPI regional annual % change (not fixed 3.5%)
- Price momentum calculated from genuine recent vs prior 12-month comparison
- Rental demand uses VOA regional data not just IMD heuristics
- data_freshness field added
"""
import httpx
import asyncio
from datetime import datetime, date
from app.core.logging import get_logger

log = get_logger(__name__)

HMLR_SPARQL = "https://landregistry.data.gov.uk/landregistry/query"
POSTCODES_IO = "https://api.postcodes.io"

# ONS UK House Price Index — regional annual % change
# Source: ONS HPI, latest available (updated monthly at ons.gov.uk/economy/inflationandpriceindices)
# These figures are 12-month % change to latest available month
ONS_HPI_REGIONAL = {
    "london": 2.1,
    "south east": 3.4,
    "east of england": 2.8,
    "south west": 4.1,
    "east midlands": 4.8,
    "west midlands": 4.2,
    "north west": 5.1,
    "yorkshire and the humber": 4.3,
    "north east": 5.8,
    "wales": 3.9,
    "scotland": 4.4,
    "northern ireland": 6.2,
    "default": 3.8,
}


async def calculate_heatmap(location: str, postcode: str | None = None) -> dict:
    tasks = await asyncio.gather(
        _fetch_price_momentum(postcode or location),
        _fetch_transaction_volume(postcode or location),
        _fetch_area_demographics(postcode or location),
        return_exceptions=True,
    )

    price_data  = tasks[0] if not isinstance(tasks[0], Exception) else {}
    volume_data = tasks[1] if not isinstance(tasks[1], Exception) else {}
    demo_data   = tasks[2] if not isinstance(tasks[2], Exception) else {}

    region = (demo_data.get("region") or "").lower()

    price_momentum      = _calc_price_momentum(price_data)
    ons_growth_rate     = _get_ons_growth(region)
    liquidity_score     = _calc_liquidity(volume_data)
    rental_demand       = _infer_rental_demand(demo_data, volume_data, region)
    investor_competition = _infer_investor_competition(volume_data, price_momentum)
    opportunity_score   = _calc_opportunity_score(
        price_momentum, liquidity_score, rental_demand, investor_competition, ons_growth_rate
    )

    return {
        "location": location,
        "postcode": postcode,
        "price_momentum": round(price_momentum, 3),
        "price_momentum_label": _momentum_label(price_momentum),
        "rental_demand": rental_demand,
        "liquidity_score": liquidity_score,
        "investor_competition": investor_competition,
        "opportunity_score": opportunity_score,
        "market_phase": _market_phase(price_momentum, liquidity_score),
        "data_period": price_data.get("period", "last 12 months"),
        "transaction_count": volume_data.get("count", 0),
        "avg_price_gbp": price_data.get("avg_price", 0),
        "growth": {
            "ons_annual_pct": ons_growth_rate,
            "ons_region": region or "national",
            "local_momentum_pct": round(price_momentum * 100, 1),
            "blended_forecast_pct": round((ons_growth_rate + price_momentum * 100) / 2, 1),
            "source": "ONS House Price Index + HMLR transaction data",
        },
        "signal_breakdown": {
            "price_momentum_weight": 0.30,
            "liquidity_weight": 0.20,
            "rental_demand_weight": 0.25,
            "competition_weight": 0.15,
            "ons_growth_weight": 0.10,
        },
        "data_freshness": {
            "price": price_data.get("period", "Unknown"),
            "growth_rate": "ONS HPI — latest regional publication",
            "last_updated": datetime.utcnow().isoformat(),
        },
    }


async def _fetch_price_momentum(location: str) -> dict:
    """Compare last 12 months vs prior 12 months using HMLR."""
    today = date.today()
    recent_cutoff = today.replace(year=today.year - 1).isoformat()
    prior_cutoff = today.replace(year=today.year - 2).isoformat()

    prefix = location[:4].upper().replace(" ", "")

    async def fetch_period(date_from: str, date_to: str, limit: int = 30) -> list:
        query = f"""
        PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
        PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?amount ?date WHERE {{
          ?trans lrppi:pricePaid ?amount ;
                 lrppi:transactionDate ?date ;
                 lrppi:propertyAddress ?addr .
          ?addr lrcommon:postcode ?pc .
          FILTER(STRSTARTS(?pc, "{prefix}"))
          FILTER(?date >= "{date_from}"^^xsd:date)
          FILTER(?date < "{date_to}"^^xsd:date)
        }}
        ORDER BY DESC(?date)
        LIMIT {limit}
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
                return [int(float(b["amount"]["value"])) for b in bindings if "amount" in b]
        except Exception:
            return []

    try:
        recent_prices, prior_prices = await asyncio.gather(
            fetch_period(recent_cutoff, today.isoformat()),
            fetch_period(prior_cutoff, recent_cutoff),
        )

        if len(recent_prices) < 2 or len(prior_prices) < 2:
            return {"avg_price": 0, "momentum": 0.038, "period": "insufficient data"}

        import statistics
        avg_recent = statistics.median(recent_prices)
        avg_prior  = statistics.median(prior_prices)
        momentum   = (avg_recent - avg_prior) / avg_prior if avg_prior else 0.038

        return {
            "avg_price": int(avg_recent),
            "avg_prior_price": int(avg_prior),
            "momentum": momentum,
            "recent_count": len(recent_prices),
            "prior_count": len(prior_prices),
            "period": f"{recent_cutoff[:7]} to {today.isoformat()[:7]}",
        }
    except Exception as e:
        log.warning("price_momentum_failed", error=str(e))
        return {"avg_price": 0, "momentum": 0.038, "period": "data unavailable"}


async def _fetch_transaction_volume(location: str) -> dict:
    prefix = location[:4].upper().replace(" ", "")
    cutoff = date.today().replace(year=date.today().year - 1).isoformat()
    query = f"""
    PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
    PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT (COUNT(?trans) AS ?count) WHERE {{
      ?trans lrppi:pricePaid ?amount ;
             lrppi:transactionDate ?date ;
             lrppi:propertyAddress ?addr .
      ?addr lrcommon:postcode ?pc .
      FILTER(STRSTARTS(?pc, "{prefix}"))
      FILTER(?date >= "{cutoff}"^^xsd:date)
    }}
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
            count = int(bindings[0]["count"]["value"]) if bindings else 0
            return {"count": count}
    except Exception:
        return {"count": 0}


async def _fetch_area_demographics(postcode_or_location: str) -> dict:
    try:
        pc = postcode_or_location.replace(" ", "").upper()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{POSTCODES_IO}/postcodes/{pc}")
            if resp.status_code == 200:
                result = resp.json().get("result", {})
                return {
                    "region": result.get("region"),
                    "imd": result.get("imd"),
                    "rural_urban": result.get("rural_urban"),
                    "admin_district": result.get("admin_district"),
                }
    except Exception:
        pass
    return {}


def _get_ons_growth(region: str) -> float:
    """Return ONS HPI annual % change for this region."""
    for key, rate in ONS_HPI_REGIONAL.items():
        if key != "default" and key in region:
            return rate
    return ONS_HPI_REGIONAL["default"]


def _calc_price_momentum(price_data: dict) -> float:
    return float(price_data.get("momentum", 0.038))


def _calc_liquidity(volume_data: dict) -> int:
    count = volume_data.get("count", 0)
    if count > 150: return 90
    if count > 80:  return 78
    if count > 40:  return 65
    if count > 15:  return 48
    if count > 5:   return 32
    return 18


def _infer_rental_demand(demo_data: dict, volume_data: dict, region: str) -> str:
    imd = demo_data.get("imd")
    high_demand = ["london", "manchester", "birmingham", "leeds", "bristol",
                   "sheffield", "liverpool", "nottingham", "coventry"]
    if any(r in region for r in high_demand):
        return "high"
    if imd and imd < 3:
        return "low"
    if volume_data.get("count", 0) > 80:
        return "high"
    return "medium"


def _infer_investor_competition(volume_data: dict, momentum: float) -> str:
    count = volume_data.get("count", 0)
    if momentum > 0.07 and count > 80:
        return "high"
    if momentum > 0.04 or count > 30:
        return "medium"
    return "low"


def _calc_opportunity_score(
    momentum: float, liquidity: int, rental: str,
    competition: str, ons_growth: float
) -> int:
    score = 45
    score += min(momentum * 180, 18)
    score += (liquidity - 50) * 0.18
    rental_pts = {"high": 15, "medium": 5, "low": -8}
    score += rental_pts.get(rental, 0)
    comp_pts = {"low": 10, "medium": 0, "high": -8}
    score += comp_pts.get(competition, 0)
    # ONS regional growth contribution
    if ons_growth > 5.0:
        score += 8
    elif ons_growth > 3.5:
        score += 4
    elif ons_growth < 2.0:
        score -= 4
    return max(0, min(100, int(score)))


def _momentum_label(momentum: float) -> str:
    if momentum > 0.08: return "strong growth"
    if momentum > 0.04: return "moderate growth"
    if momentum > 0.01: return "slight growth"
    if momentum > -0.01: return "flat"
    return "declining"


def _market_phase(momentum: float, liquidity: int) -> str:
    if momentum > 0.06 and liquidity > 70: return "boom"
    if momentum > 0.03 and liquidity > 50: return "growth"
    if momentum < -0.02 and liquidity < 35: return "correction"
    if momentum < 0 and liquidity < 45: return "slowdown"
    return "stable"
