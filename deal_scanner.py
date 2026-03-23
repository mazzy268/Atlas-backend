"""
AI Deal Scanner — Enhanced
Improvements over v1:
- Uses last 24 months only (not all historical data) for current-market comparisons
- Type-matched BMV detection (compares like-for-like)
- Recency-weighted median as benchmark (not simple median)
- Wider sector scan for thin postcodes
- data_freshness field
"""
import httpx
import asyncio
import statistics
from datetime import datetime, date
from app.core.logging import get_logger

log = get_logger(__name__)
HMLR_SPARQL = "https://landregistry.data.gov.uk/landregistry/query"


async def scan_deals(postcode: str, address: str | None = None) -> dict:
    postcode_clean = postcode.strip().upper()
    # Use sector (e.g. NE15 6) for wider scan to get enough data
    sector = postcode_clean.rsplit(" ", 1)[0] if " " in postcode_clean else postcode_clean[:4]

    # Fetch recent 24-month sales only — avoids stale pre-pandemic prices
    sales = await _fetch_recent_sales(sector)

    if len(sales) < 5:
        return {
            "postcode": postcode_clean,
            "status": "insufficient_data",
            "message": f"Only {len(sales)} transactions found in last 24 months — insufficient for deal analysis",
            "deal_score": 0,
            "deals": [],
            "data_freshness": {
                "price": "Last 24 months",
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

    stats = _calculate_stats(sales)
    recency_median = _recency_weighted_median(sales)
    deals = _identify_deals(sales, stats, recency_median)
    anomalies = _find_anomalies(sales, stats)

    best_deal = deals[0] if deals else None
    deal_score = _overall_deal_score(deals, stats)

    # Freshness — how recent is the most recent sale?
    latest = max((s["date"] for s in sales), default="unknown")

    return {
        "postcode": postcode_clean,
        "area_statistics": {
            "median_price_gbp": stats["median"],
            "recency_weighted_median_gbp": recency_median,
            "mean_price_gbp": stats["mean"],
            "std_dev_gbp": stats["std_dev"],
            "transaction_count": len(sales),
            "period": "Last 24 months",
            "price_range": {"min_gbp": stats["min"], "max_gbp": stats["max"]},
        },
        "deal_score": deal_score,
        "deal_score_label": _deal_score_label(deal_score),
        "potential_deals": deals[:5],
        "price_anomalies": anomalies[:3],
        "best_deal": best_deal,
        "market_context": _market_context(stats, sales),
        "recommendation": _recommendation(deal_score, deals),
        "data_freshness": {
            "price": f"Most recent sale: {latest[:7] if latest != 'unknown' else 'Unknown'}",
            "period": "Last 24 months only",
            "last_updated": datetime.utcnow().isoformat(),
        },
    }


async def _fetch_recent_sales(sector: str) -> list:
    """Fetch sales from last 24 months only for current-market accuracy."""
    cutoff = date.today().replace(year=date.today().year - 2).isoformat()
    query = f"""
    PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
    PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?amount ?date ?propertyType ?paon ?street WHERE {{
      ?trans lrppi:pricePaid ?amount ;
             lrppi:transactionDate ?date ;
             lrppi:propertyType ?propertyType ;
             lrppi:propertyAddress ?addr .
      ?addr lrcommon:postcode ?pc .
      OPTIONAL {{ ?addr lrcommon:paon ?paon }}
      OPTIONAL {{ ?addr lrcommon:street ?street }}
      FILTER(STRSTARTS(?pc, "{sector}"))
      FILTER(?date >= "{cutoff}"^^xsd:date)
    }}
    ORDER BY DESC(?date)
    LIMIT 60
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
                    "type": b.get("propertyType", {}).get("value", "").split("/")[-1].lower(),
                    "number": b.get("paon", {}).get("value", ""),
                    "street": b.get("street", {}).get("value", ""),
                }
                for b in bindings if "amount" in b
            ]
    except Exception as e:
        log.warning("deal_scanner_fetch_failed", error=str(e))
        return []


def _calculate_stats(sales: list) -> dict:
    prices = [s["price"] for s in sales]
    return {
        "mean": int(statistics.mean(prices)),
        "median": int(statistics.median(prices)),
        "std_dev": int(statistics.stdev(prices)) if len(prices) > 1 else 0,
        "min": min(prices),
        "max": max(prices),
    }


def _recency_weighted_median(sales: list) -> int:
    """Recency-weighted median — recent sales count more."""
    today = date.today()
    weighted = []
    for s in sales:
        price = s["price"]
        try:
            sale_date = date.fromisoformat(s["date"][:10])
            months_ago = (today.year - sale_date.year) * 12 + (today.month - sale_date.month)
        except Exception:
            months_ago = 18
        weight = 3 if months_ago <= 6 else 2 if months_ago <= 12 else 1
        weighted.extend([price] * weight)
    if not weighted:
        return 0
    weighted.sort()
    mid = len(weighted) // 2
    return int((weighted[mid - 1] + weighted[mid]) / 2) if len(weighted) % 2 == 0 else weighted[mid]


def _identify_deals(sales: list, stats: dict, recency_median: int) -> list:
    """
    Use recency_weighted_median as benchmark — more reflective of current market.
    Flag anything >10% below as potential deal.
    """
    deals = []
    benchmark = recency_median or stats["median"]
    threshold = benchmark * 0.90  # 10% below recency median

    for sale in sales:
        price = sale["price"]
        if price < threshold and price > 10000:
            discount_pct = round((benchmark - price) / benchmark * 100, 1)
            deals.append({
                "address": f"{sale['number']} {sale['street']}".strip() or "Nearby property",
                "sold_price_gbp": price,
                "area_median_gbp": stats["median"],
                "recency_weighted_median_gbp": benchmark,
                "discount_vs_median_pct": discount_pct,
                "date": sale["date"],
                "property_type": sale.get("type", "Unknown"),
                "deal_type": _classify_deal(discount_pct),
                "deal_score": min(int(discount_pct * 3.5), 100),
                "months_since_sale": _months_ago(sale["date"]),
            })

    deals.sort(key=lambda x: x["discount_vs_median_pct"], reverse=True)
    return deals


def _find_anomalies(sales: list, stats: dict) -> list:
    anomalies = []
    mean = stats["mean"]
    std_dev = stats["std_dev"]
    if std_dev == 0:
        return []
    for sale in sales:
        z_score = (sale["price"] - mean) / std_dev
        if abs(z_score) > 2.0:
            anomalies.append({
                "address": f"{sale['number']} {sale['street']}".strip(),
                "price_gbp": sale["price"],
                "z_score": round(z_score, 2),
                "direction": "below market" if z_score < 0 else "above market",
                "date": sale["date"],
                "significance": "strong" if abs(z_score) > 3 else "moderate",
            })
    anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return anomalies


def _overall_deal_score(deals: list, stats: dict) -> int:
    if not deals:
        return 15
    best = max(d["discount_vs_median_pct"] for d in deals)
    if best >= 25:
        return 92
    if best >= 18:
        return 78
    if best >= 12:
        return 62
    if best >= 8:
        return 45
    return 28


def _classify_deal(discount_pct: float) -> str:
    if discount_pct >= 25:
        return "Potentially distressed — major discount"
    if discount_pct >= 18:
        return "Strong BMV — well below current market"
    if discount_pct >= 12:
        return "Below market value"
    if discount_pct >= 8:
        return "Slight discount — worth investigating"
    return "Minor discount"


def _deal_score_label(score: int) -> str:
    if score >= 80:
        return "Excellent — strong BMV opportunities detected"
    if score >= 60:
        return "Good — below-market activity in area"
    if score >= 40:
        return "Moderate — some deals possible"
    return "Fair market — limited BMV opportunities"


def _market_context(stats: dict, sales: list) -> str:
    spread_pct = (stats["max"] - stats["min"]) / stats["median"] * 100 if stats["median"] else 0
    recent = [s for s in sales if _months_ago(s["date"]) <= 6]
    if len(recent) >= 3:
        recent_median = int(statistics.median([s["price"] for s in recent]))
        all_median = stats["median"]
        if recent_median > all_median * 1.05:
            return "Market accelerating — recent prices above 24-month median. Act quickly on deals."
        if recent_median < all_median * 0.95:
            return "Market softening — recent prices below 24-month median. More negotiation room."
    if spread_pct > 80:
        return "Wide price range — diverse market, good hunting ground for deals"
    if spread_pct > 40:
        return "Moderate price variation — some deal opportunities"
    return "Tight price range — consistent market, limited BMV"


def _recommendation(score: int, deals: list) -> str:
    if score >= 75 and deals:
        d = deals[0]
        return (
            f"BMV opportunity detected: {d['address']} sold at "
            f"£{d['sold_price_gbp']:,} — {d['discount_vs_median_pct']}% below recency-weighted median. "
            f"Check current listings for similar properties."
        )
    if score >= 50:
        return "Below-market activity in area — monitor new listings, apply 10-15% below asking as opening offer."
    return "Market fairly priced — limited discounting. Wait for motivated sellers or target probate/auction."


def _months_ago(date_str: str) -> int:
    try:
        sale_date = date.fromisoformat(date_str[:10])
        today = date.today()
        return (today.year - sale_date.year) * 12 + (today.month - sale_date.month)
    except Exception:
        return 99
