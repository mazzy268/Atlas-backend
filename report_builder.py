"""
Report builder — the core orchestrator.
1. Fetches all 8 data sources concurrently
2. Runs all 12 AI analysis features
3. Assembles and returns the full PropertyReport

Adding a new data source: add it to gather_all_data().
Adding a new AI feature: add a prompt to prompts.py, call it in build_report().
"""
import asyncio
from datetime import datetime, timedelta
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services import (
    geocoder,
)
from app.services.data_fetchers import (
    land_registry,
    epc,
    crime,
    demographics,
    flood_risk,
    planning,
    schools,
    transport,
)
from app.services.ai_analysis import prompts
from app.services.ai_analysis.openai_client import complete_json, complete_text

log = get_logger(__name__)
settings = get_settings()


async def build_report(address: str) -> dict:
    """
    Full pipeline: geocode → fetch data → AI analysis → return report dict.
    """
    log.info("report_build_start", address=address)

    # ── Step 1: Geocode ────────────────────────────────────────────────────────
    coords = await geocoder.geocode_address(address)
    lat = coords["latitude"]
    lng = coords["longitude"]
    postcode = coords.get("postcode") or _extract_postcode_from_address(address)

    log.info("geocoded", lat=lat, lng=lng, postcode=postcode)

    # ── Step 2: Fetch all data sources concurrently ────────────────────────────
    raw = await gather_all_data(lat, lng, postcode, address)

    # ── Step 3: Run all AI features concurrently ───────────────────────────────
    ai_results = await run_all_ai_features(address, raw)

    # ── Step 4: Assemble report ────────────────────────────────────────────────
    report = assemble_report(address, coords, raw, ai_results)

    log.info("report_build_complete", address=address, score=report.get("investment_score", {}).get("score"))
    return report


async def gather_all_data(lat: float, lng: float, postcode: str | None, address: str) -> dict:
    """
    Fan out to all 8 data sources simultaneously.
    Each returns a dict; failures return {"error": "..."} so the report still generates.
    """
    tasks = {
        "land_registry": land_registry.fetch(postcode or "", limit=20) if postcode else asyncio.coroutine(lambda: {"sales": [], "note": "No postcode"})(),
        "epc": epc.fetch(postcode or "", address) if postcode else asyncio.coroutine(lambda: {"ratings": []})(),
        "crime": crime.fetch(lat, lng),
        "demographics": demographics.fetch(postcode or "") if postcode else asyncio.coroutine(lambda: {})(),
        "flood": flood_risk.fetch(lat, lng),
        "planning": planning.fetch(lat, lng),
        "schools": schools.fetch(lat, lng),
        "transport": transport.fetch(lat, lng),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    raw = {}
    for key, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            log.warning("data_fetch_failed", source=key, error=str(result))
            raw[key] = {"error": str(result)}
        else:
            raw[key] = result

    log.info("data_gathered", sources=list(raw.keys()))
    return raw


async def run_all_ai_features(address: str, raw: dict) -> dict:
    """
    Run all 11 AI feature prompts concurrently (summary runs after to use other results).
    """
    # Prepare condensed data bundles for each prompt
    epc_data = raw.get("epc", {})
    epc_rating = _first_epc_rating(epc_data)
    sales = raw.get("land_registry", {}).get("sales", [])
    crime_data = raw.get("crime", {})
    transport_data = raw.get("transport", {})
    demographics_data = raw.get("demographics", {})
    flood_data = raw.get("flood", {})
    planning_data = raw.get("planning", {})
    schools_data = raw.get("schools", {})

    feature_tasks = {
        "investment_score": complete_json(
            prompts.investment_score_prompt({
                "address": address,
                "sales": _summarise_sales(sales),
                "epc": epc_rating,
                "crime": _summarise_crime(crime_data),
                "flood": flood_data.get("risk_level", "Unknown"),
                "schools": _summarise_schools(schools_data),
                "transport": transport_data.get("transport_score", "Unknown"),
                "planning": _summarise_planning(planning_data),
                "demographics": _summarise_demographics(demographics_data),
            }),
            "investment_score",
        ),
        "strategy_detector": complete_json(
            prompts.strategy_detector_prompt({
                "address": address,
                "sales": _summarise_sales(sales),
                "epc": epc_rating,
                "crime_summary": _summarise_crime(crime_data),
                "transport_summary": _summarise_transport(transport_data),
                "demographics": _summarise_demographics(demographics_data),
                "planning": _summarise_planning(planning_data),
                "flood_risk": flood_data.get("risk_level"),
            }),
            "strategy_detector",
        ),
        "renovation_predictor": complete_json(
            prompts.renovation_predictor_prompt({
                "address": address,
                "epc": epc_rating,
                "property_type": _get_property_type(sales, epc_rating),
                "comparables": _summarise_sales(sales),
                "floor_area": epc_rating.get("floor_area_sqm") if epc_rating else None,
                "epc_rating": epc_rating.get("current_energy_rating") if epc_rating else None,
                "epc_potential": epc_rating.get("potential_energy_rating") if epc_rating else None,
            }),
            "renovation_predictor",
        ),
        "floorplan_analysis": complete_json(
            prompts.floorplan_analysis_prompt({
                "address": address,
                "epc": epc_rating,
                "property_type": _get_property_type(sales, epc_rating),
                "habitable_rooms": epc_rating.get("number_habitable_rooms") if epc_rating else None,
                "floor_area": epc_rating.get("floor_area_sqm") if epc_rating else None,
                "construction_year": None,
            }),
            "floorplan_analysis",
        ),
        "neighbourhood_intelligence": complete_json(
            prompts.neighbourhood_intelligence_prompt({
                "address": address,
                "crime": _summarise_crime(crime_data),
                "schools": _summarise_schools(schools_data),
                "transport": _summarise_transport(transport_data),
                "demographics": _summarise_demographics(demographics_data),
                "flood": flood_data.get("risk_level"),
            }),
            "neighbourhood_intelligence",
        ),
        "rental_demand": complete_json(
            prompts.rental_demand_prompt({
                "address": address,
                "demographics": _summarise_demographics(demographics_data),
                "transport_summary": _summarise_transport(transport_data),
                "crime_summary": _summarise_crime(crime_data),
                "recent_sales": _summarise_sales(sales),
            }),
            "rental_demand",
        ),
        "planning_scanner": complete_json(
            prompts.planning_scanner_prompt({
                "address": address,
                "planning_applications": _summarise_planning(planning_data),
                "property_type": _get_property_type(sales, epc_rating),
                "demographics": _summarise_demographics(demographics_data),
            }),
            "planning_scanner",
        ),
        "deal_finder": complete_json(
            prompts.deal_finder_prompt({
                "address": address,
                "comparables": _summarise_sales(sales),
                "epc": epc_rating,
                "risk_factors": f"Flood: {flood_data.get('risk_level')} | Crime: {_summarise_crime(crime_data)}",
                "asking_price": None,
            }),
            "deal_finder",
        ),
        "price_growth": complete_json(
            prompts.price_growth_prompt({
                "address": address,
                "sales_history": _sales_history(sales),
                "demographics": _summarise_demographics(demographics_data),
                "transport": _summarise_transport(transport_data),
                "planning": _summarise_planning(planning_data),
                "flood": flood_data.get("risk_level"),
                "latest_price": _latest_price(sales),
            }),
            "price_growth",
        ),
        "rental_yield_simulator": complete_json(
            prompts.rental_yield_simulator_prompt({
                "address": address,
                "estimated_value": _latest_price(sales),
                "epc": epc_rating,
                "rental_demand_score": 65,      # placeholder; overridden after rental_demand resolves
                "demographics": _summarise_demographics(demographics_data),
            }),
            "rental_yield_simulator",
        ),
    }

    results = await asyncio.gather(*feature_tasks.values(), return_exceptions=True)
    ai = {}
    for key, result in zip(feature_tasks.keys(), results):
        if isinstance(result, Exception):
            log.warning("ai_feature_failed", feature=key, error=str(result))
            ai[key] = {"error": str(result)}
        else:
            ai[key] = result

    # AI summary uses the other results
    rental_demand_score = ai.get("rental_demand", {}).get("rental_demand_score", 65)
    investment_score = ai.get("investment_score", {})
    price_growth = ai.get("price_growth", {})
    neighbourhood = ai.get("neighbourhood_intelligence", {})

    ai["ai_summary"] = await complete_text(
        prompts.ai_summary_prompt({
            "address": address,
            "investment_score": investment_score.get("score", "N/A"),
            "investment_grade": investment_score.get("grade", "N/A"),
            "primary_strategy": ai.get("strategy_detector", {}).get("primary_strategy", "N/A"),
            "estimated_value": price_growth.get("current_estimate_gbp", 0),
            "gross_yield": ai.get("rental_yield_simulator", {}).get("gross_yield_pct", "N/A"),
            "five_year_growth": _pct_growth(
                price_growth.get("current_estimate_gbp"),
                price_growth.get("five_year_forecast_gbp"),
            ),
            "neighbourhood_summary": neighbourhood.get("overall_desirability", "N/A"),
            "key_risks": investment_score.get("key_risks", []),
            "key_positives": investment_score.get("key_positives", []),
        }),
        "ai_summary",
    )

    return ai


def assemble_report(address: str, coords: dict, raw: dict, ai: dict) -> dict:
    """Combine geocode, raw data, and AI results into the final report dict."""
    return {
        "address": address,
        "coordinates": {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "display_name": coords.get("display_name", address),
        },
        "generated_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(seconds=settings.cache_ttl_seconds)).isoformat(),
        "data_sources": {
            "land_registry": "error" not in raw.get("land_registry", {}),
            "epc": "error" not in raw.get("epc", {}),
            "crime": "error" not in raw.get("crime", {}),
            "demographics": "error" not in raw.get("demographics", {}),
            "flood_risk": "error" not in raw.get("flood", {}),
            "planning": "error" not in raw.get("planning", {}),
            "schools": "error" not in raw.get("schools", {}),
            "transport": "error" not in raw.get("transport", {}),
        },
        # 12 AI features
        "investment_score": ai.get("investment_score", {}),
        "strategy_detector": ai.get("strategy_detector", {}),
        "renovation_predictor": ai.get("renovation_predictor", {}),
        "floorplan_analysis": ai.get("floorplan_analysis", {}),
        "neighbourhood_intelligence": ai.get("neighbourhood_intelligence", {}),
        "rental_demand_score": ai.get("rental_demand", {}).get("rental_demand_score", 0),
        "rental_demand_detail": ai.get("rental_demand", {}),
        "planning_scanner": ai.get("planning_scanner", {}),
        "deal_finder": ai.get("deal_finder", {}),
        "price_growth_predictor": ai.get("price_growth", {}),
        "rental_yield_simulator": ai.get("rental_yield_simulator", {}),
        "ai_summary": ai.get("ai_summary", ""),
        # Raw data (always included for transparency)
        "raw_data": {
            "land_registry": raw.get("land_registry"),
            "epc": raw.get("epc"),
            "crime": raw.get("crime"),
            "demographics": raw.get("demographics"),
            "flood": raw.get("flood"),
            "planning": raw.get("planning"),
            "schools": raw.get("schools"),
            "transport": raw.get("transport"),
        },
    }


# ── Helper functions ───────────────────────────────────────────────────────────

def _first_epc_rating(epc_data: dict) -> dict | None:
    ratings = epc_data.get("ratings", [])
    return ratings[0] if ratings else None


def _summarise_sales(sales: list) -> str:
    if not sales:
        return "No recent sales data available"
    lines = []
    for s in sales[:5]:
        lines.append(f"£{s['price_gbp']:,} on {s['date']} ({s['property_type']}, {s['tenure']})")
    return " | ".join(lines)


def _sales_history(sales: list) -> str:
    if not sales:
        return "No sales history"
    return " → ".join(
        f"£{s['price_gbp']:,} ({s['date'][:7]})"
        for s in sorted(sales, key=lambda x: x.get("date", ""))[:10]
    )


def _latest_price(sales: list) -> int | None:
    if not sales:
        return None
    sorted_sales = sorted(sales, key=lambda x: x.get("date", ""), reverse=True)
    return sorted_sales[0].get("price_gbp")


def _summarise_crime(crime_data: dict) -> str:
    total = crime_data.get("total_crimes", 0)
    if not total:
        return "No crime data available"
    top = crime_data.get("by_category", [])[:3]
    top_str = ", ".join(f"{c['category']} ({c['count']})" for c in top)
    return f"{total} crimes in period. Top categories: {top_str}"


def _summarise_transport(transport_data: dict) -> str:
    score = transport_data.get("transport_score", "N/A")
    stations = transport_data.get("nearest_stations", [])
    if stations:
        nearest = stations[0]
        return f"Transport score {score}/10. Nearest station: {nearest['name']} ({nearest['distance_m']}m)"
    return f"Transport score {score}/10. No stations found within radius."


def _summarise_schools(schools_data: dict) -> str:
    total = schools_data.get("total_schools", 0)
    best = schools_data.get("best_school")
    if best:
        return f"{total} schools within radius. Best: {best.get('name')} ({best.get('ofsted_rating')})"
    return f"{total} schools within radius."


def _summarise_demographics(demo_data: dict) -> str:
    if not demo_data or demo_data.get("error"):
        return "Demographics data unavailable"
    parts = [
        f"Area: {demo_data.get('area_name', 'Unknown')}",
        f"Ward: {demo_data.get('ward', 'Unknown')}",
        f"IMD decile: {demo_data.get('imd_decile', 'N/A')}",
        f"Region: {demo_data.get('region', 'N/A')}",
    ]
    return " | ".join(parts)


def _summarise_planning(planning_data: dict) -> str:
    total = planning_data.get("total_applications", 0)
    apps = planning_data.get("applications", [])[:3]
    if not apps:
        return f"{total} planning applications found nearby."
    descriptions = "; ".join(
        f"{a.get('application_type', 'Unknown')}: {a.get('description', '')[:80]}"
        for a in apps
    )
    return f"{total} applications nearby. Recent: {descriptions}"


def _get_property_type(sales: list, epc_rating: dict | None) -> str:
    if epc_rating and epc_rating.get("property_type"):
        return epc_rating["property_type"]
    if sales:
        return sales[0].get("property_type", "Unknown")
    return "Unknown"


def _pct_growth(current: int | None, future: int | None) -> str:
    if not current or not future or current == 0:
        return "N/A"
    return f"{((future - current) / current) * 100:.1f}"


def _extract_postcode_from_address(address: str) -> str | None:
    import re
    match = re.search(r"[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}", address.upper())
    return match.group().strip() if match else None
