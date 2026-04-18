"""
Atlas Property Intelligence - Dashboard API v4.1
Production-safe. Zero database dependency. Fully self-contained.
All data fetched live from free UK government APIs.
AI analysis via HuggingFace Inference API with graceful fallback.

Run locally:  uvicorn dashboard_main:app --reload --port 8000
Deploy:       uvicorn dashboard_main:app --host 0.0.0.0 --port $PORT

Environment variables needed:
  HUGGINGFACE_API_KEY - from huggingface.co/settings/tokens (free)
  EPC_API_KEY         - from epc.opendatacommunities.org (free)
  EPC_API_EMAIL       - email used to register for EPC API
"""

import asyncio
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import date, datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_logger(name: str):
    return logging.getLogger(name)

configure_logging()
log = get_logger(__name__)

# ── Environment config ────────────────────────────────────────────────────────
HF_API_KEY    = os.getenv("HUGGINGFACE_API_KEY", "")
EPC_API_KEY   = os.getenv("EPC_API_KEY", "")
EPC_API_EMAIL = os.getenv("EPC_API_EMAIL", "")

# ── External API URLs ─────────────────────────────────────────────────────────
NOMINATIM  = "https://nominatim.openstreetmap.org/search"
POSTCODES  = "https://api.postcodes.io/postcodes"
HMLR       = "https://landregistry.data.gov.uk/landregistry/query"
POLICE_URL = "https://data.police.uk/api/crimes-street/all-crime"
EA_FLOOD   = "https://environment.data.gov.uk/flood-monitoring/id/floods"
OVERPASS   = "https://overpass-api.de/api/interpreter"
EPC_URL    = "https://epc.opendatacommunities.org/api/v1/domestic/search"

# ── VOA 2024 median rents by region and bedroom count ────────────────────────
VOA_RENTS = {
    "london":                   {1: 1750, 2: 2300, 3: 2900, 4: 3800},
    "south east":               {1: 1050, 2: 1350, 3: 1650, 4: 2100},
    "east of england":          {1: 900,  2: 1150, 3: 1400, 4: 1800},
    "south west":               {1: 850,  2: 1100, 3: 1350, 4: 1700},
    "east midlands":            {1: 650,  2: 850,  3: 1000, 4: 1300},
    "west midlands":            {1: 700,  2: 900,  3: 1050, 4: 1350},
    "north west":               {1: 700,  2: 875,  3: 1050, 4: 1350},
    "yorkshire and the humber": {1: 600,  2: 775,  3: 900,  4: 1150},
    "north east":               {1: 525,  2: 650,  3: 775,  4: 975},
    "wales":                    {1: 600,  2: 750,  3: 875,  4: 1100},
    "scotland":                 {1: 800,  2: 1000, 3: 1200, 4: 1550},
    "default":                  {1: 700,  2: 900,  3: 1100, 4: 1400},
}

# ── ONS HPI annual % growth by region ────────────────────────────────────────
ONS_GROWTH = {
    "london": 2.1, "south east": 3.4, "east of england": 2.8,
    "south west": 4.1, "east midlands": 4.8, "west midlands": 4.2,
    "north west": 5.1, "yorkshire and the humber": 4.3,
    "north east": 5.8, "wales": 3.9, "scotland": 4.4,
    "northern ireland": 6.2, "default": 3.8,
}

# ── Typical gross yields by region (used as rent fallback) ───────────────────
REGIONAL_YIELDS = {
    "london": 3.5, "south east": 4.0, "east of england": 4.2,
    "south west": 4.5, "east midlands": 5.5, "west midlands": 5.2,
    "north west": 5.8, "yorkshire and the humber": 5.5,
    "north east": 6.5, "wales": 5.0, "scotland": 5.5,
    "default": 5.0,
}

# ── Property type rent multipliers ───────────────────────────────────────────
PROP_TYPE_MULTIPLIER = {
    "detached": 1.12, "semi-detached": 1.02, "terraced": 0.97,
    "flat": 0.93, "maisonette": 0.93, "bungalow": 0.90,
}

# ── App init ──────────────────────────────────────────────────────────────────
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Atlas Property Intelligence",
    description="UK property analysis API — no database required",
    version="4.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    log.error("unhandled_exception", path=str(request.url), error=str(exc), tb=traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})


# ── Request models ────────────────────────────────────────────────────────────

class PropertyRequest(BaseModel):
    postcode: Optional[str] = None
    address: Optional[str] = None

class PortfolioAddRequest(BaseModel):
    postcode: Optional[str] = None
    address: Optional[str] = None
    property_data: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/analyse-property",
    openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": {
        "type": "object",
        "properties": {
            "address":  {"type": "string", "example": "SW1A 2AA"},
            "postcode": {"type": "string", "example": "SW1A 2AA"},
        }
    }}}}}
)
async def analyse_property(request: Request):
    """Full UK property intelligence. Send either address or postcode."""
    import traceback as _tb
    try:
        raw_text = (await request.body()).decode("utf-8", errors="ignore").strip()
        body = {}
        if raw_text:
            try:
                parsed = json.loads(raw_text)
                body = parsed if isinstance(parsed, dict) else {"address": str(parsed)}
            except (json.JSONDecodeError, ValueError):
                if raw_text.startswith("{"):
                    fixed = re.sub(r'(?<=[{,])\s*([A-Za-z_]\w*)\s*:', r'"\1":', raw_text)
                    try:
                        body = json.loads(fixed)
                    except Exception:
                        body = {"address": raw_text}
                else:
                    body = {"address": raw_text}

        input_location = (body.get("address") or body.get("postcode") or "").strip()
        if not input_location:
            raise HTTPException(status_code=422, detail="Provide 'address' or 'postcode'")

        coords = await _geocode(input_location)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": _tb.format_exc()})

    lat    = coords["latitude"]
    lng    = coords["longitude"]
    region = coords.get("region", "").lower()

    # Extract postcode for downstream APIs — prefer geocoder result, then regex from input
    _pc_match = re.search(r'[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}', input_location.upper())
    rpc = coords.get("postcode") or (_pc_match.group(0).replace(" ", "") if _pc_match else input_location.upper()[:8])

    # Step 2: Fan out to all data sources concurrently
    fetched = await asyncio.gather(
        _fetch_sales(rpc),
        _fetch_epc(rpc),
        _fetch_crime(lat, lng),
        _fetch_demographics(rpc),
        _fetch_flood(lat, lng),
        _fetch_transport(lat, lng),
        _fetch_planning_data(lat, lng),
        _fetch_schools(lat, lng),
        return_exceptions=True,
    )

    sales      = _sr(fetched[0], [])
    epc_list   = _sr(fetched[1], [])
    crime_d    = _sr(fetched[2], {})
    demo_d     = _sr(fetched[3], {})
    flood_d    = _sr(fetched[4], {})
    trans_d    = _sr(fetched[5], {})
    planning_d = _sr(fetched[6], {})
    schools_d  = _sr(fetched[7], [])
    epc        = _best_epc(epc_list)  # most recent certificate for this postcode

    # Use postcodes.io region (proper name like "North East") over Nominatim's "England"
    region = demo_d.get("region", region).lower() if demo_d.get("region") else region

    # Step 3: Derive all values
    beds        = _infer_bedrooms(epc)
    floor_area  = _f(epc.get("total-floor-area") or epc.get("floor_area_sqm"), 0.0)
    prop_type   = epc.get("property-type") or epc.get("property_type") or "Residential"
    epc_rating  = epc.get("current-energy-rating") or epc.get("current_energy_rating")

    # Resolve crime/transport first — needed for rent calculation
    crime_tot   = crime_d.get("total_crimes", 0)
    crime_sc    = _crime_score(crime_tot)
    trans_sc    = trans_d.get("transport_score", 0)
    flood_lv    = flood_d.get("risk_level", "Unknown")

    # Property character — council/tenure detection and construction era
    imd_decile    = _i(demo_d.get("imd_decile"), 5)
    tenure_info   = _detect_tenure_type(epc, sales, imd_decile)
    construction  = _construction_era(epc)
    built_form    = epc.get("built-form") or epc.get("built_form") or "Unknown"
    ext_count     = _i(epc.get("extension-count"), 0)

    # UKHPI — official LR average price + type-specific price + 6-month trend
    ukhpi_d     = await _fetch_ukhpi_data(demo_d.get("admin_district", ""), prop_type)
    ukhpi_price = ukhpi_d.get("type_avg") or ukhpi_d.get("district_avg", 0)

    est_value   = _calc_value(sales, region, floor_area, beds, prop_type, ukhpi_price)
    rent        = _voa_rent(region, beds, prop_type, trans_sc, crime_sc)
    if not sales and est_value:
        rent = max(rent, _rent_from_value(est_value, region))
    est_value, rent, _val_warnings = _validate_financials(est_value, rent, region, sales)
    g_yield     = round(rent * 12 / est_value * 100, 2) if est_value else 0.0
    deposit     = int(est_value * 0.25)
    loan        = est_value - deposit
    mortgage    = int(loan * 0.055 / 12)
    annual_costs = mortgage * 12 + int(est_value * 0.01) + int(rent * 12 * 0.10)
    net_yield   = round((rent * 12 - annual_costs) / est_value * 100, 2) if est_value else 0.0
    cashflow    = rent - mortgage - int(est_value * 0.01 / 12) - int(rent * 0.10)
    annual_p    = cashflow * 12

    growth_r    = _get_growth(region)
    val_1yr     = int(est_value * (1 + growth_r / 100))
    val_3yr     = int(est_value * ((1 + growth_r / 100) ** 3))
    val_5yr     = int(est_value * ((1 + growth_r / 100) ** 5))

    inv_sc      = _investment_score(g_yield, crime_sc, trans_sc, flood_lv, sales)
    risk_sc     = _risk_score(flood_lv, crime_tot, demo_d)
    liq_sc      = _liquidity_score(sales)
    deal_sc     = _deal_score_calc(sales)
    rd_sc       = _rental_demand_score(region, trans_sc, crime_sc)
    st_sc       = _street_score(crime_sc, liq_sc, trans_sc)

    strategy    = _recommend_strategy(g_yield, beds, floor_area, region)
    strategies  = _all_strategies(g_yield, beds, floor_area)

    loft_ok     = _loft_viable(prop_type, epc)
    ext_ok      = _extension_viable(prop_type, epc)
    dev_cost    = (35000 if loft_ok else 0) + (45000 if ext_ok else 0)
    dev_uplift  = int(dev_cost * 1.55)
    dev_roi     = round((dev_uplift - dev_cost) / dev_cost * 100, 1) if dev_cost else 0.0
    dev_sc      = (25 if loft_ok else 0) + (30 if ext_ok else 0) + 20

    hmo_rooms   = max(0, beds - 1) if beds >= 4 else 0
    hmo_rent    = hmo_rooms * 550 if hmo_rooms > 0 else 0
    hmo_yield   = round(hmo_rent * 12 / est_value * 100, 2) if est_value and hmo_rent else 0.0

    stamp       = _stamp_duty(est_value)

    # Step 4: AI analysis
    ai = await _run_ai(
        rpc, est_value, rent, g_yield, inv_sc,
        strategy, crime_tot, flood_lv, region, beds,
        trans_sc, epc_rating, floor_area, prop_type,
    )
    comps = [
        {
            "address": f"{s.get('address_paon','').strip()} {s.get('street','').strip()}".strip() or "Nearby property",
            "price_gbp": s.get("price_gbp", 0),
            "date": s.get("date", ""),
            "type": s.get("property_type", ""),
            "tenure": s.get("tenure", ""),
        }
        for s in sales[:5]
    ]

    return {
        "postcode": rpc,
        "display_address": coords.get("display_name", input_location),
        "latitude": lat,
        "longitude": lng,
        "generated_at": datetime.utcnow().isoformat(),

        "property": {
            "bedrooms": beds,
            "bathrooms": max(1, beds - 1),
            "floor_area_sqm": floor_area,
            "property_type": prop_type,
            "built_form": built_form,
            "construction_era": construction,
            "extensions": ext_count,
            "tenure_category": tenure_info["category"],
            "tenure_label": tenure_info["label"],
            "is_social_housing": tenure_info["is_social_housing"],
            "epc_rating": epc_rating,
            "epc_current_efficiency": _i(epc.get("current-energy-efficiency") or epc.get("current_energy_efficiency"), 0),
            "epc_potential_rating": epc.get("potential-energy-rating") or epc.get("potential_energy_rating"),
            "legal_tenure": sales[0].get("tenure") if sales else "Unknown",
            "walls": epc.get("walls-description") or epc.get("walls_description"),
            "roof": epc.get("roof-description") or epc.get("roof_description"),
            "heating": epc.get("main-heat-description") or epc.get("heating_description"),
            "windows": epc.get("windows-description") or epc.get("windows_description"),
            "mains_gas": bool(epc.get("mains-gas-flag") or epc.get("mains_gas_flag")),
            "epc_inspection_date": epc.get("inspection-date") or epc.get("lodgement-date"),
        },

        "financials": {
            "estimated_value": est_value,
            "monthly_rent": rent,
            "annual_rent": rent * 12,
            "rental_yield": g_yield,
            "net_yield": net_yield,
            "monthly_cashflow": cashflow,
            "annual_profit": annual_p,
            "monthly_mortgage_estimate": mortgage,
            "deposit_required": deposit,
            "stamp_duty_estimate": stamp,
            "total_acquisition_cost": deposit + stamp + 2500,
        },

        "scores": {
            "investment_score": inv_sc,
            "investment_grade": _grade(inv_sc),
            "risk_score": risk_sc,
            "risk_level": _risk_label(risk_sc),
            "liquidity_score": liq_sc,
            "liquidity_band": _liq_label(liq_sc),
            "street_score": st_sc,
            "street_grade": _grade(st_sc),
            "deal_score": deal_sc,
            "rental_demand_score": rd_sc,
            "demand_level": _demand_label(rd_sc),
        },

        "growth": {
            "current_value": est_value,
            "one_year_projection": val_1yr,
            "three_year_projection": val_3yr,
            "five_year_projection": val_5yr,
            "annual_growth_rate_pct": growth_r,
            "one_year_uplift": val_1yr - est_value,
            "five_year_uplift": val_5yr - est_value,
            "source": "ONS House Price Index regional data",
        },

        "market": {
            "district_avg_price": ukhpi_d.get("district_avg", 0),
            "type_avg_price": ukhpi_d.get("type_avg", 0),
            "price_vs_district_avg": (
                round((est_value - ukhpi_d["district_avg"]) / ukhpi_d["district_avg"] * 100, 1)
                if ukhpi_d.get("district_avg") else None
            ),
            "six_month_trend_pct": ukhpi_d.get("trend_pct_6m", 0),
            "market_direction": ukhpi_d.get("trend_label", "Unknown"),
            "ukhpi_data_period": ukhpi_d.get("data_period", ""),
            "price_per_sqm": int(est_value / floor_area) if floor_area >= 30 else None,
            "district_avg_psm": int(ukhpi_d["district_avg"] / 90) if ukhpi_d.get("district_avg") else None,
            "comparable_count": len(sales),
            "source": "UKHPI / Land Registry",
        },

        "ai_analysis": {
            "best_strategy": strategy,
            "all_strategies": strategies,
            "reason": ai.get("reason") or f"{strategy} recommended based on {g_yield:.1f}% gross yield and local market conditions.",
            "key_positives": ai.get("key_positives") or _default_positives(inv_sc, g_yield, trans_sc),
            "key_risks": ai.get("key_risks") or _default_risks(risk_sc, flood_lv, crime_tot),
            "void_period_weeks": 4 if rd_sc >= 60 else 8,
            "tenant_profiles": _tenant_profiles(region, strategy),
            "summary": ai.get("summary") or _default_summary(rpc, inv_sc, strategy, est_value, g_yield, val_5yr),
        },

        "renovation": {
            "current_value": est_value,
            "light": {"cost": 20000, "arv": int(est_value * 1.08), "roi_pct": 8.0, "works": ["New kitchen", "New bathrooms", "Redecoration"]},
            "medium": {"cost": 45000, "arv": int(est_value * 1.18), "roi_pct": 18.0, "works": ["Full refurb", "Rewire", "Insulation", "Double glazing"]},
            "heavy":  {"cost": 85000, "arv": int(est_value * 1.32), "roi_pct": 32.0, "works": ["Full refurb", "Extension", "Loft conversion", "New heating"]},
            "epc_upgrade_cost": 8000,
            "epc_upgrade_notes": "Loft insulation, cavity wall fill, and heating upgrade to reach EPC C",
        },

        "development": {
            "score": dev_sc,
            "roi_pct": dev_roi,
            "current_value": est_value,
            "post_dev_value": est_value + dev_uplift,
            "uplift": dev_uplift,
            "total_cost": dev_cost,
            "loft": {"viable": loft_ok, "feasibility": "High" if loft_ok else "Low", "cost": 35000 if loft_ok else 0, "value_add": 55000 if loft_ok else 0},
            "extension": {"viable": ext_ok, "feasibility": "Medium" if ext_ok else "Low", "cost": 45000 if ext_ok else 0, "value_add": 65000 if ext_ok else 0},
            "hmo": {"viable": hmo_rooms > 0, "rooms": hmo_rooms, "monthly_rent": hmo_rent, "conversion_cost": hmo_rooms * 3500},
        },

        "risk": {
            "overall_score": risk_sc,
            "band": _risk_label(risk_sc),
            "flood_level": flood_lv,
            "flood_zone": flood_d.get("flood_zone", "Unknown"),
            "flood_zone_label": flood_d.get("flood_zone_label", ""),
            "flood_warnings": flood_d.get("active_warning_count", 0),
            "active_flood_warnings": flood_d.get("active_warnings", []),
            "crime_score": crime_sc,
            "crime_total": crime_tot,
            "crime_breakdown": (crime_d.get("by_category") or [])[:5],
            "imd_decile": imd_decile,
            "economic_vulnerability": imd_decile * 10,
            "red_flags": _red_flags(flood_lv, crime_tot, risk_sc),
            "suitable_for": _suitable_for(risk_sc),
        },

        "neighbourhood": {
            "overall_desirability": _desirability(inv_sc, crime_sc, trans_sc),
            "desirability_score": _area_desirability_score(crime_sc, trans_sc, demo_d),
            "area_trajectory": _trajectory(region, growth_r),
            "growth_classification": _growth_classification(region, growth_r, crime_tot, imd_decile),
            "income_estimate": _income_est(region),
            "investor_appeal": "High" if inv_sc >= 65 else "Medium" if inv_sc >= 45 else "Low",
            "transport_score": trans_sc,
            "transport_summary": _transport_summary(trans_d),
            "nearest_stations": (trans_d.get("nearest_stations") or [])[:3],
            "bus_stop_count": trans_d.get("bus_stop_count", 0),
            "schools_nearby": schools_d,
            "nearest_school": schools_d[0]["name"] if schools_d else "None found within 1km",
            "school_count_1km": len(schools_d),
            "demographics": {
                "area":                     demo_d.get("area_name") or demo_d.get("admin_district"),
                "ward":                     demo_d.get("ward"),
                "region":                   demo_d.get("region"),
                "local_authority":          demo_d.get("local_authority") or demo_d.get("admin_district"),
                "lsoa":                     demo_d.get("lsoa", ""),
                "msoa":                     demo_d.get("msoa", ""),
                "parliamentary_constituency": demo_d.get("parliamentary_constituency", ""),
                "police_force":             demo_d.get("police_force", ""),
                "nhs_icb":                  demo_d.get("nhs_icb", ""),
                "imd_decile":               imd_decile,
                "imd_label":                (
                    "Most deprived 10%" if imd_decile <= 1 else
                    f"Decile {imd_decile} of 10 (1=most deprived)"
                ),
            },
        },

        "planning": {
            "risk_level": planning_d.get("risk_level", "low"),
            "risk_summary": planning_d.get("risk_summary", "No major planning restrictions detected."),
            "article_4_active": planning_d.get("article_4_active", False),
            "article_4_directions": planning_d.get("article_4_directions", []),
            "conservation_area": planning_d.get("conservation_area", False),
            "conservation_area_name": planning_d.get("conservation_area_name"),
            "listed_building": planning_d.get("listed_building", False),
            "listed_building_grade": planning_d.get("listed_building_grade"),
            "permitted_development_likely": planning_d.get("permitted_development_likely", True),
            "hmo_pd_blocked": planning_d.get("hmo_pd_blocked", False),
            "source": "GOV.UK Planning Data API",
        },

        "comparables": {
            "sales": comps,
            "total_transactions": len(sales),
            "avg_price": int(sum(s.get("price_gbp", 0) for s in sales) / len(sales)) if sales else 0,
            "median_price": int(statistics.median([s.get("price_gbp", 0) for s in sales])) if sales else 0,
            "min_price": min((s.get("price_gbp", 0) for s in sales), default=0),
            "max_price": max((s.get("price_gbp", 0) for s in sales), default=0),
            "latest_sale_price": sales[0].get("price_gbp") if sales else 0,
            "latest_sale_date": sales[0].get("date") if sales else None,
            "price_per_sqm": int(est_value / floor_area) if floor_area >= 30 and est_value else None,
        },

        "deals": {
            "score": deal_sc,
            "label": _deal_label(deal_sc),
            "potential_deals": _find_deals(sales),
            "best_deal": _best_deal(sales),
            "recommendation": _deal_recommendation(deal_sc, sales),
            "median_area_price": int(statistics.median([s.get("price_gbp", 0) for s in sales])) if sales else 0,
        },

        "hmo_analysis": {
            "feasibility": "Blocked — Article 4" if planning_d.get("hmo_pd_blocked") else ("Medium" if hmo_rooms > 0 else "Low"),
            "room_potential": hmo_rooms,
            "article_4_restriction": planning_d.get("hmo_pd_blocked", False),
            "hmo_notes": (
                "Article 4 direction active — full planning permission required before HMO conversion."
                if planning_d.get("hmo_pd_blocked")
                else ("Minimum room sizes and fire safety upgrades required. Check local HMO licensing scheme." if hmo_rooms > 0 else "Property likely too small for HMO.")
            ),
            "estimated_monthly_hmo_rent": hmo_rent,
            "hmo_gross_yield": hmo_yield,
            "hmo_cashflow": max(0, hmo_rent - mortgage - 200),
        },

        "confidence": _confidence_score(sales, epc, demo_d, crime_tot),

        "data_validation": {
            "warnings": _val_warnings,
            "yield_realistic": 1.5 <= g_yield <= 15,
            "value_realistic": 40_000 <= est_value <= 5_000_000,
        },

        "data_freshness": {
            "price": f"Latest sale: {sales[0].get('date','Unknown')[:7]}" if sales else "No sales data found",
            "rent": "VOA Private Rental Market Statistics 2024",
            "crime": "Police API — current month",
            "last_updated": datetime.utcnow().isoformat(),
        },

        "data_sources": {
            "land_registry": len(sales) > 0,
            "epc": bool(epc),
            "crime": crime_tot > 0,
            "demographics": bool(demo_d),
            "flood": flood_lv != "Unknown",
            "transport": trans_sc > 0,
            "planning": planning_d.get("risk_level") is not None,
            "schools": len(schools_d) > 0,
            "ukhpi": ukhpi_price > 0,
            "tenure_detected": tenure_info["category"] != "private",
            "active_count": sum([len(sales) > 0, bool(epc), crime_tot > 0, bool(demo_d),
                                  flood_lv != "Unknown", trans_sc > 0,
                                  planning_d.get("risk_level") is not None, len(schools_d) > 0]),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO (in-memory, no database)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/portfolio/add")
async def portfolio_add(data: PortfolioAddRequest):
    return {"status": "success"}


@app.get("/portfolio")
async def portfolio_list():
    return {"total": 0, "properties": []}


@app.delete("/portfolio/{property_id}")
async def portfolio_delete(property_id: int):
    return {"status": "removed", "total": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/market-heatmap")
async def get_market_heatmap(location: str, postcode: Optional[str] = None):
    try:
        geo = await _fetch_demographics(postcode or location)
        region = geo.get("region", "").lower()
        sales = await _fetch_sales((postcode or location)[:8])
        momentum = _calc_momentum(sales)
        growth = _get_growth(region)
        return {
            "location": location,
            "opportunity_score": min(100, int(50 + growth * 5 + momentum * 200)),
            "price_momentum": round(momentum, 3),
            "price_momentum_label": "strong growth" if momentum > 0.06 else "moderate growth" if momentum > 0.02 else "flat",
            "rental_demand": "high" if any(r in region for r in ["london", "manchester", "birmingham", "leeds"]) else "medium",
            "liquidity_score": _liquidity_score(sales),
            "investor_competition": "medium",
            "market_phase": "growth" if growth > 4 else "stable",
            "avg_price_gbp": int(statistics.mean([s.get("price_gbp", 0) for s in sales])) if sales else 0,
            "transaction_count": len(sales),
            "ons_annual_growth_pct": growth,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deal-scanner")
async def get_deal_scanner(postcode: str):
    try:
        sales = await _fetch_sales(postcode)
        score = _deal_score_calc(sales)
        median = int(statistics.median([s.get("price_gbp", 0) for s in sales])) if sales else 0
        return {
            "postcode": postcode.upper(),
            "deal_score": score,
            "deal_score_label": _deal_label(score),
            "median_price_gbp": median,
            "transaction_count": len(sales),
            "potential_deals": _find_deals(sales),
            "best_deal": _best_deal(sales),
            "recommendation": _deal_recommendation(score, sales),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-analysis")
async def get_risk_analysis(address: str, postcode: str):
    try:
        coords = await _geocode(address)
        crime, flood, demo = await asyncio.gather(
            _fetch_crime(coords["latitude"], coords["longitude"]),
            _fetch_flood(coords["latitude"], coords["longitude"]),
            _fetch_demographics(postcode),
        )
        risk = _risk_score(flood.get("risk_level", "Unknown"), crime.get("total_crimes", 0), demo)
        return {
            "address": address,
            "investment_risk_score": risk,
            "risk_band": _risk_label(risk),
            "flood_level": flood.get("risk_level", "Unknown"),
            "crime_total": crime.get("total_crimes", 0),
            "red_flags": _red_flags(flood.get("risk_level", "Unknown"), crime.get("total_crimes", 0), risk),
            "suitable_for": _suitable_for(risk),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/true-value")
async def get_true_value(address: str, postcode: str):
    try:
        sales, epc_list, geo = await asyncio.gather(
            _fetch_sales(postcode),
            _fetch_epc(postcode),
            _fetch_demographics(postcode),
        )
        epc = epc_list[0] if epc_list else {}
        region = geo.get("region", "").lower()
        floor_area = _f(epc.get("total-floor-area"), 0.0)
        beds = _infer_bedrooms(epc)
        prop_type = (epc.get("property-type") or "").lower()
        value = _calc_value(sales, region, floor_area, beds, prop_type)
        return {
            "address": address,
            "postcode": postcode.upper(),
            "consensus_value_gbp": value,
            "confidence_score": 75 if len(sales) >= 5 else 50,
            "confidence_band": "High" if len(sales) >= 5 else "Medium",
            "comparable_count": len(sales),
            "value_range_low": int(value * 0.92),
            "value_range_high": int(value * 1.08),
            "price_per_sqm_gbp": int(value / floor_area) if floor_area > 0 and value > 0 else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/liquidity-score")
async def get_liquidity(postcode: str):
    try:
        sales = await _fetch_sales(postcode)
        geo = await _fetch_demographics(postcode)
        score = _liquidity_score(sales)
        return {
            "postcode": postcode.upper(),
            "liquidity_score": score,
            "liquidity_band": _liq_label(score),
            "estimated_time_to_sell_weeks": "4-8 weeks" if score >= 75 else "8-16 weeks" if score >= 50 else "16+ weeks",
            "transaction_frequency": len(sales),
            "avg_price_gbp": int(statistics.mean([s.get("price_gbp", 0) for s in sales])) if sales else 0,
            "region": geo.get("region", "Unknown"),
            "recommendation": "Active market — strong exit strategy" if score >= 60 else "Limited liquidity — plan for 4+ months to sell",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/development-potential")
async def get_development(address: str, postcode: str):
    try:
        sales, epc_list, geo = await asyncio.gather(
            _fetch_sales(postcode),
            _fetch_epc(postcode),
            _fetch_demographics(postcode),
        )
        epc = epc_list[0] if epc_list else {}
        region = geo.get("region", "").lower()
        floor_area = _f(epc.get("total-floor-area"), 0.0)
        beds = _infer_bedrooms(epc)
        prop_type = (epc.get("property-type") or "house").lower()
        value = _calc_value(sales, region, floor_area, beds, prop_type)
        loft = _loft_viable(prop_type, epc)
        ext = _extension_viable(prop_type, epc)
        cost = (35000 if loft else 0) + (45000 if ext else 0)
        uplift = int(cost * 1.55)
        roi = round((uplift - cost) / cost * 100, 1) if cost else 0
        return {
            "address": address,
            "overall_development_score": (25 if loft else 0) + (30 if ext else 0) + 20,
            "current_value_gbp": value,
            "post_dev_value_gbp": value + uplift,
            "uplift_gbp": uplift,
            "total_dev_cost_gbp": cost,
            "development_roi_pct": roi,
            "loft_viable": loft,
            "loft_cost_gbp": 35000 if loft else 0,
            "extension_viable": ext,
            "extension_cost_gbp": 45000 if ext else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.0.0",
        "database": "none — stateless deployment",
        "hf_configured": bool(HF_API_KEY),
        "epc_configured": bool(EPC_API_KEY),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

async def _geocode(address: str) -> dict:
    params  = {"q": address, "format": "json", "addressdetails": 1, "limit": 1, "countrycodes": "gb"}
    headers = {"User-Agent": "AtlasPropertyIntelligence/4.0"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(NOMINATIM, params=params, headers=headers)
        resp.raise_for_status()
        results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode: {address}")
    r = results[0]
    addr_detail = r.get("address", {})
    return {
        "latitude":     float(r["lat"]),
        "longitude":    float(r["lon"]),
        "display_name": r.get("display_name", address),
        "postcode":     addr_detail.get("postcode"),
        "region":       addr_detail.get("state", ""),
    }


async def _fetch_demographics(postcode: str) -> dict:
    try:
        pc = postcode.replace(" ", "").upper()[:8]
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{POSTCODES}/{pc}")
            if resp.status_code == 200:
                r = resp.json().get("result", {})
                codes = r.get("codes", {})
                return {
                    "region":                    r.get("region", ""),
                    "ward":                      r.get("admin_ward", ""),
                    "area_name":                 r.get("admin_district", ""),
                    "local_authority":           r.get("admin_district", ""),
                    "imd_decile":                r.get("imd"),
                    "admin_district":            r.get("admin_district", ""),
                    "lsoa":                      r.get("lsoa", ""),
                    "msoa":                      r.get("msoa", ""),
                    "lsoa_code":                 codes.get("lsoa", ""),
                    "parliamentary_constituency": r.get("parliamentary_constituency", ""),
                    "country":                   r.get("country", "England"),
                    "latitude":                  r.get("latitude"),
                    "longitude":                 r.get("longitude"),
                    "outcode":                   r.get("outcode", ""),
                    "nuts_region":               r.get("nuts", ""),
                    "police_force":              r.get("pfa", ""),
                    "nhs_icb":                   r.get("ccg", ""),
                }
    except Exception:
        pass
    return {}


async def _fetch_sales(postcode: str, limit: int = 20) -> list:
    pc_raw = postcode.strip().upper().replace(" ", "")
    # Normalise to spaced format: "NE156DL" → "NE15 6DL" (HMLR stores with space)
    pc_spaced = (pc_raw[:-3] + " " + pc_raw[-3:]) if len(pc_raw) >= 5 else postcode.strip().upper()
    # Derive sector: "NE15 6DL" → "NE15 6"
    m_sec = re.match(r'^([A-Z]{1,2}\d{1,2}[A-Z]?) ?(\d)', pc_spaced)
    sector = f"{m_sec.group(1)} {m_sec.group(2)}" if m_sec else None

    def _build_query(pc_filter: str, is_sector: bool, n: int) -> str:
        if is_sector:
            pc_clause = f'?addr lrcommon:postcode ?_pc . FILTER(STRSTARTS(?_pc, "{pc_filter}"))'
        else:
            pc_clause = f'?addr lrcommon:postcode "{pc_filter}" .'
        return f"""
PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>
SELECT ?amount ?date ?propertyType ?estateType ?paon ?street WHERE {{
  ?trans lrppi:pricePaid ?amount ;
         lrppi:transactionDate ?date ;
         lrppi:propertyType ?propertyType ;
         lrppi:estateType ?estateType ;
         lrppi:propertyAddress ?addr .
  {pc_clause}
  OPTIONAL {{ ?addr lrcommon:paon ?paon }}
  OPTIONAL {{ ?addr lrcommon:street ?street }}
}}
ORDER BY DESC(?date)
LIMIT {n}
"""

    def _parse_bindings(bindings: list) -> list:
        return [
            {
                "price_gbp":     int(float(b["amount"]["value"])),
                "date":          b["date"]["value"],
                "property_type": b.get("propertyType", {}).get("value", "").split("/")[-1],
                "tenure":        "Freehold" if "freehold" in b.get("estateType", {}).get("value", "").lower() else "Leasehold",
                "address_paon":  b.get("paon", {}).get("value", ""),
                "street":        b.get("street", {}).get("value", ""),
            }
            for b in bindings if "amount" in b
        ]

    async def _run_sparql(query: str) -> list:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    HMLR,
                    params={"query": query, "output": "json"},
                    headers={"Accept": "application/sparql-results+json"},
                )
                resp.raise_for_status()
                return _parse_bindings(resp.json().get("results", {}).get("bindings", []))
        except Exception:
            return []

    # First: exact postcode (fast, precise)
    exact = await _run_sparql(_build_query(pc_spaced, is_sector=False, n=10))
    if len(exact) >= 5 or not sector:
        return exact

    # Widen to postcode sector for 10–50× more comparables
    sector_results = await _run_sparql(_build_query(sector, is_sector=True, n=limit))
    seen = {(s["price_gbp"], s["date"], s["address_paon"]) for s in exact}
    merged = exact + [s for s in sector_results if (s["price_gbp"], s["date"], s["address_paon"]) not in seen]
    merged.sort(key=lambda x: x.get("date", ""), reverse=True)
    if len(merged) >= 5:
        return merged[:limit]

    # Last resort: widen to full postcode district (e.g. "NE15")
    district = pc_spaced.split(" ")[0]  # "NE15 6DL" → "NE15"
    if district and district != sector:
        district_results = await _run_sparql(_build_query(district + " ", is_sector=True, n=limit))
        seen2 = {(s["price_gbp"], s["date"], s["address_paon"]) for s in merged}
        merged = merged + [s for s in district_results if (s["price_gbp"], s["date"], s["address_paon"]) not in seen2]
        merged.sort(key=lambda x: x.get("date", ""), reverse=True)
    return merged[:limit]


async def _fetch_epc(postcode: str) -> list:
    if not EPC_API_KEY:
        return []
    try:
        import base64
        creds = base64.b64encode(f"{EPC_API_EMAIL}:{EPC_API_KEY}".encode()).decode()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                EPC_URL,
                params={"postcode": postcode, "size": 5},
                headers={"Accept": "application/json", "Authorization": f"Basic {creds}"},
            )
            if resp.status_code == 200:
                return resp.json().get("rows", [])
    except Exception:
        pass
    return []


async def _fetch_crime(lat: float, lng: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(POLICE_URL, params={"lat": lat, "lng": lng})
            if resp.status_code == 200:
                crimes = resp.json()
                if isinstance(crimes, list):
                    cats = Counter(c.get("category", "") for c in crimes)
                    by_cat = [{"category": k, "count": v} for k, v in sorted(cats.items(), key=lambda x: -x[1])]
                    dates = [c.get("month", "") for c in crimes if c.get("month")]
                    return {
                        "total_crimes": len(crimes),
                        "by_category":  by_cat,
                        "period":       f"{min(dates)} to {max(dates)}" if dates else "unknown",
                    }
    except Exception:
        pass
    return {"total_crimes": 0, "by_category": [], "period": "unavailable"}


EA_FLOOD_ZONES = "https://environment.data.gov.uk/arcgis/rest/services/EA/FloodMapForPlanning/MapServer/{layer}/query"

async def _fetch_flood(lat: float, lng: float) -> dict:
    """
    Two-source flood assessment:
    1. EA Flood Map for Planning (ArcGIS) — official planning-grade Zones 1/2/3
    2. EA Flood Monitoring API — active flood warnings in the vicinity
    """
    geo_params = {
        "geometry": f"{lng},{lat}", "geometryType": "esriGeometryPoint",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outFields": "flood_zone,layer_name", "returnGeometry": "false", "f": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            # Layer 0 = Flood Zone 3 (>1% annual probability), Layer 1 = Zone 2 (0.1–1%)
            fz3_resp, fz2_resp, warn_resp = await asyncio.gather(
                client.get(EA_FLOOD_ZONES.format(layer=0), params=geo_params),
                client.get(EA_FLOOD_ZONES.format(layer=1), params=geo_params),
                client.get(EA_FLOOD, params={"lat": lat, "long": lng, "dist": 2}),
                return_exceptions=True,
            )
        in_fz3 = (not isinstance(fz3_resp, Exception) and fz3_resp.status_code == 200
                  and bool(fz3_resp.json().get("features")))
        in_fz2 = (not isinstance(fz2_resp, Exception) and fz2_resp.status_code == 200
                  and bool(fz2_resp.json().get("features")))
        warnings = []
        if not isinstance(warn_resp, Exception) and warn_resp.status_code == 200:
            warnings = [{"description": i.get("description", "")}
                        for i in warn_resp.json().get("items", [])[:3]]

        if in_fz3:
            zone, risk, label = "Zone 3", "High", "High probability (>1% annual chance). Mortgage/insurance complications likely."
        elif in_fz2:
            zone, risk, label = "Zone 2", "Medium", "Medium probability (0.1–1% annual chance). Flood resilience measures advised."
        else:
            zone, risk, label = "Zone 1", "Low", "Low probability flood zone. Standard insurance terms likely."

        return {
            "risk_level":      risk,
            "flood_zone":      zone,
            "flood_zone_label": label,
            "active_warnings": warnings,
            "active_warning_count": len(warnings),
            "source":          "EA Flood Map for Planning (ArcGIS) + EA Flood Monitoring API",
        }
    except Exception:
        pass
    return {"risk_level": "Unknown", "flood_zone": "Unknown", "flood_zone_label": "Data unavailable.", "active_warnings": []}


async def _fetch_transport(lat: float, lng: float) -> dict:
    query = f"""
[out:json][timeout:10];
(
  node["railway"~"station|halt"](around:800,{lat},{lng});
  node["public_transport"="station"](around:800,{lat},{lng});
);
out body;
"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS, data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            stations = sorted(
                [
                    {
                        "name":       e.get("tags", {}).get("name", "Unnamed station"),
                        "type":       e.get("tags", {}).get("railway", "station"),
                        "distance_m": _haversine(lat, lng, e.get("lat", lat), e.get("lon", lng)),
                    }
                    for e in elements
                ],
                key=lambda s: s["distance_m"],
            )
            score = 8 if len(stations) >= 3 else 6 if len(stations) >= 1 else 2
            return {"transport_score": score, "nearest_stations": stations[:5], "bus_stop_count": 0}
    except Exception:
        pass
    return {"transport_score": 0, "nearest_stations": [], "bus_stop_count": 0}


async def _fetch_ukhpi_data(admin_district: str, prop_type: str = "") -> dict:
    """
    UKHPI SPARQL: last 6 months of LA-level prices, type-specific average, and trend.
    Returns district_avg, type_avg, trend_pct, and the latest data period.
    """
    if not admin_district:
        return {}
    label = admin_district.lower().strip()
    query = f"""
PREFIX ukhpi: <http://landregistry.data.gov.uk/def/ukhpi/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?averagePrice ?avgDetached ?avgSemi ?avgTerraced ?avgFlat ?refPeriod WHERE {{
  ?area rdfs:label ?lbl .
  FILTER(LCASE(STR(?lbl)) = "{label}")
  ?obs ukhpi:refArea ?area ;
       ukhpi:averagePrice ?averagePrice ;
       ukhpi:refPeriod ?refPeriod .
  OPTIONAL {{ ?obs ukhpi:averagePriceDetached ?avgDetached }}
  OPTIONAL {{ ?obs ukhpi:averagePriceSemiDetached ?avgSemi }}
  OPTIONAL {{ ?obs ukhpi:averagePriceTerraced ?avgTerraced }}
  OPTIONAL {{ ?obs ukhpi:averagePriceFlatMaisonette ?avgFlat }}
}}
ORDER BY DESC(?refPeriod)
LIMIT 6
"""
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(
                HMLR,
                params={"query": query, "output": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            resp.raise_for_status()
            rows = resp.json().get("results", {}).get("bindings", [])
            if not rows:
                return {}

        def _val(row, key):
            return float(row[key]["value"]) if key in row else None

        # Type-specific field selection
        pt = prop_type.lower()
        type_key = ("avgFlat" if "flat" in pt or "maisonette" in pt
                    else "avgDetached" if "detached" in pt and "semi" not in pt
                    else "avgSemi" if "semi" in pt
                    else "avgTerraced" if "terrac" in pt
                    else None)

        prices_all = [_val(r, "averagePrice") for r in rows if "averagePrice" in r]
        prices_type = [_val(r, type_key) for r in rows if type_key and type_key in r] if type_key else []

        # 6-month trend: compare newest 3 vs oldest 3
        trend_pct = 0.0
        if len(prices_all) >= 4:
            new_avg = sum(p for p in prices_all[:3] if p) / 3
            old_avg = sum(p for p in prices_all[-3:] if p) / 3
            if old_avg:
                trend_pct = round((new_avg - old_avg) / old_avg * 100, 2)

        return {
            "district_avg":  int(prices_all[0]) if prices_all else 0,
            "type_avg":       int(prices_type[0]) if prices_type else 0,
            "trend_pct_6m":  trend_pct,
            "trend_label":   "Rising" if trend_pct > 1 else "Falling" if trend_pct < -1 else "Stable",
            "data_period":   rows[-1].get("refPeriod", {}).get("value", "")[:7] + " to " +
                             rows[0].get("refPeriod", {}).get("value", "")[:7] if rows else "",
        }
    except Exception:
        pass
    return {}


PLANNING_API = "https://www.planning.data.gov.uk/entity.json"

async def _fetch_planning_data(lat: float, lng: float) -> dict:
    """Gov.uk Planning Data API — free, no key. Article 4, conservation areas, listed buildings."""
    datasets = ["article-4-direction", "conservation-area", "listed-building"]
    results = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            fetched = await asyncio.gather(
                *[client.get(PLANNING_API, params={"longitude": lng, "latitude": lat,
                                                    "dataset": ds, "limit": 5})
                  for ds in datasets],
                return_exceptions=True,
            )
        for ds, resp in zip(datasets, fetched):
            if isinstance(resp, Exception):
                results[ds] = []
            elif resp.status_code == 200:
                results[ds] = resp.json().get("entities", [])
            else:
                results[ds] = []
    except Exception:
        results = {ds: [] for ds in datasets}

    a4      = results.get("article-4-direction", [])
    cons    = results.get("conservation-area", [])
    listed  = results.get("listed-building", [])

    a4_active   = len(a4) > 0
    cons_active = len(cons) > 0
    listed_b    = len(listed) > 0

    # Planning risk level for investors
    if listed_b:
        risk_level, risk_summary = "high", "Listed building — significant restrictions on alterations and extensions."
    elif a4_active and cons_active:
        risk_level, risk_summary = "high", "Article 4 direction and conservation area active — PD rights restricted."
    elif a4_active:
        risk_level, risk_summary = "medium", "Article 4 direction in force — permitted development rights may be restricted."
    elif cons_active:
        risk_level, risk_summary = "medium", "Conservation area — extensions and alterations subject to additional controls."
    else:
        risk_level, risk_summary = "low", "No planning restrictions detected. Standard PD rights likely apply."

    return {
        "risk_level": risk_level,
        "risk_summary": risk_summary,
        "article_4_active": a4_active,
        "article_4_directions": [e.get("name", "") for e in a4[:3]],
        "conservation_area": cons_active,
        "conservation_area_name": cons[0].get("name") if cons else None,
        "listed_building": listed_b,
        "listed_building_grade": listed[0].get("listed-building-grade") if listed else None,
        "permitted_development_likely": not a4_active and not listed_b,
        "hmo_pd_blocked": a4_active,
    }


async def _fetch_schools(lat: float, lng: float) -> list:
    """Overpass API for schools within 1km. Returns name, type, distance."""
    query = f"""
[out:json][timeout:12];
(
  node["amenity"~"school|college|university"](around:1000,{lat},{lng});
  way["amenity"~"school|college|university"](around:1000,{lat},{lng});
  node["amenity"="kindergarten"](around:800,{lat},{lng});
);
out body center;
"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS, data={"data": query})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            schools = []
            seen = set()
            for e in elements:
                tags = e.get("tags", {})
                name = tags.get("name") or tags.get("operator")
                if not name or name in seen:
                    continue
                seen.add(name)
                rlat = e.get("lat") or (e.get("center", {}).get("lat", lat))
                rlon = e.get("lon") or (e.get("center", {}).get("lon", lng))
                schools.append({
                    "name": name,
                    "type": tags.get("amenity", "school"),
                    "distance_m": _haversine(lat, lng, rlat, rlon),
                    "operator": tags.get("operator:type", ""),
                })
            return sorted(schools, key=lambda s: s["distance_m"])[:6]
    except Exception:
        pass
    return []


def _construction_era(epc: dict) -> str:
    band = (epc.get("construction-age-band") or epc.get("construction_age_band") or "").lower()
    if not band or "unknown" in band or "nd" in band:
        return "Unknown"
    if "before 1900" in band or "pre-1900" in band:          return "Pre-1900 (Victorian/Edwardian)"
    if "1900" in band and ("1929" in band or "1919" in band): return "1900–1929 (Interwar)"
    if "1930" in band or "1949" in band:                      return "1930–1949 (Pre-war)"
    if "1950" in band or "1966" in band:                      return "1950–1966 (Post-war / early council)"
    if "1967" in band or "1975" in band:                      return "1967–1975 (Council housing peak)"
    if "1976" in band or "1982" in band:                      return "1976–1982 (Late council / Thatcher era)"
    if "1983" in band or "1990" in band:                      return "1983–1990 (1980s build)"
    if "1991" in band or "2002" in band:                      return "1991–2002 (1990s/2000s)"
    if "2003" in band or "2011" in band:                      return "2003–2011 (Modern)"
    if "2012" in band:                                        return "2012+ (New build)"
    return band.title()


def _detect_tenure_type(epc: dict, sales: list, imd_decile: int) -> dict:
    """Classify property tenure using EPC fields, sale history, and area deprivation."""
    tenure_raw = (epc.get("tenure") or "").lower()
    tx_type    = (epc.get("transaction-type") or "").lower()
    era        = _construction_era(epc)
    prop_type  = (epc.get("property-type") or "").lower()

    score = 0
    signals = []

    # EPC direct tenure signals (strongest evidence)
    if "social" in tenure_raw:
        score += 8
        signals.append("EPC tenure: social rented")
    if "social" in tx_type:
        score += 6
        signals.append("EPC transaction type: social rental")

    # Construction era (council housing peak was 1950–1982)
    if any(x in era for x in ["post-war", "council housing peak", "late council"]):
        score += 3
        signals.append(f"Construction era: {era}")
    elif "1930" in era or "1949" in era:
        score += 1

    # Property form + era combination
    if any(x in prop_type for x in ["semi-detached", "mid-terrace", "terraced"]) and score >= 2:
        score += 1
        signals.append("Standard council housing form")

    # Deprivation (IMD decile 1-3 = most deprived — correlates with council estates)
    if imd_decile and imd_decile <= 2:
        score += 2
        signals.append(f"Highly deprived area (IMD {imd_decile})")
    elif imd_decile and imd_decile <= 4:
        score += 1

    # Right-to-buy evidence: low historic sale price in 1985–2010 window
    for s in sales:
        price, sale_date = s.get("price_gbp", 0), s.get("date", "")
        if sale_date and 1985 <= int(sale_date[:4]) <= 2010 and 0 < price < 70_000:
            score += 3
            signals.append(f"Possible Right to Buy sale £{price:,} in {sale_date[:4]}")
            break

    if score >= 8:
        label, category = "Council / Social Housing", "social_rented"
    elif score >= 5:
        label, category = "Likely Council / Housing Association", "probable_social"
    elif score >= 3:
        label, category = "Possibly Former Council Property", "former_council"
    elif "private" in tenure_raw:
        label, category = "Private Rented", "private_rented"
    elif "owner" in tenure_raw:
        label, category = "Owner Occupied", "owner_occupied"
    else:
        label, category = "Private / Owner Occupied", "private"

    return {
        "category": category,
        "label": label,
        "confidence_score": min(score, 10),
        "signals": signals,
        "epc_tenure_raw": tenure_raw or "not recorded",
        "is_social_housing": score >= 5,
    }


def _best_epc(epc_list: list) -> dict:
    """Pick the most recent EPC record from a list."""
    if not epc_list:
        return {}
    def _epc_date(e):
        return e.get("lodgement-date") or e.get("lodgement_date") or e.get("inspection-date") or "1900-01-01"
    return max(epc_list, key=_epc_date)


# ═══════════════════════════════════════════════════════════════════════════════
# AI VIA HUGGING FACE
# ═══════════════════════════════════════════════════════════════════════════════

HF_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"


async def generate_ai_summary(data: dict) -> str:
    """
    Call HuggingFace Inference API to generate an investor summary.
    Returns the generated text string, or a rule-based fallback if the
    API is unavailable or the key is not set.
    """
    if not HF_API_KEY:
        return _hf_fallback(data)

    prompt = (
        "You are a UK property investment expert. "
        "Analyse this property data and give a short investor summary "
        "with pros, risks, and best strategy: "
        + str(data)
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                HF_MODEL_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": prompt},
            )
            resp.raise_for_status()
            result = resp.json()

            # HF returns: [{"generated_text": "..."}]
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "")
                # Strip the echoed prompt if model includes it
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                return text if text else _hf_fallback(data)

            # Some models return a dict with "generated_text" directly
            if isinstance(result, dict):
                text = result.get("generated_text", "")
                return text if text else _hf_fallback(data)

    except Exception:
        pass

    return _hf_fallback(data)


def _hf_fallback(data: dict) -> str:
    """Rule-based fallback summary when HuggingFace is unavailable."""
    postcode    = data.get("postcode", "this property")
    value       = data.get("estimated_value", 0)
    yield_pct   = data.get("rental_yield", 0)
    inv_score   = data.get("investment_score", 0)
    strategy    = data.get("best_strategy", "BTL")
    risk        = data.get("risk_level", "Medium")
    val_5yr     = data.get("five_year_projection", 0)
    uplift      = val_5yr - value if val_5yr and value else 0
    grade       = "A" if inv_score >= 80 else "B" if inv_score >= 65 else "C" if inv_score >= 50 else "D"

    return (
        f"{postcode} scores {inv_score}/100 (Grade {grade}). "
        f"Estimated value £{value:,} with a gross yield of {yield_pct:.1f}%. "
        f"Risk profile: {risk}. "
        f"Recommended strategy: {strategy}. "
        f"Five-year price forecast: £{val_5yr:,} (uplift £{uplift:,}). "
        f"Conduct full due diligence including structural survey and local authority search before proceeding."
    )


async def _run_ai(postcode, value, rent, yield_pct, inv_score, strategy,
                  crime, flood, region, beds, transport, epc_rating,
                  floor_area, prop_type) -> dict:
    """
    Wrapper that calls generate_ai_summary and returns a dict compatible
    with the existing ai_analysis response structure.
    """
    data = {
        "postcode": postcode,
        "estimated_value": value,
        "monthly_rent": rent,
        "rental_yield": yield_pct,
        "investment_score": inv_score,
        "best_strategy": strategy,
        "crime_incidents": crime,
        "flood_risk": flood,
        "region": region,
        "bedrooms": beds,
        "transport_score": transport,
        "epc_rating": epc_rating,
        "floor_area_sqm": floor_area,
        "property_type": prop_type,
        "five_year_projection": int(value * ((1 + _get_growth(region) / 100) ** 5)) if value else 0,
        "risk_level": "High" if crime > 150 else "Medium" if crime > 50 else "Low",
    }
    summary = await generate_ai_summary(data)
    return {"summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_value(sales: list, region: str, floor_area: float, beds: int = 3,
                prop_type: str = "", ukhpi_district_avg: int = 0) -> int:
    # Regional price-per-sqm benchmarks (2024, £/sqm median)
    _PSM = {
        "london": 7500, "south east": 4200, "east of england": 3400,
        "south west": 3200, "east midlands": 2400, "west midlands": 2500,
        "north west": 2300, "yorkshire and the humber": 2000,
        "north east": 1800, "wales": 2000, "scotland": 2200, "default": 2600,
    }
    psm = next((v for k, v in _PSM.items() if k != "default" and k in region), _PSM["default"])

    if not sales:
        rent_anchor = _voa_rent(region, beds) * 12 * 18
        sqm_anchor  = int(floor_area * psm) if floor_area >= 30 else 0
        anchors = [a for a in [rent_anchor, sqm_anchor, ukhpi_district_avg] if a > 40_000]
        return int(sum(anchors) / len(anchors)) if anchors else rent_anchor

    today = date.today()
    pt_norm = prop_type.lower().split("/")[-1]
    annual_rate = _get_growth(region) / 100

    # Build (price, weight) pairs — time-adjust each sale to today's value
    pairs = []
    for s in sales:
        price = s.get("price_gbp", 0)
        if not price:
            continue
        try:
            sd = date.fromisoformat(s["date"][:10])
            months_ago = (today.year - sd.year) * 12 + (today.month - sd.month)
        except Exception:
            months_ago = 24

        # Inflate historic price to today using regional ONS growth rate
        today_price = int(price * ((1 + annual_rate) ** (months_ago / 12)))

        # Recency weight (even after inflation, recent sales are more reliable)
        w = 4 if months_ago <= 6 else 3 if months_ago <= 12 else 2 if months_ago <= 24 else 1
        # Same property type gets +50% weight
        if pt_norm and pt_norm in s.get("property_type", "").lower():
            w = max(w + 1, int(w * 1.5))
        pairs.append((today_price, w))

    if not pairs:
        return 0

    # Outlier trimming: remove bottom 10% and top 10% by price (min 4 sales)
    raw_prices = sorted(p for p, _ in pairs)
    if len(raw_prices) >= 6:
        trim = max(1, len(raw_prices) // 10)
        lo, hi = raw_prices[trim], raw_prices[-trim - 1]
        pairs = [(p, w) for p, w in pairs if lo <= p <= hi]

    # Expand to weighted list and take median
    weighted = sorted(p for p, w in pairs for _ in range(w))
    if not weighted:
        return 0
    mid = len(weighted) // 2
    comp_value = int((weighted[mid - 1] + weighted[mid]) / 2) if len(weighted) % 2 == 0 else weighted[mid]

    # --- Multi-anchor blending ---
    sqm_anchor = int(floor_area * psm) if floor_area >= 30 else 0
    has_sqm    = sqm_anchor > 40_000
    has_ukhpi  = ukhpi_district_avg > 40_000

    if has_sqm and has_ukhpi:
        # All three: 55% comps, 25% sqm, 20% UKHPI
        comp_value = int(comp_value * 0.55 + sqm_anchor * 0.25 + ukhpi_district_avg * 0.20)
    elif has_sqm:
        # 65% comps, 35% sqm (existing behaviour)
        comp_value = int(comp_value * 0.65 + sqm_anchor * 0.35)
    elif has_ukhpi:
        # 75% comps, 25% UKHPI sanity check
        comp_value = int(comp_value * 0.75 + ukhpi_district_avg * 0.25)

    return max(comp_value, 40_000)


def _voa_rent(region: str, bedrooms: int, prop_type: str = "", transport_sc: int = 5, crime_sc: int = 5) -> int:
    beds = max(1, min(bedrooms, 4))
    base = VOA_RENTS["default"][beds]
    for key in VOA_RENTS:
        if key != "default" and key in region:
            base = VOA_RENTS[key][beds]
            break
    # Property type multiplier
    pt = prop_type.lower()
    mult = next((v for k, v in PROP_TYPE_MULTIPLIER.items() if k in pt), 1.0)
    # Tenant demand: better transport and lower crime push rents up slightly
    demand = 1.0 + (transport_sc - 5) * 0.012 + (crime_sc - 5) * 0.01
    demand = max(0.85, min(1.20, demand))
    return int(base * mult * demand)


def _rent_from_value(est_value: int, region: str) -> int:
    """Fallback rent derived from value × regional yield."""
    y = REGIONAL_YIELDS.get("default", 5.0)
    for key in REGIONAL_YIELDS:
        if key != "default" and key in region:
            y = REGIONAL_YIELDS[key]
            break
    return int(est_value * y / 100 / 12)


def _validate_financials(est_value: int, rent: int, region: str, sales: list):
    """Detect and auto-correct unrealistic values. Returns (value, rent, warnings)."""
    warnings = []
    # Value bounds for UK residential property
    if est_value < 40_000:
        fallback = _voa_rent(region, 3) * 12 * 18
        warnings.append(f"Estimated value £{est_value:,} below UK minimum — adjusted to £{fallback:,}")
        est_value = fallback
    elif est_value > 5_000_000:
        warnings.append(f"Estimated value £{est_value:,} capped at £5,000,000")
        est_value = 5_000_000
    # Yield sanity — UK gross yields rarely exceed 15%
    if est_value > 0:
        gross_yield = rent * 12 / est_value * 100
        if gross_yield > 15:
            rent = _rent_from_value(est_value, region)
            warnings.append(f"Yield {gross_yield:.1f}% unrealistic — rent recalculated from value")
        elif gross_yield < 1.0 and rent > 0:
            warnings.append(f"Yield {gross_yield:.1f}% very low — verify value data")
    return est_value, rent, warnings


def _get_growth(region: str) -> float:
    for key, rate in ONS_GROWTH.items():
        if key != "default" and key in region:
            return rate
    return ONS_GROWTH["default"]


def _calc_momentum(sales: list) -> float:
    if len(sales) < 4:
        return 0.038
    half = len(sales) // 2
    r = [s.get("price_gbp", 0) for s in sales[:half]]
    o = [s.get("price_gbp", 0) for s in sales[half:]]
    avg_r = sum(r) / len(r) if r else 0
    avg_o = sum(o) / len(o) if o else 1
    return (avg_r - avg_o) / avg_o if avg_o else 0.038


def _beds_from_floor_area(floor_area: float, prop_type: str = "") -> int:
    """
    Infer bedrooms from EPC floor area.
    Thresholds based on ONS English Housing Survey 2022 + NDSS minimum space standards.
    Key: 3-bed house min (NDSS) = 74sqm; avg UK 2-bed = 67sqm → boundary at 68sqm.
    """
    is_flat = "flat" in prop_type or "maisonette" in prop_type
    if is_flat:
        if floor_area < 42:  return 1
        if floor_area < 63:  return 2
        if floor_area < 88:  return 3
        return 4
    else:
        # 2-bed avg ~67sqm; 3-bed avg ~88sqm; NDSS min 3-bed = 74sqm
        # Use 68sqm as 2→3 boundary to avoid mis-classifying small 3-beds
        if floor_area < 52:   return 1   # studio or tiny 1-bed house
        if floor_area < 68:   return 2   # 2-bed (up to NDSS 3-bed minimum)
        if floor_area < 106:  return 3   # 3-bed (covers council builds ~74-95sqm)
        if floor_area < 140:  return 4
        return max(5, int(floor_area / 28))


# Threshold values for boundary-aware reconciliation
_HOUSE_THRESHOLDS = [52, 68, 106, 140]
_FLAT_THRESHOLDS  = [42, 63, 88]


def _infer_bedrooms(epc: dict) -> int:
    prop_type  = (epc.get("property-type") or epc.get("property_type") or "").lower()
    floor_area = _f(epc.get("total-floor-area") or epc.get("floor_area_sqm"), 0.0)
    is_flat    = "flat" in prop_type or "maisonette" in prop_type

    # Signal A: EPC certified habitable room count
    beds_rooms = None
    rooms_raw  = epc.get("number-habitable-rooms") or epc.get("number_habitable_rooms")
    if rooms_raw:
        try:
            r = int(rooms_raw)
            # Flats = 1 reception; small houses ≤4 rooms = 1 reception; larger = 2 receptions
            receptions = 1 if (is_flat or r <= 4) else 2
            beds_rooms = max(1, r - receptions)
        except (ValueError, TypeError):
            pass

    # Signal B: floor area → bedroom count via ONS-calibrated thresholds
    beds_area = _beds_from_floor_area(floor_area, prop_type) if floor_area >= 30 else None

    if beds_rooms is None and beds_area is None:
        return 3  # UK median default
    if beds_rooms is None:
        return beds_area
    if beds_area is None:
        return beds_rooms

    # ── Reconcile two signals ──────────────────────────────────────────────────
    if beds_rooms == beds_area:
        return beds_rooms  # perfect agreement

    diff = abs(beds_rooms - beds_area)

    if is_flat:
        # Flats: habitable rooms is very reliable (exactly 1 reception)
        return beds_rooms if diff <= 1 else beds_area

    # Houses: floor area thresholds have uncertainty near boundaries (±8 sqm).
    # When floor_area sits close to a threshold, habitable rooms is the better signal
    # because it directly measures room count rather than inferring from total sqm.
    thresholds = _HOUSE_THRESHOLDS
    near_boundary = any(abs(floor_area - t) < 8 for t in thresholds)

    if diff == 1:
        # Small disagreement: prefer habitable rooms near a boundary
        # (floor area in the 'grey zone'), otherwise prefer floor area
        return beds_rooms if near_boundary else beds_area
    else:
        # Large disagreement (≥2): floor area is usually more reliable for houses;
        # habitable-room subtraction has the largest errors at extremes
        return beds_area


def _investment_score(g_yield, crime_sc, transport, flood, sales) -> int:
    score = 40
    if g_yield >= 8:   score += 25
    elif g_yield >= 6: score += 18
    elif g_yield >= 4: score += 10
    score += crime_sc * 2
    score += transport
    if flood == "Low":   score += 5
    elif flood == "High": score -= 10
    if len(sales) >= 5:  score += 5
    return max(0, min(100, score))


def _risk_score(flood, crime_total, demo) -> int:
    score = 30
    if flood == "High":   score += 30
    elif flood == "Medium": score += 15
    if crime_total > 200:  score += 25
    elif crime_total > 80:  score += 15
    elif crime_total > 30:  score += 8
    imd = demo.get("imd_decile") or 5
    if imd <= 2:   score += 15
    elif imd <= 4: score += 8
    return max(0, min(100, score))


def _liquidity_score(sales: list) -> int:
    count = len(sales)
    if count >= 15: return 85
    if count >= 8:  return 70
    if count >= 4:  return 55
    if count >= 2:  return 38
    return 18


def _deal_score_calc(sales: list) -> int:
    if len(sales) < 3: return 20
    prices = [s.get("price_gbp", 0) for s in sales if s.get("price_gbp")]
    if not prices: return 20
    median = statistics.median(prices)
    discount = (median - min(prices)) / median * 100 if median else 0
    if discount >= 25: return 90
    if discount >= 15: return 72
    if discount >= 8:  return 55
    return 28


def _rental_demand_score(region, transport, crime_sc) -> int:
    score = 50
    if any(r in region for r in ["london", "manchester", "birmingham", "leeds", "bristol"]): score += 20
    score += transport * 2
    score += crime_sc * 2
    return max(0, min(100, score))


def _street_score(crime_sc, liq_sc, transport) -> int:
    return max(0, min(100, int(crime_sc * 4 + liq_sc * 0.3 + transport * 3)))


def _crime_score(total: int) -> int:
    if total == 0: return 9
    if total < 20: return 7
    if total < 50: return 6
    if total < 100: return 4
    if total < 200: return 3
    return 1


def _recommend_strategy(g_yield, beds, floor_area, region) -> str:
    if g_yield >= 10 and beds >= 4: return "HMO"
    if any(r in region for r in ["london", "oxford", "cambridge", "bath"]): return "SA"
    if g_yield >= 6: return "BTL"
    if g_yield < 4: return "Flip"
    return "BTL"


def _all_strategies(g_yield, beds, floor_area) -> list:
    s = ["BTL"]
    if beds >= 4: s.append("HMO")
    if g_yield < 5: s.append("Flip")
    if floor_area >= 120: s.append("BRRR")
    s.append("SA")
    return list(dict.fromkeys(s))[:4]


def _loft_viable(prop_type: str, epc: dict) -> bool:
    if "flat" in prop_type.lower(): return False
    roof = (epc.get("roof-description") or epc.get("roof_description") or "").lower()
    return "pitched" in roof or not roof


def _extension_viable(prop_type: str, epc: dict) -> bool:
    if "flat" in prop_type.lower(): return False
    form = (epc.get("built-form") or epc.get("built_form") or "").lower()
    return "mid-terrace" not in form


def _find_deals(sales: list) -> list:
    if len(sales) < 3: return []
    prices = [s.get("price_gbp", 0) for s in sales if s.get("price_gbp")]
    if not prices: return []
    median = statistics.median(prices)
    deals = []
    for s in sales:
        price = s.get("price_gbp", 0)
        if price and price < median * 0.88:
            disc = round((median - price) / median * 100, 1)
            deals.append({
                "address":                f"{s.get('address_paon','').strip()} {s.get('street','').strip()}".strip() or "Nearby property",
                "sold_price_gbp":         price,
                "area_median_gbp":        int(median),
                "discount_vs_median_pct": disc,
                "date":                   s.get("date", ""),
                "deal_type":              "Strong BMV" if disc >= 20 else "Below market value",
            })
    return sorted(deals, key=lambda x: x["discount_vs_median_pct"], reverse=True)[:3]


def _best_deal(sales: list) -> Optional[dict]:
    deals = _find_deals(sales)
    return deals[0] if deals else None


def _stamp_duty(price: int) -> int:
    if not price or price <= 250000: return 0
    if price <= 925000: return int((price - 250000) * 0.05)
    return int(675000 * 0.05 + (price - 925000) * 0.10)


def _haversine(lat1, lon1, lat2, lon2) -> int:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(2 * R * math.asin(math.sqrt(a)))


# ── Label helpers ─────────────────────────────────────────────────────────────

def _grade(s):
    if s >= 80: return "A"
    if s >= 65: return "B"
    if s >= 50: return "C"
    if s >= 35: return "D"
    return "F"

def _risk_label(s):
    if s >= 70: return "High"
    if s >= 50: return "Medium-High"
    if s >= 35: return "Medium"
    if s >= 20: return "Low-Medium"
    return "Low"

def _liq_label(s):
    if s >= 75: return "High"
    if s >= 50: return "Medium"
    if s >= 25: return "Low"
    return "Very Low"

def _demand_label(s):
    if s >= 75: return "Very High"
    if s >= 55: return "High"
    if s >= 35: return "Medium"
    return "Low"

def _deal_label(s):
    if s >= 80: return "Excellent BMV opportunities"
    if s >= 60: return "Good deal activity"
    if s >= 40: return "Some deals possible"
    return "Fair market pricing"

def _deal_recommendation(score, sales):
    if score >= 70 and sales:
        return "Below-market activity detected. Target properties 15-20% below median."
    if score >= 50:
        return "Some deal activity. Negotiate 8-12% below asking price."
    return "Fair market — limited discounting. Target auctions or motivated sellers."

def _desirability(inv, crime, transport):
    score = inv * 0.5 + crime * 5 + transport * 3
    if score >= 70: return "Prime"
    if score >= 55: return "Desirable"
    if score >= 40: return "Average"
    if score >= 25: return "Below average"
    return "Regeneration area"


def _area_desirability_score(crime_sc: int, trans_sc: int, demo_d: dict) -> int:
    imd = _i(demo_d.get("imd_decile"), 5)
    score = crime_sc * 7 + trans_sc * 3 + imd * 2
    return max(0, min(100, score))


def _growth_classification(region: str, growth_r: float, crime_tot: int, imd_decile) -> str:
    imd = imd_decile or 5
    if growth_r >= 5.0 and crime_tot < 120:
        return "strong_growth"
    if imd <= 2 or crime_tot > 250:
        return "declining"
    if growth_r < 3.0 and any(r in region for r in ["north east", "wales", "yorkshire", "west midlands"]):
        return "regeneration"
    if growth_r >= 3.5:
        return "stable"
    return "stable"


def _trajectory(region, growth):
    cls = _growth_classification(region, growth, 0, 5)
    return {"strong_growth": "Strong growth", "stable": "Stable",
            "regeneration": "Regeneration zone", "declining": "Declining"}.get(cls, "Stable")

def _income_est(region):
    if any(r in region for r in ["london", "south east", "east of england"]): return "Above average — median ~£48k"
    if any(r in region for r in ["north east", "wales", "yorkshire"]): return "Below average — median ~£28k"
    return "Average — median ~£35k"

def _transport_summary(transport):
    stations = transport.get("nearest_stations") or []
    score = transport.get("transport_score", 0)
    if not stations:
        return f"Transport score {score}/10. No stations within 800m."
    return f"Transport score {score}/10. Nearest: {stations[0]['name']} ({stations[0]['distance_m']}m)."

def _red_flags(flood, crime_total, risk_score):
    flags = []
    if flood == "High": flags.append("Active flood warnings in area")
    if crime_total > 150: flags.append("High crime rate — above national average")
    if risk_score >= 65: flags.append("High overall risk profile")
    return flags or ["No major red flags identified"]

def _suitable_for(risk_score):
    if risk_score < 25: return "Suitable for all investors including first-time landlords"
    if risk_score < 45: return "Suitable for experienced investors — moderate risk"
    if risk_score < 65: return "Experienced investors only"
    return "High risk — specialist investors only"

def _confidence_score(sales: list, epc: dict, demo_d: dict, crime_tot: int) -> dict:
    n = len(sales)
    if n >= 8:   val_label, val_sc = "high",   85
    elif n >= 3: val_label, val_sc = "medium",  65
    elif n >= 1: val_label, val_sc = "low",     40
    else:        val_label, val_sc = "low",     20
    if bool(epc) and bool(demo_d):
        rent_label, rent_sc = "high",   80
    elif bool(demo_d):
        rent_label, rent_sc = "medium", 60
    else:
        rent_label, rent_sc = "low",    40
    return {
        "valuation": val_label,
        "valuation_score": val_sc,
        "rent": rent_label,
        "rent_score": rent_sc,
        "overall": int((val_sc + rent_sc) / 2),
        "data_points": n,
    }


def _default_positives(inv_sc, g_yield, transport):
    p = []
    if g_yield >= 6: p.append(f"Strong gross yield of {g_yield:.1f}%")
    if transport >= 6: p.append("Good transport connectivity")
    if inv_sc >= 60: p.append("Above average investment score")
    if g_yield >= 4: p.append(f"Yield of {g_yield:.1f}% above savings rate")
    return p[:3] or ["Requires further due diligence"]

def _default_risks(risk_sc, flood, crime_total):
    r = []
    if flood not in ["Low", "Unknown"]: r.append(f"Flood risk: {flood}")
    if crime_total > 80: r.append("Above average crime rate")
    if risk_sc >= 50: r.append("Consider specialist insurance products")
    return r[:2] or ["Standard investment risks — conduct full due diligence"]

def _default_summary(postcode, inv_sc, strategy, value, yield_pct, val_5yr):
    grade = _grade(inv_sc)
    uplift = val_5yr - value
    return (
        f"The property at {postcode} scores {inv_sc}/100 (Grade {grade}), "
        f"with an estimated value of £{value:,} and gross yield of {yield_pct:.1f}%.\n\n"
        f"The recommended strategy is {strategy}. "
        f"The 5-year price forecast is £{val_5yr:,}, an uplift of £{uplift:,} at regional ONS growth rates.\n\n"
        f"Conduct standard due diligence including a structural survey and local authority search before proceeding."
    )

def _tenant_profiles(region, strategy):
    profiles = {
        "HMO": ["Young professionals", "Students"],
        "SA":  ["Business travellers", "Tourists"],
        "BTL": ["Families", "Young professionals"],
        "Flip": [],
        "BRRR": ["Families", "Long-term tenants"],
    }
    return profiles.get(strategy, ["Families", "Young professionals"])


# ── Safe helpers ──────────────────────────────────────────────────────────────

def _sr(result, default):
    if isinstance(result, Exception): return default
    return result if result is not None else default

def _s(val) -> dict:
    return val if isinstance(val, dict) else {}

def _i(val, default=0) -> int:
    try: return int(val) if val is not None else default
    except: return default

def _f(val, default=0.0) -> float:
    try: return float(val) if val is not None else default
    except: return default
