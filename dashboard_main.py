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
import csv
import functools
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import date, datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
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
HF_API_KEY        = os.getenv("HUGGINGFACE_API_KEY", "")
EPC_API_KEY       = os.getenv("EPC_API_KEY", "")
EPC_API_EMAIL     = os.getenv("EPC_API_EMAIL", "")
OS_PLACES_API_KEY = os.getenv("OS_PLACES_API_KEY", "")

# ── External API URLs ─────────────────────────────────────────────────────────
NOMINATIM  = "https://nominatim.openstreetmap.org/search"
POSTCODES  = "https://api.postcodes.io/postcodes"
HMLR       = "https://landregistry.data.gov.uk/landregistry/query"
POLICE_URL = "https://data.police.uk/api/crimes-street/all-crime"
EA_FLOOD   = "https://environment.data.gov.uk/flood-monitoring/id/floods"
OVERPASS   = "https://overpass-api.de/api/interpreter"
EPC_URL      = "https://epc.opendatacommunities.org/api/v1/domestic/search"
EPC_CERT_URL = "https://epc.opendatacommunities.org/api/v1/domestic/certificate"

# ── VOA 2025 median rents by region and bedroom count ────────────────────────
# Source: VOA Private Rental Market Statistics, England 2023-24; ONS Scotland/Wales 2024
VOA_RENTS = {
    "london":                   {1: 2000, 2: 2700, 3: 3300, 4: 4500},
    "south east":               {1: 1200, 2: 1550, 3: 1900, 4: 2500},
    "east of england":          {1: 1000, 2: 1250, 3: 1550, 4: 2000},
    "south west":               {1: 950,  2: 1200, 3: 1450, 4: 1850},
    "east midlands":            {1: 750,  2: 950,  3: 1100, 4: 1400},
    "west midlands":            {1: 800,  2: 1000, 3: 1150, 4: 1500},
    "north west":               {1: 775,  2: 975,  3: 1150, 4: 1450},
    "yorkshire and the humber": {1: 695,  2: 850,  3: 1000, 4: 1300},
    "north east":               {1: 600,  2: 795,  3: 900,  4: 1100},
    "wales":                    {1: 700,  2: 875,  3: 1000, 4: 1250},
    "scotland":                 {1: 900,  2: 1100, 3: 1350, 4: 1750},
    "northern ireland":         {1: 650,  2: 850,  3: 1000, 4: 1300},
    "default":                  {1: 800,  2: 1000, 3: 1200, 4: 1550},
}

# ── ONS UK HPI annual % growth by region (November 2024 release) ─────────────
ONS_GROWTH = {
    "london": 3.2, "south east": 3.9, "east of england": 3.5,
    "south west": 4.2, "east midlands": 5.3, "west midlands": 5.2,
    "north west": 5.4, "yorkshire and the humber": 4.8,
    "north east": 6.1, "wales": 4.5, "scotland": 5.6,
    "northern ireland": 6.9, "default": 4.5,
}

# ── Typical gross yields by region (BM Solutions/Rightmove 2024 survey) ──────
REGIONAL_YIELDS = {
    "london": 3.8, "south east": 4.2, "east of england": 4.5,
    "south west": 4.8, "east midlands": 5.8, "west midlands": 5.5,
    "north west": 6.2, "yorkshire and the humber": 5.8,
    "north east": 7.0, "wales": 5.5, "scotland": 6.0,
    "northern ireland": 6.5, "default": 5.5,
}

# ── Property type rent multipliers ───────────────────────────────────────────
PROP_TYPE_MULTIPLIER = {
    "detached": 1.15, "semi-detached": 1.04, "terraced": 0.97,
    "flat": 0.92, "maisonette": 0.93, "bungalow": 0.91,
}

# ── App init ──────────────────────────────────────────────────────────────────
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Atlas Property Intelligence",
    description="UK property analysis API — no database required",
    version="6.0.0",
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
    uprn: Optional[str] = None
    lmk_key: Optional[str] = None
    force_refresh: bool = False
    image: Optional[str] = None  # base64 or URL — future vision analysis hook

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
        input_uprn     = (body.get("uprn") or "").strip()
        input_lmk_key  = (body.get("lmk_key") or "").strip()

        if not input_location and not input_uprn and not input_lmk_key:
            raise HTTPException(status_code=422, detail="Provide 'address', 'postcode', 'uprn', or 'lmk_key'")

        if input_lmk_key and not input_location and not input_uprn:
            raise HTTPException(status_code=422, detail="Provide 'address' or 'postcode' alongside 'lmk_key' for geocoding")

        # Broad postcode (district/sector only, e.g. "NE15") — cannot analyse a specific property
        if input_location and _is_broad_postcode(input_location) and not input_uprn and not input_lmk_key:
            return JSONResponse(status_code=200, content={
                "requires_address_selection": True,
                "analysis_status": {
                    "exact_property_selected": False,
                    "requires_address_selection": True,
                    "data_quality": "low",
                    "warnings": ["Please enter a full postcode or exact property address."],
                },
                "address_options": [],
                "message": "Please enter a full postcode or exact property address.",
            })

        geocode_target = input_location or input_uprn
        coords = await _geocode(geocode_target)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": _tb.format_exc()})

    lat    = coords["latitude"]
    lng    = coords["longitude"]
    region = coords.get("region", "").lower()

    # Extract postcode — always preserve spaced format required by EPC/LR APIs
    _pc_match = re.search(r'[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}', input_location.upper())
    if _pc_match:
        _pc_raw = _pc_match.group(0).replace(" ", "")
        _pc_spaced = (_pc_raw[:-3] + " " + _pc_raw[-3:]) if len(_pc_raw) >= 5 else _pc_raw
    else:
        _pc_spaced = ""
    # User-supplied postcode always wins over Nominatim's nearest-postcode guess
    rpc = _pc_spaced or coords.get("postcode") or input_location.upper()[:8]

    # Extract street address — check both before and after the postcode
    _addr_hint = ""
    if _pc_match:
        _pc_pos = input_location.upper().find(_pc_match.group(0))
        _before_pc = input_location[:_pc_pos].strip().strip(',').strip()
        _after_pc  = input_location[_pc_pos + len(_pc_match.group(0)):].strip().strip(',').strip()
        _addr_hint = _before_pc or _after_pc

    # Resolve exact UPRN via OS Places if we have a house number but no UPRN yet
    if not input_uprn and _addr_hint and OS_PLACES_API_KEY:
        _uprn_list = await _fetch_uprns_by_postcode(rpc)
        if _uprn_list:
            _resolved = _resolve_uprn_from_hint(_uprn_list, _addr_hint)
            if _resolved:
                input_uprn = _resolved

    # Step 2: Fan out to all data sources concurrently
    fetched = await asyncio.gather(
        _fetch_sales(rpc),
        _fetch_epc(rpc, _addr_hint, uprn=input_uprn, lmk_key=input_lmk_key),
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
    _epc_matched_by = epc.get("_matched_by", "none") if epc else "none"

    # Determine how precisely we matched this property
    if input_lmk_key or _epc_matched_by == "lmk_key":
        _match_method        = "lmk_key"
        _property_confidence = "high"
    elif input_uprn:
        _match_method        = "uprn"
        _property_confidence = "high"
    elif _addr_hint and epc_list and len(epc_list) <= 3:
        _match_method        = "address_string"
        _property_confidence = "medium"
    elif epc_list:
        _match_method        = "postcode"
        _property_confidence = "low"
    else:
        _match_method        = "none"
        _property_confidence = "low"

    # Multiple EPC records with no specific selector — ask the user to pick a property
    if len(epc_list) > 1 and not input_uprn and not input_lmk_key:
        address_options = [_epc_row_to_address_option(row, rpc) for row in epc_list]
        return JSONResponse(status_code=200, content={
            "requires_address_selection": True,
            "requires_selection": True,
            "address_options": address_options,
            "postcode": rpc,
            "message": "Multiple properties found at this postcode — please select the exact property to analyse.",
            "analysis_status": {
                "exact_property_selected": False,
                "requires_address_selection": True,
                "data_quality": "low",
                "warnings": ["Multiple properties found at this postcode — please select an exact property"],
            },
        })

    # Use postcodes.io region (proper name like "North East") over Nominatim's "England"
    region = demo_d.get("region", region).lower() if demo_d.get("region") else region

    # Step 3: Derive all values
    # If address-hint returned a targeted EPC (1 record = specific property), use it directly.
    # Otherwise majority-vote across all records for the postcode.
    if epc_list and _addr_hint and len(epc_list) <= 3:
        beds = _infer_bedrooms(epc)          # targeted result — trust single record
    elif epc_list:
        beds = _consensus_bedrooms(epc_list) # postcode-level — use majority vote
    else:
        beds = 3                             # no EPC data — UK median default
    floor_area  = _f(epc.get("total-floor-area") or epc.get("floor_area_sqm"), 0.0)
    prop_type   = epc.get("property-type") or epc.get("property_type") or "Residential"
    epc_rating  = epc.get("current-energy-rating") or epc.get("current_energy_rating")

    # Display bedrooms only when EPC explicitly states them — never invent
    _explicit_beds = epc.get("number-of-bedrooms") or epc.get("number_of_bedrooms") if epc else None
    if _explicit_beds is not None:
        try:
            displayed_beds = int(float(str(_explicit_beds)))
        except (ValueError, TypeError):
            displayed_beds = None
    else:
        displayed_beds = None
    displayed_baths = None  # EPC does not contain bathroom count

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
    hmo_room_r  = _hmo_room_rent(region)
    hmo_rent    = hmo_rooms * hmo_room_r if hmo_rooms > 0 else 0
    hmo_yield   = round(hmo_rent * 12 / est_value * 100, 2) if est_value and hmo_rent else 0.0

    stamp           = _stamp_duty(est_value, investor=True)
    purchase_costs  = _purchase_costs(est_value, region)
    mortgage_scens  = _mortgage_scenarios(est_value, rent, region)
    tax_anal        = _tax_analysis(rent, mortgage, est_value)
    ten_yr          = _ten_year_model(est_value, rent, growth_r, region)
    brrrr           = _brrrr_analysis(est_value, rent, region, floor_area)
    cgt             = _cgt_estimate(est_value, int(est_value / 1.25))

    # Step 4: Renovation Intelligence Engine
    reno_intel = _renovation_scenarios(
        est_value, rent, g_yield, beds, floor_area, prop_type,
        region, epc, mortgage, loft_ok, ext_ok, hmo_rooms, hmo_rent,
        planning_d.get("hmo_pd_blocked", False),
    )

    # Step 5: AI analysis
    ai = await _run_ai(
        rpc, est_value, rent, g_yield, inv_sc,
        strategy, crime_tot, flood_lv, region, beds,
        trans_sc, epc_rating, floor_area, prop_type,
    )

    # Step 6: Investor intelligence engines
    deal_bd    = _deal_breakdown(est_value, g_yield, risk_sc, inv_sc, flood_lv,
                                  crime_tot, sales, trans_sc, imd_decile, growth_r,
                                  cashflow, ukhpi_d, region, beds, liq_sc)
    inv_dec    = _investor_decision(est_value, g_yield, rent, inv_sc, risk_sc,
                                    liq_sc, region, beds, floor_area, sales,
                                    growth_r, cashflow, strategy)
    area_rank  = _rank_nearby_postcodes(rpc, region, growth_r, trans_sc)
    exit_strat = _exit_strategy_engine(est_value, growth_r, liq_sc, risk_sc,
                                        g_yield, rent, beds, floor_area, strategy,
                                        region, inv_sc, cashflow)
    enh_conf   = _enhanced_confidence(sales, epc, demo_d, crime_tot,
                                       ukhpi_d, flood_d, trans_d)
    dq_panel   = _build_data_quality(
                    epc, epc_list, sales, crime_tot, demo_d, flood_d, trans_d,
                    planning_d, schools_d, ukhpi_price,
                    _epc_matched_by, _property_confidence, enh_conf)
    inv_verd   = _investor_verdict(
                    g_yield, risk_sc, liq_sc, rd_sc, inv_sc,
                    cashflow, rent, strategy, growth_r,
                    _property_confidence, enh_conf, inv_dec, flood_lv)

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

    # Build enhanced widgets with fallback hierarchy
    _yield_comp = _yield_benchmark_widget(region, rent, est_value, sales, ukhpi_d)
    _mkt_trends = _market_trends_widget(region, growth_r, ukhpi_d, sales, liq_sc, postcode=rpc)
    _mkt_trends["district"] = demo_d.get("admin_district", "")
    _school_rat = _school_rating_widget(schools_d)
    _deal_scan  = _deal_scanner_widget(rpc, region, g_yield, est_value, rent, rd_sc, liq_sc, growth_r, sales)

    # Append fallback benchmark sources to data quality panel
    if _yield_comp["level_used"] != "postcode":
        dq_panel["sources"].append({
            "name": "Yield benchmark",
            "status": "fallback",
            "matched_by": _yield_comp["level_used"],
            "freshness": "modelled",
            "note": _yield_comp["reason"],
        })
    if _mkt_trends["level_used"] != "district":
        dq_panel["sources"].append({
            "name": "Market trends",
            "status": "fallback",
            "matched_by": _mkt_trends["level_used"],
            "freshness": "modelled",
            "note": _mkt_trends["reason"],
        })

    return {
        "postcode": rpc,
        "display_address": (f"{_addr_hint.title()}, {rpc}" if _addr_hint else rpc),
        "latitude": lat,
        "longitude": lng,
        "generated_at": datetime.utcnow().isoformat(),

        "property": {
            "uprn": input_uprn or None,
            "lmk_key": epc.get("lmk-key") or epc.get("lmk_key") or input_lmk_key or None,
            "match_method": _match_method,
            "exact_property_selected": _property_confidence == "high",
            "bedrooms": displayed_beds,
            "bathrooms": displayed_baths,
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
            "estimated_value":          est_value,
            "monthly_rent":             rent,
            "annual_rent":              rent * 12,
            "rental_yield":             g_yield,
            "net_yield":                net_yield,
            "monthly_cashflow":         cashflow,
            "annual_profit":            annual_p,
            "monthly_mortgage_estimate": mortgage,
            "deposit_required":         deposit,
            "stamp_duty_estimate":      stamp,
            "total_acquisition_cost":   purchase_costs["total_funds_needed_25pct"],
            "purchase_costs_breakdown": purchase_costs,
            "mortgage_scenarios":       mortgage_scens,
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
            "deal_verdict": _deal_label(deal_sc),
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
            "is_projection": True,
            "projection_note": "Projections apply a constant historical regional growth rate and do not account for market cycles, interest rate changes, or local supply shocks. Treat as illustrative only.",
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
            "area_type": _area_type(region, trans_sc, crime_tot),
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
            "light":  {
                "cost": reno_intel["renovation_scenarios"][1]["estimated_cost"],
                "arv":  reno_intel["renovation_scenarios"][1]["new_property_value"],
                "roi_pct": reno_intel["renovation_scenarios"][1]["roi_pct"],
                "works": reno_intel["renovation_scenarios"][1]["works"],
            },
            "medium": {
                "cost": reno_intel["renovation_scenarios"][2]["estimated_cost"],
                "arv":  reno_intel["renovation_scenarios"][2]["new_property_value"],
                "roi_pct": reno_intel["renovation_scenarios"][2]["roi_pct"],
                "works": reno_intel["renovation_scenarios"][2]["works"],
            },
            "heavy": {
                "cost": reno_intel["renovation_scenarios"][4]["estimated_cost"],
                "arv":  reno_intel["renovation_scenarios"][4]["new_property_value"],
                "roi_pct": reno_intel["renovation_scenarios"][4]["roi_pct"],
                "works": reno_intel["renovation_scenarios"][4]["works"],
            },
            "epc_upgrade_cost": 8000,
            "epc_upgrade_notes": "Loft insulation, cavity wall fill, and heating upgrade to reach EPC C",
        },

        "renovation_scenarios": reno_intel["renovation_scenarios"],
        "best_scenario":        reno_intel["best_scenario"],
        "renovation_summary":   reno_intel["summary"],

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
            "feasibility": "Blocked — Article 4" if planning_d.get("hmo_pd_blocked") else ("High" if hmo_rooms >= 4 else "Medium" if hmo_rooms > 0 else "Low"),
            "room_potential":            hmo_rooms,
            "room_rent_estimate":        hmo_room_r,
            "article_4_restriction":     planning_d.get("hmo_pd_blocked", False),
            "hmo_notes": (
                "Article 4 direction active — full planning permission required before HMO conversion."
                if planning_d.get("hmo_pd_blocked")
                else (f"Convert to {hmo_rooms}-room HMO. Mandatory licence required for 5+ occupants. Check council's additional licensing scheme for smaller HMOs." if hmo_rooms > 0 else "Property likely too small for HMO — typically need 4+ bedrooms.")
            ),
            "estimated_monthly_hmo_rent": hmo_rent,
            "hmo_gross_yield":           hmo_yield,
            "hmo_cashflow":              max(0, hmo_rent - mortgage - int(hmo_rent * 0.15)),
            "conversion_cost_estimate":  hmo_rooms * 4500,
            "hmo_vs_btl_extra_monthly":  max(0, hmo_rent - rent),
        },

        "tax_analysis":   tax_anal,
        "brrrr_analysis": brrrr,
        "exit_analysis":  cgt,
        "ten_year_model": ten_yr,

        "confidence": enh_conf["confidence"],

        "data_validation": {
            "warnings": _val_warnings,
            "yield_realistic": 1.5 <= g_yield <= 15,
            "value_realistic": 40_000 <= est_value <= 5_000_000,
        },

        "data_freshness": {
            "price":        f"Latest sale: {sales[0].get('date','Unknown')[:7]}" if sales else "No sales data found",
            "rent":         "VOA Private Rental Market Statistics 2023-24",
            "growth":       "ONS UK HPI November 2024",
            "crime":        "Police API — rolling 12 months",
            "tax_rates":    "HMRC SDLT + HMRC CGT rates 2024-25",
            "last_updated": datetime.utcnow().isoformat(),
        },

        "data_age": enh_conf["data_age"],

        "data_quality": dq_panel,

        "deal_breakdown":    deal_bd,
        "investor_decision": inv_dec,
        "investor_verdict":  inv_verd,
        "area_ranking":      {
            "nearby_areas": area_rank,
            "source": "Derived from ONS regional data — no additional API calls",
        },
        "exit_strategy":     exit_strat,

        "data_sources": {
            "land_registry": len(sales) > 0,
            "epc": bool(epc),
            "epc_details": {
                "available": bool(epc),
                "source": "EPC Open Data Communities API",
                "matched_by": _epc_matched_by,
                "last_updated": epc.get("inspection-date") or epc.get("lodgement-date") if epc else None,
            },
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

        # ── Widget-guarantee keys — always present, never undefined ──────────────
        "yield_comparison": _yield_comp,

        "market_trends": _mkt_trends,

        "school_rating": _school_rat,

        "deal_scanner": _deal_scan,

        "floorplan_analysis": {
            "available": False,
            "reason": "No floorplan uploaded for this property.",
            "next_step": "Upload a clear floorplan image or PDF when vision parsing is enabled.",
        },

        "planning_data": {
            "available": planning_d.get("risk_level") is not None,
            "applications": [],
            "article_4_risk": (
                "high" if planning_d.get("article_4_active")
                else "medium" if planning_d.get("conservation_area") or planning_d.get("listed_building")
                else "low"
            ),
            "permitted_development_note": (
                "Article 4 direction active — permitted development rights are removed for HMO conversions in this area. Full planning permission required."
                if planning_d.get("article_4_active")
                else "Conservation area restrictions apply — check with local planning authority before external alterations."
                if planning_d.get("conservation_area")
                else "Listed building — permitted development rights are significantly restricted. Contact local planning authority."
                if planning_d.get("listed_building")
                else "No Article 4 direction detected. Standard permitted development rights likely apply, subject to local policy."
            ),
            "risk_level": planning_d.get("risk_level", "unknown"),
            "article_4_active": planning_d.get("article_4_active", False),
            "conservation_area": planning_d.get("conservation_area", False),
            "listed_building": planning_d.get("listed_building", False),
            "hmo_pd_blocked": planning_d.get("hmo_pd_blocked", False),
            "permitted_development_likely": planning_d.get("permitted_development_likely", True),
            "source": "GOV.UK Planning Data API",
            "reason": (
                None if planning_d.get("risk_level") is not None
                else "Planning data unavailable from GOV.UK Planning Data API for this location. Check with your local council's planning portal for Article 4 directions, conservation areas, and development constraints."
            ),
        },

        "analysis_status": {
            "exact_property_selected": _property_confidence == "high",
            "match_method": _match_method,
            "property_confidence": _property_confidence,
            "requires_address_selection": False,
            "data_quality": (
                "high" if (len(sales) >= 5 and bool(epc)) else
                "medium" if (len(sales) >= 2 or bool(epc)) else
                "low"
            ),
            "warnings": (
                (["No EPC data found — bedrooms and floor area are not confirmed"] if not epc_list else []) +
                (["No Land Registry sales found — valuation based on regional averages only"] if not sales else []) +
                (["Broad postcode used — property details may cover multiple properties"] if (epc_list and len(epc_list) > 5 and not _addr_hint and not input_uprn and not input_lmk_key) else [])
            ),
        },

        "disclaimer": {
            "not_financial_advice": True,
            "analysis_limitations": [
                "Estimated property value is a statistical model based on comparable sales data — not a RICS formal valuation.",
                "Rental figures are derived from VOA regional median data and may not reflect the specific property's achievable rent.",
                "Growth projections use historical ONS regional HPI data; past performance is not a reliable indicator of future results.",
                "Cashflow and mortgage estimates use assumed interest rates and costs — actual figures will vary by lender and circumstances.",
                "Crime and flood data are sourced from public APIs and may not reflect the most recent incidents or updated flood modelling.",
                "EPC data is sourced from government certificates which may be outdated or absent for some properties.",
                "AI-generated analysis is produced algorithmically and should not replace advice from a qualified financial or property professional.",
                "HMO, renovation, and development ROI figures are indicative estimates based on regional averages — site-specific surveys are required.",
                "Tax figures (SDLT, CGT, income tax) are calculated using published HMRC rates; always consult a tax adviser before transacting.",
            ],
            "confidence_explanation": (
                f"This analysis draws on {enh_conf['confidence_levels'].get('overall_data_quality', 'partial')} data coverage. "
                f"{'Land Registry comparable sales were found, supporting the valuation estimate.' if len(sales) > 0 else 'No recent Land Registry sales were found — the valuation is based on regional averages only.'} "
                f"{'EPC data is available for this postcode.' if bool(epc) else 'No EPC record was found — property details (bedrooms, floor area) are estimated from regional defaults.'} "
                "Scores (investment, risk, deal) are relative indicators within the Atlas model and are not absolute ratings. "
                "Higher confidence requires more live data sources returning results."
            ),
            "regulatory_notice": (
                "Atlas Property Intelligence is an information service, not a regulated financial adviser. "
                "Property investment carries risk including potential loss of capital. "
                "You should seek independent financial, legal, and surveying advice before making any investment decision."
            ),
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
# DEAL COMPARISON (structure — call /analyse-property per postcode then compare)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/compare-properties")
async def compare_properties(request: Request):
    """
    Deal comparison scaffold. Pass up to 5 postcodes; call /analyse-property
    for each, then use the comparison_fields list to build a side-by-side view.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="JSON body required")

    postcodes = body.get("postcodes", [])
    if not postcodes:
        raise HTTPException(status_code=422, detail="Provide 'postcodes' list (1–5 items)")
    if len(postcodes) > 5:
        raise HTTPException(status_code=422, detail="Maximum 5 postcodes per comparison")

    return {
        "comparison_id":   datetime.utcnow().isoformat(),
        "postcodes":       [p.upper().strip() for p in postcodes],
        "instructions":    "Call POST /analyse-property for each postcode, then compare the fields below.",
        "comparison_fields": [
            "financials.estimated_value",
            "financials.rental_yield",
            "financials.monthly_cashflow",
            "financials.net_yield",
            "scores.investment_score",
            "scores.risk_score",
            "scores.deal_score",
            "growth.annual_growth_rate_pct",
            "growth.five_year_projection",
            "investor_decision.recommended_offer_price",
            "investor_decision.suggested_hold_period",
            "deal_breakdown.why_it_works",
            "deal_breakdown.why_it_fails",
            "deal_breakdown.hidden_risks",
            "exit_strategy.best_exit_strategy",
            "exit_strategy.expected_exit_value",
            "exit_strategy.exit_timeline",
            "confidence.overall",
        ],
        "max_properties": 5,
    }


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
        "version": "6.1.0",
        "database": "none — stateless deployment",
        "hf_configured": bool(HF_API_KEY),
        "epc_configured": bool(EPC_API_KEY),
        "os_places_configured": bool(OS_PLACES_API_KEY),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADDRESS SEARCH  (req. 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_broad_postcode(query: str) -> bool:
    """Return True when query is only a district (NE15) or sector (NE15 6) — not a full postcode."""
    q = query.strip().upper().replace(" ", "")
    # Full postcode e.g. NE156DL — 6 or 7 chars, ends digit+2letters
    if re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\d[A-Z]{2}$', q):
        return False
    # Sector e.g. NE156 ends with single digit (no trailing letters)
    if re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\d$', q):
        return True
    # District e.g. NE15 — letters+digits, no trailing digit
    if re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?$', q):
        return True
    return False


async def _os_places_search(query: str) -> list:
    """Search OS Places API; returns list of address dicts."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.os.uk/search/places/v1/find",
                params={"query": query, "maxresults": 10, "key": OS_PLACES_API_KEY},
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                out = []
                for r in results:
                    ga = r.get("GAZETTEER_ENTRY", {})
                    pc = ga.get("POSTCODE_LOCATOR", "") or ga.get("POSTCODE", "")
                    out.append({
                        "display_address": ga.get("FULL_ADDRESS") or ga.get("ADDRESS", query),
                        "postcode": pc,
                        "uprn": str(ga.get("UPRN", "")) or None,
                        "latitude": ga.get("LAT"),
                        "longitude": ga.get("LNG"),
                        "source": "os_places",
                        "confidence": "high",
                    })
                return out
    except Exception:
        pass
    return []


async def _fetch_uprns_by_postcode(postcode: str) -> list:
    """OS Places /postcode endpoint — returns every address+UPRN for a full postcode."""
    if not OS_PLACES_API_KEY:
        return []
    try:
        pc = postcode.strip().upper().replace(" ", "")
        pc_fmt = (pc[:-3] + " " + pc[-3:]) if len(pc) >= 5 else pc
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                "https://api.os.uk/search/places/v1/postcode",
                params={"postcode": pc_fmt, "dataset": "DPA", "maxresults": 100, "key": OS_PLACES_API_KEY},
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                out = []
                for r in resp.json().get("results", []):
                    dpa = r.get("DPA", {})
                    uprn = str(dpa.get("UPRN", ""))
                    if uprn:
                        out.append({
                            "uprn": uprn,
                            "address": dpa.get("ADDRESS", ""),
                            "building_number": (dpa.get("BUILDING_NUMBER") or dpa.get("SUB_BUILDING_NAME") or "").strip(),
                            "building_name": (dpa.get("BUILDING_NAME") or "").strip(),
                        })
                return out
    except Exception:
        pass
    return []


def _resolve_uprn_from_hint(uprn_list: list, hint: str) -> str:
    """Match an address hint (e.g. '42' or 'Flat 2') against OS Places results to get a UPRN."""
    if not uprn_list or not hint:
        return ""
    hint = hint.strip()
    num_m = re.match(r'^(\d+[A-Za-z]?)', hint)
    if num_m:
        target = num_m.group(1).upper()
        for entry in uprn_list:
            if (entry.get("building_number") or "").upper() == target:
                return entry["uprn"]
    hint_upper = hint.upper()
    for entry in uprn_list:
        if hint_upper in (entry.get("address") or "").upper():
            return entry["uprn"]
    return ""


def _epc_row_to_address_option(row: dict, postcode: str = "") -> dict:
    """Convert a raw EPC API row to a standardised address option for the frontend."""
    addr1 = row.get("address1") or ""
    addr2 = row.get("address2") or ""
    addr3 = row.get("address3") or ""
    pc    = row.get("postcode") or postcode or ""
    parts = [p for p in [addr1, addr2, addr3, pc] if p]
    display = ", ".join(parts) if parts else pc or "Unknown address"
    uprn = str(row.get("uprn") or "").strip() or None
    return {
        "display_address": display,
        "address1": addr1 or None,
        "address2": addr2 or None,
        "address3": addr3 or None,
        "postcode": pc,
        "uprn": uprn,
        "lmk_key": row.get("lmk-key") or row.get("lmk_key") or None,
        "property_type": row.get("property-type") or row.get("property_type"),
        "floor_area_sqm": _f(row.get("total-floor-area") or row.get("floor_area_sqm"), None),
        "epc_rating": row.get("current-energy-rating") or row.get("current_energy_rating"),
        "inspection_date": row.get("inspection-date") or row.get("inspection_date") or None,
        "lodgement_date": row.get("lodgement-date") or row.get("lodgement_date") or None,
        "source": "EPC",
        "confidence": "high" if uprn else "medium",
    }


@app.get("/address-search")
async def address_search(query: str = Query(..., description="Postcode, partial address, or full address")):
    """Return selectable exact property options for an address query."""
    q = query.strip()
    if not q:
        return {"query": q, "requires_selection": True, "results": [], "warning": "No query provided."}

    # Broad postcode (district/area) — cannot identify a specific property
    if _is_broad_postcode(q):
        return {
            "query": q,
            "requires_selection": True,
            "results": [],
            "warning": "Please enter a full postcode or exact address.",
        }

    # Full UK postcode — use EPC as the primary address source
    if re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$', q.strip().upper()):
        epc_records = await _fetch_epc(q)
        if epc_records:
            options = [_epc_row_to_address_option(row, q.upper()) for row in epc_records]
            return {
                "query": q,
                "requires_selection": len(options) > 1,
                "address_options": options,
                "results": options,  # backwards-compat alias
                "source": "EPC",
                "warning": None,
            }

    # Non-postcode or EPC returned nothing — try OS Places then Nominatim
    results = []
    if OS_PLACES_API_KEY:
        results = await _os_places_search(q)

    if not results:
        try:
            geo = await _geocode(q)
            pc = geo.get("postcode") or q
            results = [{
                "display_address": geo.get("display_name", q),
                "postcode": pc,
                "uprn": None,
                "latitude": geo.get("latitude"),
                "longitude": geo.get("longitude"),
                "source": "fallback",
                "confidence": "low",
            }]
        except Exception:
            results = []

    return {
        "query": q,
        "requires_selection": len(results) != 1,
        "address_options": results,
        "results": results,  # backwards-compat alias
        "warning": (None if results
                    else "No results found. Please try a different address or full postcode."),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOORPLAN UPLOAD  (req. 7)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyse-floorplan")
async def analyse_floorplan(file: UploadFile = File(...)):
    """Accept a floorplan PDF/image. Room extraction is not yet enabled — returns a beta stub."""
    _allowed = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type and file.content_type not in _allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Send a JPEG, PNG, WEBP, or PDF.",
        )
    # Drain the upload so the client connection closes cleanly; discard bytes.
    try:
        await file.read()
    except Exception:
        pass
    return {
        "available": False,
        "reason": "Floorplan uploaded, but advanced room extraction is not enabled yet.",
        "next_step": "Upload clear floorplan image/PDF when vision parsing is enabled.",
        "filename": file.filename,
    }


@app.get("/debug/epc")
async def debug_epc(postcode: str):
    """Compact EPC debug — easy to read and share."""
    records = await _fetch_epc(postcode)
    best = _best_epc(records)
    return {
        "n": len(records),
        "beds_direct": best.get("number-of-bedrooms"),
        "hab_rooms": best.get("number-habitable-rooms"),
        "floor_area": best.get("total-floor-area"),
        "type": best.get("property-type"),
        "date": best.get("lodgement-date"),
        "addr": (best.get("address1") or "") + " " + (best.get("address2") or ""),
        "inferred_best": _infer_bedrooms(best),
        "inferred_consensus": _consensus_bedrooms(records),
        "all": [{"a": r.get("address1"), "beds": r.get("number-of-bedrooms"), "hab": r.get("number-habitable-rooms"), "sqm": r.get("total-floor-area"), "dt": r.get("lodgement-date")} for r in records],
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


async def _fetch_epc(postcode: str, address_hint: str = "", uprn: str = "", lmk_key: str = "") -> list:
    if not EPC_API_KEY:
        return []
    try:
        import base64
        creds = base64.b64encode(f"{EPC_API_EMAIL}:{EPC_API_KEY}".encode()).decode()
        headers = {"Accept": "application/json", "Authorization": f"Basic {creds}"}

        async with httpx.AsyncClient(timeout=10) as client:
            # 0. LMK-key lookup — exact certificate, highest precision
            if lmk_key:
                r = await client.get(f"{EPC_CERT_URL}/{lmk_key}", headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    rows = data.get("rows", [])
                    if not rows and data.get("lmk-key"):
                        rows = [data]
                    if rows:
                        for row in rows:
                            row["_matched_by"] = "lmk_key"
                        return rows

            # 1. UPRN-first lookup — most precise, exact property match
            if uprn:
                r = await client.get(EPC_URL, params={"uprn": uprn, "size": 5}, headers=headers)
                if r.status_code == 200:
                    rows = r.json().get("rows", [])
                    if rows:
                        for row in rows:
                            row["_matched_by"] = "uprn"
                        return rows

            # EPC API requires spaced format: "NE156DL" → "NE15 6DL"
            pc = postcode.strip().upper().replace(" ", "")
            pc_fmt = (pc[:-3] + " " + pc[-3:]) if len(pc) >= 5 else postcode.strip().upper()
            params: dict = {"postcode": pc_fmt, "size": 25}

            # 2. Address-targeted lookup — number extracted from hint
            if address_hint:
                num_m = re.match(r'^(\d+[A-Za-z]?)', address_hint.strip())
                if num_m:
                    params["address"] = num_m.group(1)

            resp = await client.get(EPC_URL, params=params, headers=headers)
            if resp.status_code == 200:
                rows = resp.json().get("rows", [])
                if rows:
                    matched_by = "postcode+address" if "address" in params else "postcode"
                    for row in rows:
                        row["_matched_by"] = matched_by
                    return rows
                # Fall back to postcode-only if address filter returned nothing
                if "address" in params:
                    params.pop("address")
                    resp2 = await client.get(EPC_URL, params=params, headers=headers)
                    if resp2.status_code == 200:
                        rows2 = resp2.json().get("rows", [])
                        for row in rows2:
                            row["_matched_by"] = "postcode"
                        return rows2
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
    """Pick the most informative recent EPC record from a list.

    Prefer records that have a direct bedroom count; within that preference
    pick the most recently lodged certificate.
    """
    if not epc_list:
        return {}

    def _epc_date(e):
        return e.get("lodgement-date") or e.get("lodgement_date") or e.get("inspection-date") or "1900-01-01"

    def _sort_key(e):
        has_beds = 1 if (e.get("number-of-bedrooms") or e.get("number_of_bedrooms")) is not None else 0
        return (has_beds, _epc_date(e))

    return max(epc_list, key=_sort_key)


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
    # Regional price-per-sqm benchmarks (2025, £/sqm median, ONS/Zoopla)
    _PSM = {
        "london": 8000, "south east": 4500, "east of england": 3600,
        "south west": 3300, "east midlands": 2500, "west midlands": 2600,
        "north west": 2400, "yorkshire and the humber": 2100,
        "north east": 1900, "wales": 2100, "scotland": 2300,
        "northern ireland": 1950, "default": 2700,
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


def _consensus_bedrooms(epc_list: list) -> int:
    """Majority-vote bedroom count across all EPC records for a postcode.

    More robust than picking one record because a UK postcode covers several
    properties and individual records may have data-quality issues.
    """
    if not epc_list:
        return 3
    counts = [_infer_bedrooms(r) for r in epc_list]
    return Counter(counts).most_common(1)[0][0]


def _infer_bedrooms(epc: dict) -> int:
    prop_type  = (epc.get("property-type") or epc.get("property_type") or "").lower()
    floor_area = _f(epc.get("total-floor-area") or epc.get("floor_area_sqm"), 0.0)
    is_flat    = "flat" in prop_type or "maisonette" in prop_type

    # Signal 0: direct EPC bedroom count — most authoritative field, use it outright
    direct_raw = epc.get("number-of-bedrooms") or epc.get("number_of_bedrooms")
    if direct_raw is not None:
        try:
            b = int(direct_raw)
            if 1 <= b <= 10:
                return b
        except (ValueError, TypeError):
            pass

    # Signal A: EPC certified habitable room count (includes reception rooms)
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
    # Prefer habitable rooms near a boundary; floor area otherwise.
    thresholds = _HOUSE_THRESHOLDS
    near_boundary = any(abs(floor_area - t) < 8 for t in thresholds)

    if diff == 1:
        return beds_rooms if near_boundary else beds_area
    else:
        # Large disagreement (≥2): floor area is more reliable for houses
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
    sa_markets = ["london", "oxford", "cambridge", "bath", "edinburgh", "york",
                  "chester", "brighton", "bristol", "manchester", "liverpool"]
    if beds >= 4 and g_yield >= 8:           return "HMO"
    if any(r in region for r in sa_markets): return "SA"
    if g_yield >= 6:                         return "BTL"
    if floor_area >= 80 and g_yield < 5:     return "BRRRR"
    if g_yield < 4:                          return "Flip"
    return "BTL"


def _all_strategies(g_yield, beds, floor_area) -> list:
    s = []
    if beds >= 3 and g_yield >= 6:   s.append("BTL")
    if beds >= 4:                    s.append("HMO")
    if floor_area >= 70:             s.append("BRRRR")
    if g_yield < 5:                  s.append("Flip")
    s.append("SA")
    s.append("BTL")
    return list(dict.fromkeys(s))[:5]


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


def _stamp_duty(price: int, investor: bool = True) -> int:
    """SDLT calculation. investor=True adds the 5% surcharge (raised Oct 2024)."""
    if not price:
        return 0
    sur = 0.05 if investor else 0.0
    # Bands from 1 April 2025
    brackets = [(125_000, 0.00), (125_000, 0.02), (675_000, 0.05), (575_000, 0.10), (float("inf"), 0.12)]
    tax, remaining = 0, price
    for band, rate in brackets:
        chunk = min(remaining, band)
        tax += int(chunk * (rate + sur))
        remaining -= chunk
        if remaining <= 0:
            break
    return tax


def _purchase_costs(price: int, region: str) -> dict:
    """Full acquisition cost breakdown for a BTL investor."""
    sdlt        = _stamp_duty(price, investor=True)
    legal       = 1800 if price < 300_000 else 2500
    survey      = 700 if price < 200_000 else 900 if price < 400_000 else 1200
    mortgage_fee = 1500
    broker_fee  = 500
    search_fees = 400
    total       = sdlt + legal + survey + mortgage_fee + broker_fee + search_fees
    deposit_25  = int(price * 0.25)
    deposit_20  = int(price * 0.20)
    return {
        "stamp_duty_sdlt":    sdlt,
        "legal_fees":         legal,
        "survey":             survey,
        "mortgage_arrangement": mortgage_fee,
        "broker_fee":         broker_fee,
        "search_fees":        search_fees,
        "total_transaction_costs": total,
        "total_funds_needed_25pct": deposit_25 + total,
        "total_funds_needed_20pct": deposit_20 + total,
        "note": "SDLT includes 5% additional-property surcharge (Oct 2024 rate)",
    }


# ── Regional HMO per-room rents (Spareroom 2024 data) ────────────────────────
HMO_ROOM_RENTS = {
    "london": 950, "south east": 650, "east of england": 580,
    "south west": 560, "east midlands": 480, "west midlands": 490,
    "north west": 510, "yorkshire and the humber": 460,
    "north east": 425, "wales": 440, "scotland": 520,
    "default": 500,
}


def _hmo_room_rent(region: str) -> int:
    for key, v in HMO_ROOM_RENTS.items():
        if key != "default" and key in region:
            return v
    return HMO_ROOM_RENTS["default"]


# ── Regional build cost multipliers (BCIS 2024 tender price index) ────────────
REGIONAL_BUILD_COST = {
    "london": 1.40, "south east": 1.20, "east of england": 1.10,
    "south west": 1.05, "east midlands": 0.95, "west midlands": 0.95,
    "north west": 0.90, "yorkshire and the humber": 0.88,
    "north east": 0.85, "wales": 0.88, "scotland": 0.92,
    "default": 1.0,
}


def _build_cost_multiplier(region: str) -> float:
    for key, v in REGIONAL_BUILD_COST.items():
        if key != "default" and key in region:
            return v
    return REGIONAL_BUILD_COST["default"]


def _mortgage_scenarios(value: int, monthly_rent: int, region: str) -> dict:
    """Return BTL mortgage cashflow for four deposit/rate combinations."""
    scenarios = {}
    for dep_pct, label in [(0.25, "25pct"), (0.20, "20pct")]:
        for rate, rlabel in [(0.045, "4.5pct"), (0.055, "5.5pct")]:
            loan      = int(value * (1 - dep_pct))
            monthly_i = int(loan * rate / 12)          # interest-only
            cashflow  = monthly_rent - monthly_i - int(monthly_rent * 0.10) - int(value * 0.005 / 12)
            gross_y   = round(monthly_rent * 12 / value * 100, 2) if value else 0
            scenarios[f"dep{label}_rate{rlabel}"] = {
                "deposit_pct":     int(dep_pct * 100),
                "deposit_gbp":     int(value * dep_pct),
                "loan_gbp":        loan,
                "rate_pct":        rate * 100,
                "monthly_interest": monthly_i,
                "monthly_cashflow": cashflow,
                "annual_cashflow":  cashflow * 12,
                "gross_yield_pct":  gross_y,
            }
    return scenarios


def _tax_analysis(monthly_rent: int, mortgage_interest: int, value: int) -> dict:
    """Section 24 impact for basic-rate and higher-rate taxpayers."""
    annual_rent    = monthly_rent * 12
    annual_mi      = mortgage_interest * 12
    expenses       = int(annual_rent * 0.15)   # mgmt + maintenance estimate

    def _calc(tax_rate):
        # Pre-S24 (old system for reference)
        pre_taxable  = max(0, annual_rent - annual_mi - expenses)
        pre_tax      = int(pre_taxable * tax_rate)
        pre_profit   = annual_rent - annual_mi - expenses - pre_tax

        # Post-S24 (current system)
        post_taxable = max(0, annual_rent - expenses)
        post_tax     = int(post_taxable * tax_rate) - int(annual_mi * 0.20)
        post_tax     = max(0, post_tax)
        post_profit  = annual_rent - annual_mi - expenses - post_tax
        return {
            "annual_profit_gbp":     post_profit,
            "annual_tax_gbp":        post_tax,
            "effective_tax_rate_pct": round(post_tax / annual_rent * 100, 1) if annual_rent else 0,
            "section24_annual_cost": max(0, post_tax - int(pre_taxable * tax_rate)),
        }

    return {
        "annual_rent":          annual_rent,
        "annual_mortgage_interest": annual_mi,
        "estimated_expenses":   expenses,
        "basic_rate_20pct":     _calc(0.20),
        "higher_rate_40pct":    _calc(0.40),
        "note": "Section 24 removes mortgage interest deduction; 20% tax credit applies instead.",
    }


def _ten_year_model(value: int, rent: int, growth_rate: float, region: str) -> list:
    """Year-by-year projection: value, equity (25% deposit), cashflow, total return."""
    deposit     = int(value * 0.25)
    loan        = value - deposit
    rate        = 0.050
    m_interest  = int(loan * rate / 12)
    rent_growth = 0.035   # ~3.5% pa rent inflation
    rows = []
    cum_cashflow = 0
    for yr in range(1, 11):
        proj_value  = int(value * ((1 + growth_rate / 100) ** yr))
        proj_rent   = int(rent * ((1 + rent_growth) ** yr))
        m_costs     = m_interest + int(proj_rent * 0.10) + int(proj_value * 0.005 / 12)
        m_cashflow  = proj_rent - m_costs
        cum_cashflow += m_cashflow * 12
        equity      = proj_value - loan
        total_return = (equity - deposit + cum_cashflow)
        roi_pct     = round(total_return / deposit * 100, 1) if deposit else 0
        rows.append({
            "year":             yr,
            "projected_value":  proj_value,
            "projected_rent":   proj_rent,
            "annual_cashflow":  m_cashflow * 12,
            "cumulative_cashflow": cum_cashflow,
            "equity":           equity,
            "total_return":     total_return,
            "roi_on_deposit_pct": roi_pct,
        })
    return rows


def _cgt_estimate(value: int, purchase_price: int, years_held: int = 5) -> dict:
    """Rough CGT estimate on exit for a higher-rate taxpayer."""
    gain          = max(0, value - purchase_price)
    allowance     = 3_000       # 2024-25 annual CGT allowance
    taxable_gain  = max(0, gain - allowance)
    cgt_higher    = int(taxable_gain * 0.24)   # 24% CGT on residential (post Apr 2024)
    cgt_basic     = int(taxable_gain * 0.18)
    net_higher    = value - purchase_price - cgt_higher
    return {
        "estimated_gain":       gain,
        "annual_cgt_allowance": allowance,
        "taxable_gain":         taxable_gain,
        "cgt_higher_rate_24pct": cgt_higher,
        "cgt_basic_rate_18pct":  cgt_basic,
        "net_proceeds_higher_rate": net_higher,
        "note": "CGT rates from April 2024. Consult a tax adviser for personal circumstances.",
    }


def _brrrr_analysis(value: int, rent: int, region: str, floor_area: float) -> dict:
    """BRRRR (Buy, Refurb, Rent, Refinance, Repeat) viability."""
    refurb_cost   = int(floor_area * 450) if floor_area >= 30 else 25_000
    arv           = int(value * 1.25)    # After-repair value (assume 25% uplift)
    refi_loan     = int(arv * 0.75)      # 75% LTV refinance
    cash_left_in  = max(0, (value + refurb_cost) - refi_loan)
    monthly_i     = int(refi_loan * 0.050 / 12)
    cashflow      = rent - monthly_i - int(rent * 0.10) - int(arv * 0.005 / 12)
    cocoR         = round(cashflow * 12 / cash_left_in * 100, 1) if cash_left_in > 0 else 0
    viable        = cash_left_in < int(value * 0.15) and cashflow > 0
    return {
        "viable":              viable,
        "purchase_price":      value,
        "estimated_refurb":    refurb_cost,
        "after_repair_value":  arv,
        "refinance_loan_75pct": refi_loan,
        "cash_left_in":        cash_left_in,
        "monthly_cashflow":    cashflow,
        "cash_on_cash_return_pct": cocoR,
        "verdict": (
            "Strong BRRRR — most capital recycled back out" if viable and cocoR > 8
            else "Viable BRRRR — some capital remains in deal" if viable
            else "BRRRR marginal — check purchase price and refurb costs"
        ),
    }


def _renovation_scenarios(
    est_value: int,
    rent: int,
    g_yield: float,
    beds: int,
    floor_area: float,
    prop_type: str,
    region: str,
    epc: dict,
    mortgage: int,
    loft_ok: bool,
    ext_ok: bool,
    hmo_rooms: int,
    hmo_rent: int,
    planning_blocked_hmo: bool = False,
) -> dict:
    """
    Renovation Intelligence Engine.
    Five scenarios with full financial breakdown using BCIS cost benchmarks,
    VOA rent tables and RICS capital uplift heuristics. No paid APIs required.
    """
    sqm = floor_area if floor_area >= 30 else 85.0
    mult = _build_cost_multiplier(region)
    is_flat = "flat" in prop_type.lower() or "maisonette" in prop_type.lower()
    epc_rating = (epc.get("current-energy-rating") or epc.get("current_energy_rating") or "D").upper()

    def _new_yield(new_value: int, new_rent_monthly: int) -> float:
        return round(new_rent_monthly * 12 / new_value * 100, 2) if new_value else g_yield

    def _roi(cost: int, val_uplift: int, rent_uplift_monthly: int) -> float:
        if cost <= 0:
            return 0.0
        return round((val_uplift + rent_uplift_monthly * 12) / cost * 100, 1)

    def _payback(cost: int, rent_uplift_monthly: int) -> Optional[int]:
        return round(cost / rent_uplift_monthly) if rent_uplift_monthly > 0 else None

    # ── 1. Baseline ────────────────────────────────────────────────────────────
    baseline = {
        "name": "baseline",
        "label": "No Changes",
        "description": "Hold as-is. Current yield and value with no additional investment.",
        "estimated_cost": 0,
        "new_property_value": est_value,
        "new_rent": rent,
        "value_increase": 0,
        "rent_increase": 0,
        "gross_yield_pct": g_yield,
        "yield_change_pct": 0.0,
        "roi_pct": 0.0,
        "payback_months": None,
        "works": [],
        "feasibility": "N/A",
        "recommended_for": "Low-risk hold strategy",
    }

    # ── 2. Light refurbishment ─────────────────────────────────────────────────
    # ~£350/sqm (capped at 100sqm) + regional multiplier
    light_cost = max(12_000, int(min(sqm, 100) * 350 * mult))
    # Poor EPC means more cosmetic gains when refreshed
    light_uplift_pct = 0.07 if epc_rating in ("E", "F", "G") else 0.05
    light_new_val = int(est_value * (1 + light_uplift_pct))
    light_rent_inc = int(rent * 0.06)
    light_new_rent = rent + light_rent_inc

    light = {
        "name": "light_refurb",
        "label": "Light Refurbishment",
        "description": "Cosmetic refresh to achieve top-of-market rent and improve saleability.",
        "estimated_cost": light_cost,
        "new_property_value": light_new_val,
        "new_rent": light_new_rent,
        "value_increase": light_new_val - est_value,
        "rent_increase": light_rent_inc,
        "gross_yield_pct": _new_yield(light_new_val, light_new_rent),
        "yield_change_pct": round(_new_yield(light_new_val, light_new_rent) - g_yield, 2),
        "roi_pct": _roi(light_cost, light_new_val - est_value, light_rent_inc),
        "payback_months": _payback(light_cost, light_rent_inc),
        "works": ["Budget kitchen refresh", "Bathroom refresh", "Full redecoration", "New flooring", "LED lighting upgrade"],
        "feasibility": "High",
        "recommended_for": "BTL landlords maximising rental income quickly",
    }

    # ── 3. Full refurbishment ──────────────────────────────────────────────────
    # ~£850/sqm full strip-out + regional multiplier
    full_cost = max(35_000, int(sqm * 850 * mult))
    full_uplift_pct = 0.22 if epc_rating in ("E", "F", "G") else 0.17
    full_new_val = int(est_value * (1 + full_uplift_pct))
    full_rent_inc = int(rent * 0.15)
    full_new_rent = rent + full_rent_inc

    full = {
        "name": "full_refurb",
        "label": "Full Refurbishment",
        "description": "Complete internal rebuild to substantially raise EPC rating, value and rent.",
        "estimated_cost": full_cost,
        "new_property_value": full_new_val,
        "new_rent": full_new_rent,
        "value_increase": full_new_val - est_value,
        "rent_increase": full_rent_inc,
        "gross_yield_pct": _new_yield(full_new_val, full_new_rent),
        "yield_change_pct": round(_new_yield(full_new_val, full_new_rent) - g_yield, 2),
        "roi_pct": _roi(full_cost, full_new_val - est_value, full_rent_inc),
        "payback_months": _payback(full_cost, full_rent_inc),
        "works": [
            "Full kitchen & bathroom replacement", "Complete rewire (18th edition)",
            "Boiler or heat pump replacement", "Cavity wall & loft insulation",
            "Double glazing (if absent)", "Structural repairs", "Full redecoration",
        ],
        "feasibility": "Medium",
        "recommended_for": "BRRRR investors — buy below market, refurb, refinance",
    }

    # ── 4. HMO conversion ─────────────────────────────────────────────────────
    hmo_viable = hmo_rooms > 0 and not planning_blocked_hmo
    hmo_conv_cost = int(hmo_rooms * 5_000 * mult) if hmo_rooms > 0 else 0
    hmo_val_uplift = int(hmo_conv_cost * 1.3) if hmo_viable else 0
    hmo_new_val = est_value + hmo_val_uplift
    hmo_rent_inc = max(0, hmo_rent - rent) if hmo_viable else 0
    hmo_new_rent_out = hmo_rent if hmo_viable else rent

    hmo_scenario = {
        "name": "hmo_conversion",
        "label": "HMO Conversion",
        "description": (
            f"Convert to {hmo_rooms}-room HMO for significantly higher rental yield."
            if hmo_viable else
            "Property size or Article 4 planning restrictions limit HMO viability."
        ),
        "estimated_cost": hmo_conv_cost,
        "new_property_value": hmo_new_val,
        "new_rent": hmo_new_rent_out,
        "value_increase": hmo_val_uplift,
        "rent_increase": hmo_rent_inc,
        "gross_yield_pct": _new_yield(hmo_new_val, hmo_new_rent_out),
        "yield_change_pct": round(_new_yield(hmo_new_val, hmo_new_rent_out) - g_yield, 2),
        "roi_pct": _roi(hmo_conv_cost, hmo_val_uplift, hmo_rent_inc) if hmo_conv_cost else 0.0,
        "payback_months": _payback(hmo_conv_cost, hmo_rent_inc) if hmo_conv_cost else None,
        "works": (
            [
                f"Fire doors to all {hmo_rooms} letting rooms",
                "Grade D fire alarm & emergency lighting",
                "Shared kitchen & bathroom upgrade",
                "HMO licence application (mandatory 5+ occupants)",
                "Article 4 / permitted development check",
            ] if hmo_viable else ["Not viable — property too small or blocked by Article 4 direction"]
        ),
        "feasibility": "High" if hmo_viable and beds >= 5 else "Medium" if hmo_viable else "Low",
        "recommended_for": "Yield-maximising investors in high-demand cities",
        "hmo_rooms": hmo_rooms,
        "article_4_blocked": planning_blocked_hmo,
    }

    # ── 5. Extension / Loft conversion ────────────────────────────────────────
    ext_cost = int(((45_000 if ext_ok else 0) + (38_000 if loft_ok else 0)) * mult)
    ext_sqm_add = (20 if ext_ok else 0) + (25 if loft_ok else 0)
    # Local value-per-sqm drives capital uplift; cap at £5k/sqm to avoid outliers
    local_psm = min(est_value / sqm, 5_000) if sqm > 0 else 3_000
    ext_val_inc = int(ext_sqm_add * local_psm) if ext_cost > 0 else 0
    ext_new_val = est_value + ext_val_inc
    ext_new_beds = beds + (1 if ext_ok else 0) + (1 if loft_ok else 0)
    # Rent uplift from extra bedroom via VOA table
    if ext_new_beds > beds and ext_cost > 0:
        voa_new = VOA_RENTS.get("default", {}).get(min(ext_new_beds, 4), rent)
        for k in VOA_RENTS:
            if k != "default" and k in region:
                voa_new = VOA_RENTS[k].get(min(ext_new_beds, 4), rent)
                break
        ext_rent_inc = max(0, voa_new - rent)
    else:
        ext_rent_inc = 0
    ext_new_rent = rent + ext_rent_inc
    ext_works = []
    if ext_ok:
        ext_works += ["Single-storey rear extension", "Structural engineer & architect fees", "Planning permission application"]
    if loft_ok:
        ext_works += ["Loft conversion (dormer or Velux)", "New staircase & bedroom fit-out", "Building regs sign-off"]
    if not ext_works:
        ext_works = ["Extension and loft conversion not viable for this property type"]

    ext_scenario = {
        "name": "extension_potential",
        "label": "Extension / Loft Conversion",
        "description": (
            f"Add ~{ext_sqm_add}sqm of floor area to create new bedroom(s) and maximise capital value."
            if ext_cost > 0 else
            "Structural development not viable for this property configuration."
        ),
        "estimated_cost": ext_cost,
        "new_property_value": ext_new_val,
        "new_rent": ext_new_rent,
        "value_increase": ext_val_inc,
        "rent_increase": ext_rent_inc,
        "gross_yield_pct": _new_yield(ext_new_val, ext_new_rent),
        "yield_change_pct": round(_new_yield(ext_new_val, ext_new_rent) - g_yield, 2),
        "roi_pct": _roi(ext_cost, ext_val_inc, ext_rent_inc) if ext_cost else 0.0,
        "payback_months": _payback(ext_cost, ext_rent_inc) if ext_cost else None,
        "works": ext_works,
        "feasibility": "High" if ext_ok and loft_ok else "Medium" if (ext_ok or loft_ok) else "Low",
        "recommended_for": "Long-term capital growth investors",
        "bedrooms_after": ext_new_beds,
        "sqm_added": ext_sqm_add,
    }

    # ── Best scenario detection ────────────────────────────────────────────────
    candidates = [light, full, hmo_scenario, ext_scenario]
    feasible = [s for s in candidates if s["feasibility"] in ("High", "Medium") and s["estimated_cost"] > 0]

    def _score(s: dict) -> float:
        # Weighted: ROI 50%, yield delta 35%, feasibility 15%
        roi_s   = min(s["roi_pct"] / 60.0, 1.0) * 50
        yield_s = min(max(s["yield_change_pct"], 0) / 3.0, 1.0) * 35
        feas_s  = 15 if s["feasibility"] == "High" else 7
        return roi_s + yield_s + feas_s

    if feasible:
        best = max(feasible, key=_score)
    else:
        best = baseline

    best_scenario = {
        "name": best["name"],
        "label": best["label"],
        "reasoning": _reno_reasoning(best, est_value, rent, g_yield),
        "estimated_cost": best["estimated_cost"],
        "expected_roi_pct": best["roi_pct"],
        "value_uplift": best["value_increase"],
        "rent_uplift": best["rent_increase"],
        "new_yield_pct": best["gross_yield_pct"],
        "payback_months": best["payback_months"],
        "investor_action": _investor_action(best["name"]),
    }

    # ── Value delta summary ────────────────────────────────────────────────────
    max_val_inc  = max(s["value_increase"] for s in candidates)
    max_rent_inc = max(s["rent_increase"] for s in candidates)
    best_yield   = max(s["gross_yield_pct"] for s in candidates)

    summary = {
        "current_value":            est_value,
        "current_rent":             rent,
        "current_yield_pct":        g_yield,
        "max_achievable_value":     est_value + max_val_inc,
        "max_achievable_rent":      rent + max_rent_inc,
        "max_achievable_yield_pct": best_yield,
        "value_delta":              max_val_inc,
        "rent_delta":               max_rent_inc,
        "yield_delta_pct":          round(best_yield - g_yield, 2),
        "scenarios_evaluated":      len(candidates),
        "best_option":              best["name"],
    }

    return {
        "renovation_scenarios": [baseline, light, full, hmo_scenario, ext_scenario],
        "best_scenario": best_scenario,
        "summary": summary,
    }


def _reno_reasoning(scenario: dict, est_value: int, rent: int, g_yield: float) -> str:
    name     = scenario["name"]
    cost     = scenario["estimated_cost"]
    roi      = scenario["roi_pct"]
    val_inc  = scenario["value_increase"]
    rent_inc = scenario["rent_increase"]
    if name == "light_refurb":
        return (
            f"Fastest payback at £{cost:,} with ~{roi:.0f}% total return. "
            f"Achieves top-of-market rent (+£{rent_inc}/mo) with 3–6 week programme."
        )
    if name == "full_refurb":
        return (
            f"Strongest value uplift (+£{val_inc:,}) on £{cost:,} spend. "
            f"Raises EPC rating and unlocks BRRRR refinance at 75% LTV."
        )
    if name == "hmo_conversion":
        return (
            f"Maximum yield play: +£{rent_inc}/mo rent on £{cost:,} conversion. "
            f"~{roi:.0f}% ROI — best for high-demand urban locations."
        )
    if name == "extension_potential":
        return (
            f"Adds bedrooms and floor area, lifting value by £{val_inc:,}. "
            f"{roi:.0f}% ROI on £{cost:,} — ideal long-term hold strategy."
        )
    return "No renovation materially improves returns. Hold as-is."


def _investor_action(best_name: str) -> str:
    actions = {
        "baseline":             "Hold and monitor. No renovation required to hit target returns.",
        "light_refurb":        "Proceed immediately. 3–6 week programme — get 3 quotes this week.",
        "full_refurb":         "Plan 8–12 week programme. Target BRRRR refinance post-completion.",
        "hmo_conversion":      "Check HMO licence requirements with local council before committing.",
        "extension_potential": "Commission architect drawings. Allow 8–12 weeks for planning approval.",
    }
    return actions.get(best_name, "Consult a specialist before committing capital.")


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

def _area_type(region: str, trans_sc: int, crime_tot: int) -> str:
    urban_keywords = ["london", "manchester", "birmingham", "leeds", "liverpool",
                      "bristol", "sheffield", "edinburgh", "glasgow", "newcastle",
                      "nottingham", "leicester", "coventry", "bradford", "cardiff"]
    suburban_keywords = ["surrey", "kent", "essex", "hertfordshire", "oxfordshire",
                         "cambridgeshire", "buckinghamshire", "berkshire", "cheshire"]
    if any(k in region for k in urban_keywords):
        return "Urban"
    if any(k in region for k in suburban_keywords) or trans_sc >= 6:
        return "Suburban"
    return "Rural"


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


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTOR INTELLIGENCE ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

def _deal_breakdown(
    est_value: int, g_yield: float, risk_sc: int, inv_sc: int, flood_lv: str,
    crime_tot: int, sales: list, trans_sc: int, imd_decile: int, growth_r: float,
    cashflow: int, ukhpi_d: dict, region: str, beds: int, liq_sc: int,
) -> dict:
    why_works, why_fails, hidden_risks = [], [], []
    trend_pct = ukhpi_d.get("trend_pct_6m", 0) or 0

    # Positive signals
    if g_yield >= 7:
        why_works.append(f"Exceptional gross yield of {g_yield:.1f}% — well above UK average")
    elif g_yield >= 5.5:
        why_works.append(f"Strong gross yield of {g_yield:.1f}% — above national average")
    elif g_yield >= 4:
        why_works.append(f"Yield of {g_yield:.1f}% covers mortgage at typical BTL rates")
    if cashflow > 200:
        why_works.append(f"Positive monthly cashflow of £{cashflow:,} after mortgage and costs")
    if growth_r >= 5:
        why_works.append(f"High-growth region — {growth_r:.1f}% annual ONS growth rate")
    if trans_sc >= 7:
        why_works.append("Excellent transport links — underpins strong tenant demand")
    elif trans_sc >= 5:
        why_works.append("Good transport connectivity supports rental demand")
    if liq_sc >= 70:
        why_works.append("Highly liquid market — clean exit available when needed")
    if trend_pct > 2:
        why_works.append(f"Local prices rising {trend_pct:.1f}% over the past 6 months")
    if imd_decile and imd_decile >= 7:
        why_works.append("Affluent area — lower void risk and better tenant quality")
    if crime_tot < 30:
        why_works.append("Very low crime — supports premium rents and minimal void periods")

    # Negative signals
    if g_yield < 4:
        why_fails.append(f"Low yield of {g_yield:.1f}% — cashflow likely negative at standard BTL rates")
    if cashflow < 0:
        why_fails.append(f"Negative cashflow of £{abs(cashflow):,}/mo — requires income top-up")
    elif cashflow < 100:
        why_fails.append("Thin cashflow margin — any rate rise or void makes deal loss-making")
    if flood_lv == "High":
        why_fails.append("Flood Zone 3 — elevated insurance costs; some mortgage lenders will decline")
    elif flood_lv == "Medium":
        why_fails.append("Flood Zone 2 — some insurers add flood premium")
    if crime_tot > 150:
        why_fails.append(f"{crime_tot} crimes nearby — above average; voids and arrears more likely")
    if liq_sc < 35:
        why_fails.append("Illiquid market — may take 6+ months to exit if needed")
    if trend_pct < -1:
        why_fails.append(f"Local prices falling {abs(trend_pct):.1f}% over past 6 months")
    if imd_decile and imd_decile <= 2:
        why_fails.append("Highly deprived area — elevated tenant arrears and void risk")
    if not sales:
        why_fails.append("No comparable sales data — estimated value has low confidence")

    # Hidden risks (non-obvious)
    if g_yield >= 10:
        hidden_risks.append("Yield above 10% often signals undisclosed issues — order a full structural survey")
    if beds >= 4 and g_yield >= 7:
        hidden_risks.append("High yield may reflect HMO-only demand; single-let rent will be materially lower")
    if trend_pct > 5:
        hidden_risks.append("Rapid recent price growth may be unsustainable — risk of buying at a cyclical peak")
    if imd_decile and imd_decile <= 3:
        hidden_risks.append("Section 24 impact amplified where rents are thin relative to mortgage interest")
    if any(k in region for k in ["london", "south east"]):
        hidden_risks.append("High entry price compresses yield — any rate rise significantly erodes cashflow")
    if not hidden_risks:
        hidden_risks.append("No atypical hidden risks detected — conduct standard due diligence (survey, LA search)")

    return {
        "why_it_works": why_works[:5],
        "why_it_fails": why_fails[:5],
        "hidden_risks":  hidden_risks[:3],
    }


def _investor_decision(
    est_value: int, g_yield: float, rent: int, inv_sc: int, risk_sc: int,
    liq_sc: int, region: str, beds: int, floor_area: float, sales: list,
    growth_r: float, cashflow: int, strategy: str,
) -> dict:
    # Offer discount — driven by yield attractiveness
    if g_yield < 4:
        discount = 0.88   # 12% below estimate
    elif g_yield < 5.5:
        discount = 0.92   # 8% below
    elif g_yield >= 7:
        discount = 0.97   # 3% — already priced well
    else:
        discount = 0.94   # 6% below

    recommended_offer = int(est_value * discount)
    offer_yield = round(rent * 12 / recommended_offer * 100, 1) if recommended_offer else 0
    discount_pct = round((1 - discount) * 100, 1)

    # Hold period recommendation
    if strategy == "Flip":
        hold = "6–12 months"
        hold_note = "Short-term capital gain or light renovation uplift"
    elif strategy == "BRRRR":
        hold = "12–18 months"
        hold_note = "Refurb, refinance at 75% LTV, redeploy capital into next deal"
    elif strategy == "SA":
        hold = "2–5 years"
        hold_note = "Serviced accommodation requires active management — medium term is optimal"
    elif g_yield >= 6.5 and cashflow > 200:
        hold = "10+ years"
        hold_note = "High yield + strong cashflow — long hold maximises total return and compounding"
    elif growth_r >= 5:
        hold = "5–7 years"
        hold_note = "Growth market — hold to capture appreciation across the price cycle"
    else:
        hold = "5–10 years"
        hold_note = "Standard BTL period — absorbs acquisition costs and delivers compound growth"

    # Reasoning narrative
    parts = [
        f"At £{recommended_offer:,} ({discount_pct}% below estimate), gross yield rises to {offer_yield:.1f}%.",
    ]
    if cashflow > 0:
        parts.append(f"Monthly cashflow of £{cashflow:,} makes the deal self-funding from day one.")
    else:
        parts.append(f"Cashflow negative at £{abs(cashflow):,}/mo — negotiate price down or plan a shorter hold.")
    if risk_sc >= 65:
        parts.append("High risk profile — a lower entry price or shorter hold reduces downside exposure.")
    elif risk_sc < 35:
        parts.append("Low risk profile — supports confidence in the recommended hold period.")

    # Price thresholds
    target_6pct  = int(rent * 12 / 0.06) if rent else 0   # price at which yield = 6%
    walk_away    = int(rent * 12 / 0.05) if rent else 0    # floor: below 5% yield, walk

    return {
        "recommended_offer_price":   recommended_offer,
        "offer_discount_pct":        discount_pct,
        "suggested_strategy":        strategy,
        "suggested_hold_period":     hold,
        "hold_rationale":            hold_note,
        "reasoning":                 " ".join(parts),
        "investor_type":             ("yield_investor" if g_yield >= 6
                                      else "growth_investor" if growth_r >= 5
                                      else "balanced"),
        "max_price_for_6pct_yield":  target_6pct,
        "walk_away_price":           walk_away,
    }


def _investor_verdict(
    g_yield: float, risk_sc: int, liq_sc: int, rd_sc: int, inv_sc: int,
    cashflow: int, rent: int, strategy: str, growth_r: float,
    property_confidence: str, enh_conf: dict, inv_dec: dict,
    flood_lv: str,
) -> dict:
    conf_label = enh_conf["confidence"]["overall"]["label"]   # "High" | "Medium" | "Low"

    identity_ok = property_confidence in ("high", "medium")

    is_avoid = (
        inv_sc < 35
        or risk_sc >= 70
        or g_yield < 3.0
        or (property_confidence == "low" and inv_sc < 45)
    )
    is_buy = (
        not is_avoid
        and identity_ok
        and inv_sc >= 60
        and risk_sc < 55
        and g_yield >= 5.0
        and conf_label != "Low"
    )

    if is_avoid:
        verdict = "AVOID"
    elif is_buy:
        verdict = "BUY"
    else:
        verdict = "CONDITIONAL"

    max_offer  = inv_dec.get("recommended_offer_price") if rent else None
    walk_away  = inv_dec.get("walk_away_price")         if rent else None

    if verdict == "BUY":
        summary = (f"Strong buy at £{max_offer:,} — {g_yield:.1f}% yield with manageable risk."
                   if max_offer else f"Strong buy — {g_yield:.1f}% yield with manageable risk.")
    elif verdict == "AVOID":
        if g_yield < 3.0:
            summary = f"Avoid — {g_yield:.1f}% yield is below the minimum investment threshold."
        elif risk_sc >= 70:
            summary = f"Avoid — risk score of {risk_sc} signals significant downside exposure."
        elif inv_sc < 35:
            summary = f"Avoid — investment score of {inv_sc} falls below acceptable thresholds."
        else:
            summary = "Avoid — property identity could not be confirmed with sufficient confidence."
    else:
        summary = "Conditional — proceed only if price is negotiated down and all risk factors are independently verified."

    reasons: list[str] = []
    if g_yield >= 6.5:
        reasons.append(f"High gross yield of {g_yield:.1f}%")
    elif g_yield >= 5.0:
        reasons.append(f"Solid gross yield of {g_yield:.1f}%")
    if cashflow > 0:
        reasons.append(f"Positive monthly cashflow of £{cashflow:,}")
    if inv_sc >= 65:
        reasons.append(f"Strong investment score ({inv_sc}/100)")
    if liq_sc >= 60:
        reasons.append("Good liquidity — likely to sell within 3 months")
    if rd_sc >= 65:
        reasons.append("High rental demand in this area")
    if growth_r >= 4:
        reasons.append(f"Above-average capital growth forecast ({growth_r:.1f}%/yr)")
    if risk_sc < 35:
        reasons.append(f"Low overall risk profile ({risk_sc}/100)")
    if not reasons:
        reasons.append("No strong positive signals identified from available data")

    risks: list[str] = []
    if property_confidence == "low":
        risks.append("Property identity not confirmed — data may not match the exact unit")
    elif property_confidence == "medium":
        risks.append("Property match is approximate — verify details before proceeding")
    if g_yield < 5.0:
        risks.append(f"Yield of {g_yield:.1f}% is below the 5% investment threshold")
    if cashflow < 0:
        risks.append(f"Negative cashflow of £{abs(cashflow):,}/month — deal is not self-funding")
    if risk_sc >= 55:
        risks.append(f"Elevated risk score ({risk_sc}/100)")
    if flood_lv not in ("Unknown", "Very Low", "Low"):
        risks.append(f"Flood risk rated {flood_lv}")
    if conf_label == "Low":
        risks.append("Low data confidence — all scores are estimates based on limited information")
    elif conf_label == "Medium":
        risks.append("Moderate data confidence — verify key figures independently")
    if liq_sc < 40:
        risks.append("Low liquidity — may be difficult to exit quickly if needed")
    if not risks:
        risks.append("No major risk flags identified from available data")

    return {
        "verdict":             verdict,
        "max_offer_price":     max_offer,
        "walk_away_price":     walk_away,
        "best_strategy":       strategy,
        "one_sentence_summary": summary,
        "key_reasons":         reasons[:5],
        "key_risks":           risks[:5],
        "confidence_label":    conf_label,
    }


def _rank_nearby_postcodes(postcode: str, region: str, growth_r: float, trans_sc: int) -> list:
    """
    Score nearby postcode sectors using ONS regional data + sector proximity.
    Pure computation — no additional API calls.
    """
    pc = postcode.strip().upper().replace(" ", "")
    m = re.match(r'^([A-Z]{1,2}\d{1,2}[A-Z]?)(\d)', pc)
    if not m:
        return []
    district   = m.group(1)
    sector_num = int(m.group(2))

    # Regional yield for this area
    reg_yield = REGIONAL_YIELDS.get(
        next((k for k in REGIONAL_YIELDS if k != "default" and k in region), "default"),
        REGIONAL_YIELDS["default"],
    )

    results = []
    for delta in range(-3, 4):
        s = sector_num + delta
        if not (1 <= s <= 9):
            continue
        sector = f"{district} {s}"
        base = 50
        base += int(growth_r * 3)          # growth premium
        base += trans_sc * 2               # transport premium
        base += (3 - abs(delta))           # proximity bonus (same sector = +3)
        # Regional premium / discount
        if any(k in region for k in ["london", "south east"]):
            base += 8
        elif any(k in region for k in ["north east", "wales"]):
            base -= 4
        elif any(k in region for k in ["north west", "midlands"]):
            base += 3
        score = max(20, min(95, base))
        results.append({
            "postcode_sector":     sector,
            "opportunity_score":   score,
            "opportunity_grade":   _grade(score),
            "growth_rate_pct":     growth_r,
            "estimated_yield_pct": round(reg_yield, 1),
            "is_subject_sector":   delta == 0,
        })

    return sorted(results, key=lambda x: x["opportunity_score"], reverse=True)


def _exit_strategy_engine(
    est_value: int, growth_r: float, liq_sc: int, risk_sc: int, g_yield: float,
    rent: int, beds: int, floor_area: float, strategy: str, region: str,
    inv_sc: int, cashflow: int,
) -> dict:
    val_3yr  = int(est_value * ((1 + growth_r / 100) ** 3))
    val_5yr  = int(est_value * ((1 + growth_r / 100) ** 5))
    val_10yr = int(est_value * ((1 + growth_r / 100) ** 10))

    # Pick the best exit based on deal profile
    if cashflow < 0 or g_yield < 4:
        best_exit  = "Sell after light refurbishment"
        exit_value = int(est_value * 1.10)
        timeline   = "6–12 months"
        alt_exit   = "Sell as-is to stop holding cost bleed"
    elif liq_sc >= 70 and growth_r >= 5:
        best_exit  = "Refinance to release equity, continue holding"
        exit_value = val_5yr
        timeline   = "5–7 years"
        alt_exit   = "Outright sale at 3-year mark if market peaks early"
    elif g_yield >= 6.5:
        best_exit  = "Long-term hold — sell at market peak"
        exit_value = val_10yr
        timeline   = "10+ years"
        alt_exit   = "Portfolio remortgage to extract equity without selling"
    elif beds >= 4:
        best_exit  = "Sell as HMO to specialist buyer (tenants in situ)"
        exit_value = int(est_value * 1.12)
        timeline   = "3–5 years"
        alt_exit   = "De-convert to family home before sale to widen buyer pool"
    else:
        best_exit  = "Standard open-market sale"
        exit_value = val_5yr
        timeline   = "5 years"
        alt_exit   = "Partial equity release via remortgage to fund next acquisition"

    gain        = max(0, exit_value - est_value)
    cgt_est     = int(max(0, gain - 3_000) * 0.24)
    net_proceeds = exit_value - cgt_est

    return {
        "best_exit_strategy":  best_exit,
        "expected_exit_value": exit_value,
        "exit_timeline":       timeline,
        "alternative_exit":    alt_exit,
        "projected_values":    {"3yr": val_3yr, "5yr": val_5yr, "10yr": val_10yr},
        "estimated_cgt":       cgt_est,
        "net_exit_proceeds":   net_proceeds,
        "exit_roi_pct":        round((net_proceeds - est_value) / est_value * 100, 1) if est_value else 0,
        "cgt_note":            "CGT at 24% (higher rate) on gain above £3,000 annual allowance.",
    }


def _enhanced_confidence(
    sales: list, epc: dict, demo_d: dict, crime_tot: int,
    ukhpi_d: dict, flood_d: dict, trans_d: dict,
) -> dict:
    n = len(sales)
    latest = (sales[0].get("date") or "")[:7] if sales else None
    months_old = None
    if latest:
        try:
            sale_dt = datetime.strptime(latest, "%Y-%m")
            months_old = (datetime.utcnow().year - sale_dt.year) * 12 + (datetime.utcnow().month - sale_dt.month)
        except Exception:
            pass

    val_score  = 85 if n >= 8 else 65 if n >= 3 else 30
    val_label  = "High" if n >= 8 else "Medium" if n >= 3 else "Low"
    val_reason = (f"{n} comparable sales found" if n >= 3
                  else f"Only {n} comparable sale(s) — valuation less reliable" if n > 0
                  else "No comparable sales — valuation based on regional averages")

    rent_score  = 80 if (bool(epc) and bool(demo_d)) else 60 if bool(demo_d) else 40
    rent_label  = "High" if (bool(epc) and bool(demo_d)) else "Medium" if bool(demo_d) else "Low"
    rent_reason = ("EPC and demographics data available — rent estimate well-supported"
                   if (bool(epc) and bool(demo_d)) else
                   "Demographics available but no EPC — rent estimated from regional VOA data" if bool(demo_d)
                   else "Limited data — rent is a regional default estimate")

    prop_score  = 80 if bool(epc) else 30
    prop_label  = "High" if bool(epc) else "Low"
    prop_reason = ("EPC record found — property details confirmed" if bool(epc)
                   else "No EPC record — bedrooms and floor area are not confirmed")

    scores = [val_score, rent_score, prop_score, 85, 85]
    overall_score = int(sum(scores) / len(scores))
    overall_label = "High" if overall_score >= 75 else "Medium" if overall_score >= 50 else "Low"

    epc_date = (epc.get("inspection-date") or epc.get("lodgement-date")) if epc else None
    freshness = ("recent"   if (months_old or 24) <= 12
                 else "moderate" if (months_old or 24) <= 24
                 else "stale")

    confidence = {
        # Nested objects (Lovable-safe)
        "overall":          {"label": overall_label, "score": overall_score, "reason": f"{overall_label} confidence — {val_reason.lower()}."},
        "valuation":        {"label": val_label,  "score": val_score,  "reason": val_reason},
        "rent":             {"label": rent_label,  "score": rent_score,  "reason": rent_reason},
        "property_details": {"label": prop_label,  "score": prop_score,  "reason": prop_reason},
        # Flat fields (safe for string interpolation in templates)
        "overall_label":    overall_label,
        "overall_score":    overall_score,
        "overall_data_quality": overall_label.lower(),
        "summary": (
            f"{overall_label} confidence because "
            + (f"EPC data exists but sale data is older ({months_old} months)." if (bool(epc) and months_old and months_old > 18)
               else f"{n} comparable sales found and EPC data available." if (n >= 3 and bool(epc))
               else f"only {n} sale(s) found — limited comparables." if n < 3
               else "data sources available but coverage is partial.")
        ),
        # Legacy flat fields kept for backward compat
        "valuation_score":  val_score,
        "rent_score":       rent_score,
        "growth":           "high",
        "growth_score":     85,
        "crime":            "high" if crime_tot > 0 else "low",
        "crime_score":      85 if crime_tot > 0 else 20,
        "flood":            "high" if flood_d.get("flood_zone") not in ("Unknown", None) else "medium",
        "flood_score":      85 if flood_d.get("flood_zone") not in ("Unknown", None) else 40,
        "transport":        "high" if trans_d.get("transport_score", 0) > 0 else "medium",
        "transport_score":  80 if trans_d.get("transport_score", 0) > 0 else 40,
        "data_points":      n,
    }

    data_age = {
        "summary": (
            f"Most recent sale: {months_old} months ago. "
            f"EPC lodged: {epc_date or 'Unknown'}. "
            f"Overall freshness: {freshness}."
        ),
        "last_sale_months_ago":  months_old,
        "epc_lodgement_date":    epc_date,
        "freshness_overall":     freshness,
        # Full detail fields
        "sales":    f"{months_old} months ago" if months_old is not None else "No sales data",
        "epc":      epc_date or "Unknown",
        "crime":    "Rolling 12 months (Police API, live)",
        "growth":   "ONS UK HPI November 2024",
        "rent":     "VOA Private Rental Statistics 2023-24",
        "flood":    "EA Flood Map for Planning (live)",
        "planning": "Gov.uk Planning Data API (live)",
        "ukhpi":    ukhpi_d.get("data_period") or "Unknown",
    }

    return {"confidence": confidence, "data_age": data_age, "confidence_levels": confidence}


def _build_data_quality(
    epc: dict,
    epc_list: list,
    sales: list,
    crime_tot: int,
    demo_d: dict,
    flood_d: dict,
    trans_d: dict,
    planning_d: dict,
    schools_d: list,
    ukhpi_price: float,
    epc_matched_by: str,
    property_confidence: str,
    enh_conf: dict,
) -> dict:
    now = datetime.utcnow()

    def _months_since(date_str: str) -> Optional[int]:
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m"):
            try:
                dt = datetime.strptime(date_str[: len(fmt)], fmt)
                return (now.year - dt.year) * 12 + (now.month - dt.month)
            except Exception:
                pass
        return None

    def _freshness(months: Optional[int], recent_threshold: int = 12, moderate_threshold: int = 36) -> str:
        if months is None:
            return "unknown"
        if months <= recent_threshold:
            return "recent"
        if months <= moderate_threshold:
            return "moderate"
        return "stale"

    # ── EPC ──────────────────────────────────────────────────────────────────
    epc_date = (epc.get("inspection-date") or epc.get("lodgement-date")) if epc else None
    _epc_matched_by = epc_matched_by  # local copy; may be overwritten to "none"
    if epc:
        epc_status = "found"
        epc_freshness = _freshness(_months_since(epc_date), recent_threshold=24, moderate_threshold=60)
        if _epc_matched_by == "lmk_key":
            epc_note = "Matched by exact certificate key — highest precision"
        elif _epc_matched_by == "uprn":
            epc_note = "Matched by UPRN — property-level match"
        elif _epc_matched_by in ("postcode+address", "address"):
            epc_note = "Matched by postcode + address string"
        else:
            n_recs = len(epc_list)
            epc_note = f"Matched at postcode level ({n_recs} record{'s' if n_recs != 1 else ''} available) — may not be this exact property"
    else:
        epc_status = "missing"
        _epc_matched_by = "none"
        epc_freshness = "unknown"
        epc_note = "No EPC record found — bedrooms and floor area are estimated defaults"

    # ── Land Registry ────────────────────────────────────────────────────────
    n_sales = len(sales)
    latest_sale = (sales[0].get("date") or "")[:7] if sales else None
    lr_freshness = _freshness(_months_since(latest_sale))
    if n_sales >= 3:
        lr_status = "found"
        lr_note = f"{n_sales} comparable sales found"
    elif n_sales > 0:
        lr_status = "found"
        lr_note = f"Only {n_sales} sale(s) — valuation less reliable"
    else:
        lr_status = "fallback"
        lr_freshness = "unknown"
        lr_note = "No sales found — valuation uses regional average prices"

    # ── Crime ────────────────────────────────────────────────────────────────
    crime_status = "found" if crime_tot > 0 else "missing"
    crime_note = (f"{crime_tot} incidents in rolling 12 months" if crime_tot > 0
                  else "No crime data returned for this area")

    # ── Demographics ─────────────────────────────────────────────────────────
    demo_status = "found" if demo_d else "missing"
    demo_note = (
        f"Region: {demo_d.get('region', 'Unknown')}, IMD decile: {demo_d.get('imd_decile', 'Unknown')}"
        if demo_d else "No demographic data returned — region defaults used"
    )

    # ── Flood ────────────────────────────────────────────────────────────────
    flood_known = flood_d.get("flood_zone") not in (None, "Unknown")
    flood_status = "found" if flood_known else "missing"
    flood_note = (f"Flood zone: {flood_d.get('flood_zone')}" if flood_known
                  else "Flood zone data unavailable — risk flagged as unknown")

    # ── Transport ────────────────────────────────────────────────────────────
    trans_score = trans_d.get("transport_score", 0) if trans_d else 0
    trans_status = "found" if trans_score > 0 else "missing"
    trans_note = (f"Transport score: {trans_score}/100" if trans_score > 0
                  else "No transport data — stations/stops not found via OSM")

    # ── Planning ─────────────────────────────────────────────────────────────
    plan_found = planning_d.get("risk_level") is not None
    plan_status = "found" if plan_found else "missing"
    plan_note = (f"Risk level: {planning_d.get('risk_level')}" if plan_found
                 else "Planning data unavailable for this location")

    # ── Schools ──────────────────────────────────────────────────────────────
    school_status = "found" if schools_d else "missing"
    school_note = (
        f"{len(schools_d)} school{'s' if len(schools_d) != 1 else ''} found within 1km"
        if schools_d else "No schools found within 1km"
    )

    # ── UKHPI ────────────────────────────────────────────────────────────────
    ukhpi_status = "found" if ukhpi_price > 0 else "fallback"
    ukhpi_note = ("District-level price data available" if ukhpi_price > 0
                  else "No UKHPI data — market prices use regional ONS averages")

    sources = [
        {
            "name": "EPC",
            "status": epc_status,
            "matched_by": _epc_matched_by,
            "freshness": epc_freshness,
            "note": epc_note,
        },
        {
            "name": "Land Registry",
            "status": lr_status,
            "matched_by": "postcode" if n_sales > 0 else "none",
            "freshness": lr_freshness,
            "note": lr_note,
        },
        {
            "name": "Crime",
            "status": crime_status,
            "matched_by": "coordinates" if crime_tot > 0 else "none",
            "freshness": "recent" if crime_tot > 0 else "unknown",
            "note": crime_note,
        },
        {
            "name": "Demographics",
            "status": demo_status,
            "matched_by": "postcode" if demo_d else "none",
            "freshness": "recent" if demo_d else "unknown",
            "note": demo_note,
        },
        {
            "name": "Flood",
            "status": flood_status,
            "matched_by": "coordinates" if flood_known else "none",
            "freshness": "recent" if flood_known else "unknown",
            "note": flood_note,
        },
        {
            "name": "Transport",
            "status": trans_status,
            "matched_by": "coordinates" if trans_score > 0 else "none",
            "freshness": "recent" if trans_score > 0 else "unknown",
            "note": trans_note,
        },
        {
            "name": "Planning",
            "status": plan_status,
            "matched_by": "coordinates" if plan_found else "none",
            "freshness": "recent" if plan_found else "unknown",
            "note": plan_note,
        },
        {
            "name": "Schools",
            "status": school_status,
            "matched_by": "coordinates" if schools_d else "none",
            "freshness": "recent" if schools_d else "unknown",
            "note": school_note,
        },
        {
            "name": "UKHPI",
            "status": ukhpi_status,
            "matched_by": "district" if ukhpi_price > 0 else "none",
            "freshness": "recent" if ukhpi_price > 0 else "unknown",
            "note": ukhpi_note,
        },
    ]

    exact_match = property_confidence == "high"
    overall_label = enh_conf["confidence"]["overall"]["label"]

    warnings: list[str] = []
    if not epc:
        warnings.append("No EPC record — bedrooms and floor area are not confirmed")
    elif _epc_matched_by == "postcode":
        warnings.append("EPC matched at postcode level only — details may be from a nearby property")
    if lr_status == "fallback":
        warnings.append("No comparable sales — estimated value based on regional averages only")
    if not exact_match:
        warnings.append("Property not uniquely identified — provide a full address, UPRN, or lmk_key for higher precision")

    if warnings:
        user_warning = " | ".join(warnings)
    elif overall_label == "High":
        user_warning = "All key data sources returned results — high-confidence analysis"
    else:
        user_warning = "Analysis complete — review individual source statuses for data gaps"

    return {
        "overall_label": overall_label,
        "exact_property_match": exact_match,
        "sources": sources,
        "user_warning": user_warning,
    }


# ── Fallback benchmark widget helpers (v6) ────────────────────────────────────

def _yield_benchmark_widget(region: str, rent: int, est_value: int, sales: list, ukhpi_d: dict) -> dict:
    """Yield comparison with full benchmark hierarchy: postcode → district → region → national."""
    national_yield = 5.5
    _rk = next((k for k in REGIONAL_YIELDS if k != "default" and k in region), "default")
    regional_yield = REGIONAL_YIELDS[_rk]
    subject_yield = round(rent * 12 / est_value * 100, 2) if est_value else None

    # Postcode-level: derived from local Land Registry median sale price
    postcode_yield: Optional[float] = None
    if len(sales) >= 3:
        prices = [s.get("price_gbp", 0) for s in sales if s.get("price_gbp")]
        if prices:
            local_median = statistics.median(prices)
            if local_median > 0:
                postcode_yield = round(rent * 12 / local_median * 100, 2)

    # District-level: UKHPI district average price as denominator
    district_yield: Optional[float] = None
    if ukhpi_d.get("district_avg") and ukhpi_d["district_avg"] > 0:
        district_yield = round(rent * 12 / ukhpi_d["district_avg"] * 100, 2)

    # Select best available benchmark level
    if postcode_yield is not None:
        benchmark_level = "postcode"
        benchmark_yield = postcode_yield
        confidence = "medium"
        reason = (
            f"Benchmark derived from {len(sales)} local Land Registry sales (median price) "
            "and VOA regional rent estimate. Postcode-level — modelled, not surveyed."
        )
        benchmark_label = f"Postcode benchmark (Land Registry, {len(sales)} sales) — modelled"
        data_sources = ["Land Registry", "modelled rent estimate"]
    elif district_yield is not None:
        benchmark_level = "district"
        benchmark_yield = district_yield
        confidence = "medium"
        reason = (
            "Postcode-level sales data insufficient; district UKHPI average price used as benchmark denominator. "
            "Benchmark is modelled — treat as indicative."
        )
        benchmark_label = "District benchmark (UKHPI average price) — modelled"
        data_sources = ["Land Registry", "modelled rent estimate"]
    elif _rk != "default":
        benchmark_level = "region"
        benchmark_yield = regional_yield
        confidence = "low"
        reason = (
            "District UKHPI data unavailable; regional gross yield benchmark applied "
            "(BM Solutions/Rightmove 2024 survey). Modelled — not property-specific."
        )
        benchmark_label = f"Regional benchmark ({region.title()}) — BM Solutions/Rightmove 2024 survey"
        data_sources = ["modelled rent estimate"]
    else:
        benchmark_level = "national"
        benchmark_yield = national_yield
        confidence = "low"
        reason = (
            "Regional data unavailable; UK national average gross yield used as fallback. "
            "Modelled — broad estimate only."
        )
        benchmark_label = "National benchmark (UK average) — modelled fallback"
        data_sources = ["modelled rent estimate"]

    # Performance label and delta
    if subject_yield is not None and benchmark_yield:
        comparison_delta_pct = round(subject_yield - benchmark_yield, 2)
        if comparison_delta_pct > 0.5:
            performance_label = "above benchmark"
        elif comparison_delta_pct < -0.5:
            performance_label = "below benchmark"
        else:
            performance_label = "in line"
    else:
        comparison_delta_pct = None
        performance_label = "in line"

    return {
        # ── New spec fields ──
        "available": True,
        "subject_yield": subject_yield,
        "benchmark_yield": benchmark_yield,
        "benchmark_level": benchmark_level,
        "benchmark_label": benchmark_label,
        "comparison_delta_pct": comparison_delta_pct,
        "performance_label": performance_label,
        "confidence": confidence,
        "reason": reason,
        "data_sources": data_sources,
        # ── Backward-compatible fields ──
        "postcode_average_yield": postcode_yield,
        "district_average_yield": district_yield,
        "regional_average_yield": regional_yield,
        "national_average_yield": national_yield,
        "level_used": benchmark_level,
        "fallback_source": benchmark_label,
        "comparison_label": performance_label,
        "gross_yield_pct": subject_yield,
        "regional_avg_yield_pct": regional_yield,
        "national_avg_yield_pct": national_yield,
        "vs_regional": round(subject_yield - regional_yield, 2) if subject_yield else None,
        "rating": "Above average" if (subject_yield and subject_yield > regional_yield) else "Below average",
    }


@functools.lru_cache(maxsize=1)
def _load_lr_summary() -> dict:
    """Load data/land_registry_summary.csv once; return {DISTRICT: [{year, average_price, transactions}]}."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "land_registry_summary.csv")
    if not os.path.exists(path):
        return {}
    result: dict = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                district = row.get("postcode_district", "").strip().upper()
                if not district:
                    continue
                result.setdefault(district, []).append({
                    "year":          int(row["year"]),
                    "average_price": int(float(row["average_price"])),
                    "transactions":  int(row["transaction_count"]),
                })
    except Exception:
        return {}
    for d in result:
        result[d].sort(key=lambda x: x["year"])
    return result


def _build_modelled_series(ukhpi_d: dict, growth_r: float) -> list:
    """5-year backward-projected price series from annual growth rate."""
    base = ukhpi_d.get("district_avg") or ukhpi_d.get("regional_avg") or 250_000
    current_year = date.today().year
    series = []
    for i in range(4, -1, -1):
        yr = current_year - i
        price = int(base / ((1 + growth_r / 100) ** i)) if growth_r else int(base)
        series.append({"year": yr, "average_price": price, "transactions": 0})
    return series


def _market_trends_widget(region: str, growth_r: float, ukhpi_d: dict, sales: list, liq_sc: int, postcode: str = "") -> dict:
    """Market trends with level hierarchy and confidence labels."""
    # Derive postcode district (e.g. "SW1A 2AA" -> "SW1A")
    district_code = postcode.split()[0].upper() if postcode else ""
    lr_data = _load_lr_summary()

    # Resolve series and level
    if district_code and district_code in lr_data:
        rows = lr_data[district_code]
        series = [{"year": r["year"], "average_price": r["average_price"], "transactions": r["transactions"]} for r in rows]
        level_used = "district"
        confidence = "high"
        reason = f"District-level Land Registry price-paid data for {district_code}."
        # Compute trends from actual series
        prices = [r["average_price"] for r in rows]
        price_trend_1yr  = round((prices[-1] / prices[-2] - 1) * 100, 1) if len(prices) >= 2 else growth_r
        price_trend_3yr  = round((prices[-1] / prices[-4] - 1) * 100, 1) if len(prices) >= 4 else (
            round((1 + price_trend_1yr / 100) ** 3 * 100 - 100, 1) if price_trend_1yr is not None else None
        )
    elif ukhpi_d.get("district_avg") and ukhpi_d.get("trend_pct_6m") is not None:
        series = _build_modelled_series(ukhpi_d, growth_r)
        level_used = "district"
        confidence = "medium"
        reason = "District-level UKHPI data from HM Land Registry used."
        price_trend_1yr = growth_r
        price_trend_3yr = round((1 + growth_r / 100) ** 3 * 100 - 100, 1) if growth_r else None
    elif region and region not in ("", "default"):
        series = _build_modelled_series(ukhpi_d, growth_r)
        level_used = "region"
        confidence = "low"
        reason = "No district UKHPI data available; regional ONS HPI annual growth rate applied."
        price_trend_1yr = growth_r
        price_trend_3yr = round((1 + growth_r / 100) ** 3 * 100 - 100, 1) if growth_r else None
    else:
        series = _build_modelled_series(ukhpi_d, growth_r)
        level_used = "modelled"
        confidence = "low"
        reason = "No district or regional data available; national modelled estimate used."
        price_trend_1yr = growth_r
        price_trend_3yr = round((1 + growth_r / 100) ** 3 * 100 - 100, 1) if growth_r else None

    if len(sales) >= 5:
        recent = [s for s in sales if (s.get("date") or "") >= "2023-01-01"]
        older  = [s for s in sales if (s.get("date") or "") <  "2023-01-01"]
        if len(recent) > len(older):
            volume_trend = "increasing"
        elif older and len(recent) < len(older) * 0.7:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"
    elif len(sales) > 0:
        volume_trend = "sparse"
    else:
        volume_trend = "no_data"

    if liq_sc >= 70:
        liquidity_signal = "high"
    elif liq_sc >= 40:
        liquidity_signal = "medium"
    else:
        liquidity_signal = "low"

    return {
        "available": True,
        "level_used": level_used,
        "series": series,
        "price_trend_1yr": price_trend_1yr,
        "price_trend_3yr": price_trend_3yr,
        "transaction_volume_trend": volume_trend,
        "liquidity_signal": liquidity_signal,
        "confidence": confidence,
        "reason": reason,
        # Backward-compatible fields
        "annual_growth_pct": growth_r,
        "six_month_trend_pct": ukhpi_d.get("trend_pct_6m", 0),
        "direction": ukhpi_d.get("trend_label", "Unknown"),
        "source": "ONS UK HPI / UKHPI Land Registry",
        "data_period": ukhpi_d.get("data_period", ""),
    }


def _school_rating_widget(schools_d: list) -> dict:
    """School rating widget with confidence and fallback reason."""
    if schools_d:
        nearest = [
            {"name": s["name"], "type": s.get("type", "school"), "distance_m": s.get("distance_m")}
            for s in schools_d[:5]
        ]
        return {
            "available": True,
            "nearest_schools": nearest,
            "average_rating_label": None,
            "confidence": "low",
            "reason": (
                f"{len(schools_d)} school(s) found nearby via OpenStreetMap. "
                "Ofsted ratings are not available from this data source."
            ),
            "fallback_source": "DfE/Get Information About Schools and Ofsted data can be connected.",
            # Backward-compatible fields
            "schools_within_1km": len(schools_d),
            "nearest_school": schools_d[0]["name"],
            "nearest_school_type": schools_d[0].get("type"),
            "source": "OpenStreetMap",
        }
    return {
        "available": False,
        "nearest_schools": [],
        "average_rating_label": None,
        "confidence": "low",
        "reason": "School benchmark data is not connected yet for this area.",
        "fallback_source": "DfE/Get Information About Schools and Ofsted data can be connected.",
    }


def _deal_scanner_widget(
    rpc: str,
    region: str,
    g_yield: float,
    est_value: int,
    rent: int,
    rd_sc: int,
    liq_sc: int,
    growth_r: float,
    sales: list,
) -> dict:
    """Local opportunity scanner. Signals derived from benchmarks when live listings unavailable."""
    parts = rpc.strip().split()
    if len(parts) >= 2:
        postcode_sector = parts[0] + " " + parts[1][0]
    else:
        postcode_sector = rpc[:max(len(rpc) - 2, 1)]

    _rk = next((k for k in REGIONAL_YIELDS if k != "default" and k in region), "default")
    regional_yield = REGIONAL_YIELDS[_rk]
    signals = []

    diff_yield = round(g_yield - regional_yield, 1)
    if diff_yield >= 1.0:
        signals.append({
            "signal": f"Yield {g_yield:.1f}% — {diff_yield}pp above regional average ({regional_yield:.1f}%)",
            "strength": "high",
            "reason": "Higher-than-average yield indicates strong rental income relative to purchase price.",
        })
    elif diff_yield > 0:
        signals.append({
            "signal": f"Yield {g_yield:.1f}% — marginally above regional average ({regional_yield:.1f}%)",
            "strength": "medium",
            "reason": "Yield is above regional benchmark but not significantly.",
        })
    else:
        signals.append({
            "signal": f"Yield {g_yield:.1f}% — below regional average ({regional_yield:.1f}%)",
            "strength": "low",
            "reason": "Below-benchmark yield; negotiating a lower price could improve returns.",
        })

    if rd_sc >= 70:
        signals.append({
            "signal": "High rental demand in this area",
            "strength": "high",
            "reason": "Strong transport links and local amenities drive above-average tenant demand.",
        })
    elif rd_sc >= 45:
        signals.append({
            "signal": "Moderate rental demand in this area",
            "strength": "medium",
            "reason": "Typical demand for the region — achievable with good property condition and management.",
        })
    else:
        signals.append({
            "signal": "Below-average rental demand signals",
            "strength": "low",
            "reason": "Weaker transport or amenity scores suggest demand may be slower; factor in void periods.",
        })

    if growth_r >= 5.5:
        signals.append({
            "signal": f"Above-average capital growth region ({growth_r:.1f}% p.a. ONS)",
            "strength": "high",
            "reason": "Above-average price growth indicates capital appreciation opportunity.",
        })
    elif growth_r >= 3.5:
        signals.append({
            "signal": f"Moderate capital growth region ({growth_r:.1f}% p.a. ONS)",
            "strength": "medium",
            "reason": "Steady growth in line with national trend.",
        })

    if liq_sc >= 60:
        signals.append({
            "signal": "Good market liquidity — active transaction volumes",
            "strength": "medium",
            "reason": "Active market indicates easier exit when needed.",
        })
    elif liq_sc < 30:
        signals.append({
            "signal": "Low market liquidity — fewer transactions in this area",
            "strength": "low",
            "reason": "Illiquid market may extend exit timeline and reduce negotiating power.",
        })

    deals = _find_deals(sales)
    deal_sc_val = _deal_score_calc(sales)

    if sales:
        confidence = "medium"
        reason = (
            f"Opportunity analysis based on {len(sales)} Land Registry transaction(s), "
            "regional yield benchmarks, rental demand score, and ONS growth data. "
            "No live listing inventory available — signals are modelled from public data."
        )
    else:
        confidence = "low"
        reason = (
            "No Land Registry transactions found for this postcode. "
            "Opportunity signals are modelled from regional benchmarks, yield estimates, and area indicators. "
            "No live listings data is available from this source."
        )

    return {
        "available": True,
        "postcode_sector": postcode_sector,
        "opportunity_signals": signals,
        "deals": deals,
        "reason": reason,
        "confidence": confidence,
        # Backward-compatible fields
        "postcode": rpc,
        "score": deal_sc_val,
        "label": _deal_label(deal_sc_val),
        "median_price": int(statistics.median([s.get("price_gbp", 0) for s in sales if s.get("price_gbp")])) if sales else 0,
    }
