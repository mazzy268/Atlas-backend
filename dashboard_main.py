"""
Atlas Property Intelligence - Dashboard API
Master endpoint for Lovable frontend.

Run with:
    uvicorn dashboard_main:app --reload --port 8000

Or if using ngrok, keep this running and tunnel port 8000.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import re
from datetime import datetime

# ── Core app imports ──────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ MUST be AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Lovable + browser
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Advanced service imports ──────────────────────────────────────────────────
from app.services.advanced import (
    liquidity_engine,
    true_value,
    development_potential,
    infrastructure_impact,
    street_intelligence,
    market_risk,
    deal_scanner,
    market_heatmap,
)

configure_logging()
log = get_logger(__name__)
settings = get_settings()

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Atlas Property Intelligence",
    description="Master API for Lovable property dashboard",
    version="3.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory portfolio store ─────────────────────────────────────────────────
portfolio_store: list[dict] = []


# ── Request models ────────────────────────────────────────────────────────────

class PropertyRequest(BaseModel):
    postcode: str

class PortfolioAddRequest(BaseModel):
    postcode: str
    address: Optional[str] = None
    property_data: Optional[dict] = None


# ── MASTER ENDPOINT ───────────────────────────────────────────────────────────

@app.post("/analyse-property")
async def analyse_property(data: PropertyRequest):
    """
    Master endpoint — accepts a postcode, returns full dashboard JSON.
    All widgets in Lovable should read from this single response.
    """
    postcode = data.postcode.strip().upper()
    address = postcode  # use postcode as address seed for geocoding

    # Step 1: Geocode the postcode
    try:
        coords = await geocode_address(address)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not resolve postcode: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Geocoding failed: {e}")

    lat = coords["latitude"]
    lng = coords["longitude"]
    resolved_postcode = coords.get("postcode") or postcode

    # Step 2: Fan out to all data sources + advanced services concurrently
    try:
        raw_task = gather_all_data(lat, lng, resolved_postcode, address)

        advanced_task = asyncio.gather(
            _safe_call(liquidity_engine.calculate_liquidity(resolved_postcode)),
            _safe_call(true_value.estimate_true_value(address, resolved_postcode)),
            _safe_call(development_potential.analyse_development_potential(address, resolved_postcode, lat, lng)),
            _safe_call(infrastructure_impact.analyse_infrastructure(lat, lng, resolved_postcode)),
            _safe_call(street_intelligence.analyse_street(address, lat, lng)),
            _safe_call(market_risk.analyse_risk(address, resolved_postcode, lat, lng)),
            _safe_call(deal_scanner.scan_deals(resolved_postcode, address)),
        )

        raw, adv = await asyncio.gather(raw_task, advanced_task)

    except Exception as e:
        log.error("data_gather_failed", error=str(e))
        raw = {}
        adv = [{} for _ in range(7)]

    # Step 3: Run AI analysis
    try:
        ai = await run_all_ai_features(address, raw)
    except Exception as e:
        log.error("ai_features_failed", error=str(e))
        ai = {}

    # Step 4: Unpack all results safely
    liq   = _s(adv[0])
    tv    = _s(adv[1])
    dev   = _s(adv[2])
    infra = _s(adv[3])
    st    = _s(adv[4])
    risk  = _s(adv[5])
    deals = _s(adv[6])

    inv   = _s(ai.get("investment_score"))
    strat = _s(ai.get("strategy_detector"))
    reno  = _s(ai.get("renovation_predictor"))
    nb    = _s(ai.get("neighbourhood_intelligence"))
    rys   = _s(ai.get("rental_yield_simulator"))
    pgp   = _s(ai.get("price_growth"))
    rd    = _s(ai.get("rental_demand"))
    plan  = _s(ai.get("planning_scanner"))
    flip  = _s(ai.get("floorplan_analysis"))

    # EPC data
    epc_raw    = _s(raw.get("epc"))
    epc_list   = epc_raw.get("ratings") or []
    epc        = epc_list[0] if epc_list else {}

    # Sales data
    sales_raw  = _s(raw.get("land_registry"))
    sales_list = sales_raw.get("sales") or []

    # Core financials
    est_value  = _i(tv.get("consensus_value_gbp") or pgp.get("current_estimate_gbp") or reno.get("current_estimated_value_gbp"), 0)
    est_rent   = _i(rys.get("estimated_monthly_rent_gbp"), 0)
    gross_yield = _f(rys.get("gross_yield_pct"), 0.0)
    net_yield  = _f(rys.get("net_yield_pct"), 0.0)
    cashflow   = _i(rys.get("monthly_cashflow_gbp"), 0)
    annual_p   = _i(rys.get("annual_profit_gbp"), 0)

    # Growth
    val_1yr    = _i(pgp.get("one_year_forecast_gbp"), 0)
    val_3yr    = _i(pgp.get("three_year_forecast_gbp"), 0)
    val_5yr    = _i(pgp.get("five_year_forecast_gbp"), 0)
    growth_pct = _f(pgp.get("annual_growth_rate_pct"), 3.5)

    # Scores
    inv_score  = _i(inv.get("score"), 0)
    risk_score = _i(risk.get("investment_risk_score"), 50)
    liq_score  = _i(liq.get("liquidity_score"), 0)
    st_score   = _i(st.get("street_investment_score"), 0)
    deal_score = _i(deals.get("deal_score"), 0)
    rd_score   = _i(rd.get("rental_demand_score"), 0)

    # Dev
    dev_fin    = _s(dev.get("financial_summary"))
    dev_opps   = _s(dev.get("development_opportunities"))
    dev_roi    = _f(dev_fin.get("development_roi_pct"), 0.0)
    dev_score  = _i(dev.get("overall_development_score"), 0)

    # Property details from EPC
    bedrooms   = _i(epc.get("number-habitable-rooms") or flip.get("estimated_bedrooms"), 3)
    bathrooms  = _i(flip.get("estimated_bathrooms"), 1)
    floor_area = _f(epc.get("floor-area") or epc.get("total-floor-area") or epc.get("floor_area_sqm"), 0.0)
    prop_type  = epc.get("property-type") or epc.get("property_type") or "Residential"
    epc_rating = epc.get("current-energy-rating") or epc.get("current_energy_rating")

    # Risk flags
    flood_level = _s(raw.get("flood")).get("risk_level") or "Unknown"
    crime_score_val = _i(nb.get("crime_score"), 5)
    transport_s = _i(nb.get("transport_score"), 0)

    # Strategy
    strategy   = strat.get("primary_strategy") or "BTL"
    strategies = strat.get("recommended_strategies") or [strategy]
    reason     = strat.get("reasoning") or inv.get("reasoning") or "Analysis based on local market data"

    # Renovation
    light_reno = _s(reno.get("light_refurb"))
    medium_reno = _s(reno.get("medium_refurb"))
    heavy_reno  = _s(reno.get("heavy_refurb"))

    # Recent sales for comparable table
    comparables = [
        {
            "address": f"{s.get('address_paon', '')} {s.get('street', '')}".strip() or "Nearby property",
            "price_gbp": s.get("price_gbp", 0),
            "date": s.get("date", ""),
            "type": s.get("property_type", ""),
            "tenure": s.get("tenure", ""),
        }
        for s in sales_list[:5]
    ]

    # Crime breakdown
    crime_raw   = _s(raw.get("crime"))
    crime_cats  = crime_raw.get("by_category") or []
    total_crimes = crime_raw.get("total_crimes", 0)

    # Transport
    transport_raw = _s(raw.get("transport"))
    stations = transport_raw.get("nearest_stations") or []

    # Schools
    schools_raw = _s(raw.get("schools"))
    school_list = schools_raw.get("schools") or []

    # Planning
    planning_raw = _s(raw.get("planning"))
    planning_apps = planning_raw.get("applications") or []

    return {
        # ── Identity ─────────────────────────────────────────────────────────
        "postcode": resolved_postcode,
        "display_address": coords.get("display_name", postcode),
        "latitude": lat,
        "longitude": lng,
        "generated_at": datetime.utcnow().isoformat(),

        # ── Property overview widget ──────────────────────────────────────────
        "property": {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floor_area_sqm": floor_area,
            "property_type": prop_type,
            "epc_rating": epc_rating,
            "epc_current_efficiency": _i(epc.get("current-energy-efficiency") or epc.get("current_energy_efficiency"), 0),
            "epc_potential_rating": epc.get("potential-energy-rating") or epc.get("potential_energy_rating"),
            "tenure": sales_list[0].get("tenure") if sales_list else "Unknown",
            "construction": epc.get("built-form") or epc.get("built_form") or "Unknown",
            "walls": epc.get("walls-description") or epc.get("walls_description"),
            "roof": epc.get("roof-description") or epc.get("roof_description"),
            "heating": epc.get("main-heat-description") or epc.get("heating_description"),
            "windows": epc.get("windows-description") or epc.get("windows_description"),
        },

        # ── Financials widget ─────────────────────────────────────────────────
        "financials": {
            "estimated_value": est_value,
            "monthly_rent": est_rent,
            "annual_rent": est_rent * 12,
            "rental_yield": gross_yield,
            "net_yield": net_yield,
            "monthly_cashflow": cashflow,
            "annual_profit": annual_p,
            "monthly_mortgage_estimate": _i(rys.get("monthly_mortgage_estimate_gbp"), 0),
            "deposit_required": int(est_value * 0.25) if est_value else 0,
            "stamp_duty_estimate": _calc_stamp_duty(est_value),
            "total_acquisition_cost": _calc_acquisition_cost(est_value),
        },

        # ── Scores widget ─────────────────────────────────────────────────────
        "scores": {
            "investment_score": inv_score,
            "investment_grade": _grade(inv_score),
            "risk_score": risk_score,
            "risk_level": _risk_label(risk_score),
            "liquidity_score": liq_score,
            "liquidity_band": liq.get("liquidity_band") or _liquidity_label(liq_score),
            "street_score": st_score,
            "street_grade": _grade(st_score),
            "deal_score": deal_score,
            "rental_demand_score": rd_score,
            "demand_level": rd.get("demand_level") or "Medium",
        },

        # ── Growth widget ─────────────────────────────────────────────────────
        "growth": {
            "current_value": est_value,
            "one_year_projection": val_1yr,
            "three_year_projection": val_3yr,
            "five_year_projection": val_5yr,
            "annual_growth_rate_pct": growth_pct,
            "one_year_uplift": val_1yr - est_value if val_1yr and est_value else 0,
            "five_year_uplift": val_5yr - est_value if val_5yr and est_value else 0,
            "infrastructure_boost_pct": _f(infra.get("infrastructure_growth_boost"), 0.0),
            "market_phase": _s(infra.get("signals", {})).get("transport", {}).get("boost_contribution_pct", 0),
        },

        # ── AI analysis widget ────────────────────────────────────────────────
        "ai_analysis": {
            "best_strategy": strategy,
            "all_strategies": strategies,
            "reason": reason,
            "key_positives": inv.get("key_positives") or [],
            "key_risks": inv.get("key_risks") or [],
            "void_period_weeks": _i(rd.get("average_void_period_weeks"), 4),
            "tenant_profiles": rd.get("key_tenant_profiles") or [],
            "summary": ai.get("ai_summary") or "",
        },

        # ── Renovation widget ─────────────────────────────────────────────────
        "renovation": {
            "current_value": _i(reno.get("current_estimated_value_gbp"), est_value),
            "light": {
                "cost": _i(light_reno.get("estimated_cost_gbp"), 0),
                "arv": _i(light_reno.get("after_repair_value_gbp"), 0),
                "roi_pct": _f(light_reno.get("roi_pct"), 0.0),
                "works": light_reno.get("recommended_works") or [],
            },
            "medium": {
                "cost": _i(medium_reno.get("estimated_cost_gbp"), 0),
                "arv": _i(medium_reno.get("after_repair_value_gbp"), 0),
                "roi_pct": _f(medium_reno.get("roi_pct"), 0.0),
                "works": medium_reno.get("recommended_works") or [],
            },
            "heavy": {
                "cost": _i(heavy_reno.get("estimated_cost_gbp"), 0),
                "arv": _i(heavy_reno.get("after_repair_value_gbp"), 0),
                "roi_pct": _f(heavy_reno.get("roi_pct"), 0.0),
                "works": heavy_reno.get("recommended_works") or [],
            },
            "epc_upgrade_cost": _i(reno.get("epc_upgrade_cost_gbp"), 0),
            "epc_upgrade_notes": reno.get("epc_upgrade_notes") or "",
        },

        # ── Development widget ────────────────────────────────────────────────
        "development": {
            "score": dev_score,
            "roi_pct": dev_roi,
            "current_value": _i(dev_fin.get("estimated_current_value_gbp"), est_value),
            "post_dev_value": _i(dev_fin.get("estimated_post_development_value_gbp"), 0),
            "uplift": _i(dev_fin.get("estimated_uplift_gbp"), 0),
            "total_cost": _i(dev_fin.get("estimated_total_dev_cost_gbp"), 0),
            "loft": {
                "viable": bool(_s(dev_opps.get("loft_conversion")).get("viable")),
                "feasibility": _s(dev_opps.get("loft_conversion")).get("feasibility") or "Unknown",
                "cost": _i(_s(dev_opps.get("loft_conversion")).get("estimated_cost_gbp"), 0),
                "value_add": _i(_s(dev_opps.get("loft_conversion")).get("estimated_value_add_gbp"), 0),
            },
            "extension": {
                "viable": bool(_s(dev_opps.get("extension")).get("viable")),
                "feasibility": _s(dev_opps.get("extension")).get("feasibility") or "Unknown",
                "cost": _i(_s(dev_opps.get("extension")).get("estimated_cost_gbp"), 0),
                "value_add": _i(_s(dev_opps.get("extension")).get("estimated_value_add_gbp"), 0),
            },
            "hmo": {
                "viable": bool(_s(dev_opps.get("hmo_conversion")).get("viable")),
                "rooms": _i(_s(dev_opps.get("hmo_conversion")).get("potential_lettable_rooms"), 0),
                "monthly_rent": _i(_s(dev_opps.get("hmo_conversion")).get("estimated_monthly_hmo_rent_gbp"), 0),
                "conversion_cost": _i(_s(dev_opps.get("hmo_conversion")).get("estimated_conversion_cost_gbp"), 0),
            },
        },

        # ── Risk widget ───────────────────────────────────────────────────────
        "risk": {
            "overall_score": risk_score,
            "band": risk.get("risk_band") or _risk_label(risk_score),
            "flood_level": flood_level,
            "flood_warnings": len(_s(raw.get("flood")).get("active_warnings") or []),
            "crime_score": crime_score_val,
            "crime_total": total_crimes,
            "crime_breakdown": crime_cats[:5],
            "economic_vulnerability": _s(_s(risk.get("risk_breakdown")).get("economic_vulnerability")).get("score", 50),
            "red_flags": risk.get("red_flags") or [],
            "suitable_for": risk.get("suitable_for") or "All investor types",
        },

        # ── Neighbourhood widget ──────────────────────────────────────────────
        "neighbourhood": {
            "overall_desirability": nb.get("overall_desirability") or "Unknown",
            "area_trajectory": nb.get("area_trajectory") or "Stable",
            "income_estimate": nb.get("income_estimate") or "Unknown",
            "investor_appeal": nb.get("investor_appeal") or "Medium",
            "transport_score": transport_s,
            "transport_summary": nb.get("transport_summary") or "",
            "nearest_stations": stations[:3],
            "bus_stop_count": transport_raw.get("bus_stop_count", 0),
            "school_rating": nb.get("school_rating") or "Unknown",
            "best_school": nb.get("best_nearby_school") or "None found",
            "schools_nearby": school_list[:3],
            "demographics": {
                "area": _s(raw.get("demographics")).get("area_name"),
                "ward": _s(raw.get("demographics")).get("ward"),
                "region": _s(raw.get("demographics")).get("region"),
                "local_authority": _s(raw.get("demographics")).get("local_authority"),
            },
        },

        # ── Planning widget ───────────────────────────────────────────────────
        "planning": {
            "risk_level": plan.get("risk_level") or "low",
            "risk_summary": plan.get("risk_summary") or "",
            "article_4_risk": bool(plan.get("article_4_risk")),
            "permitted_development_likely": bool(plan.get("permitted_development_likely")),
            "development_opportunity": plan.get("development_opportunity") or "",
            "nearby_applications": planning_apps[:5],
            "total_applications": planning_raw.get("total_applications", 0),
        },

        # ── Comparable sales widget ───────────────────────────────────────────
        "comparables": {
            "sales": comparables,
            "total_transactions": len(sales_list),
            "avg_price": int(sum(s.get("price_gbp", 0) for s in sales_list) / len(sales_list)) if sales_list else 0,
            "latest_sale_price": sales_list[0].get("price_gbp") if sales_list else 0,
            "latest_sale_date": sales_list[0].get("date") if sales_list else None,
        },

        # ── Deal finder widget ────────────────────────────────────────────────
        "deals": {
            "score": deal_score,
            "label": deals.get("deal_score_label") or "",
            "potential_deals": (deals.get("potential_deals") or [])[:3],
            "best_deal": deals.get("best_deal"),
            "recommendation": deals.get("recommendation") or "",
            "median_area_price": _s(deals.get("area_statistics")).get("median_price_gbp", 0),
        },

        # ── HMO simulator widget ──────────────────────────────────────────────
        "hmo_analysis": {
            "feasibility": _s(ai.get("floorplan_analysis")).get("hmo_feasibility") or "Unknown",
            "room_potential": _i(_s(ai.get("floorplan_analysis")).get("hmo_room_potential"), 0),
            "hmo_notes": _s(ai.get("floorplan_analysis")).get("hmo_notes") or "",
            "estimated_monthly_hmo_rent": _i(_s(rys.get("hmo_scenario")).get("estimated_monthly_rent_gbp"), 0),
            "hmo_gross_yield": _f(_s(rys.get("hmo_scenario")).get("gross_yield_pct"), 0.0),
            "hmo_cashflow": _i(_s(rys.get("hmo_scenario")).get("monthly_cashflow_gbp"), 0),
        },

        # ── Data source status ────────────────────────────────────────────────
        "data_sources": {
            "land_registry": "error" not in _s(raw.get("land_registry")),
            "epc": "error" not in _s(raw.get("epc")),
            "crime": "error" not in _s(raw.get("crime")),
            "demographics": "error" not in _s(raw.get("demographics")),
            "flood": "error" not in _s(raw.get("flood")),
            "planning": "error" not in _s(raw.get("planning")),
            "schools": "error" not in _s(raw.get("schools")),
            "transport": "error" not in _s(raw.get("transport")),
            "active_count": sum(1 for v in raw.values() if "error" not in _s(v)),
        },
    }


# ── Portfolio endpoints ───────────────────────────────────────────────────────

@app.post("/portfolio/add")
async def portfolio_add(data: PortfolioAddRequest):
    """Save a property to the portfolio store."""
    entry = {
        "id": len(portfolio_store) + 1,
        "postcode": data.postcode.upper(),
        "address": data.address or data.postcode.upper(),
        "added_at": datetime.utcnow().isoformat(),
        "property_data": data.property_data or {},
    }
    portfolio_store.append(entry)
    log.info("portfolio_item_added", postcode=data.postcode)
    return {"status": "saved", "id": entry["id"], "total": len(portfolio_store)}


@app.get("/portfolio")
async def portfolio_list():
    """Return all saved portfolio properties."""
    return {
        "total": len(portfolio_store),
        "properties": portfolio_store,
    }


@app.delete("/portfolio/{property_id}")
async def portfolio_delete(property_id: int):
    """Remove a property from the portfolio."""
    global portfolio_store
    before = len(portfolio_store)
    portfolio_store = [p for p in portfolio_store if p["id"] != property_id]
    if len(portfolio_store) == before:
        raise HTTPException(status_code=404, detail="Property not found in portfolio")
    return {"status": "removed", "total": len(portfolio_store)}


# ── Individual endpoints ──────────────────────────────────────────────────────

@app.get("/market-heatmap")
async def get_market_heatmap(location: str, postcode: Optional[str] = None):
    try:
        return await market_heatmap.calculate_heatmap(location, postcode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deal-scanner")
async def get_deal_scanner(postcode: str):
    try:
        return await deal_scanner.scan_deals(postcode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-analysis")
async def get_risk_analysis(address: str, postcode: str):
    try:
        coords = await geocode_address(address)
        return await market_risk.analyse_risk(address, postcode, coords["latitude"], coords["longitude"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/true-value")
async def get_true_value(address: str, postcode: str):
    try:
        return await true_value.estimate_true_value(address, postcode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/liquidity-score")
async def get_liquidity(postcode: str):
    try:
        return await liquidity_engine.calculate_liquidity(postcode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/development-potential")
async def get_development(address: str, postcode: str):
    try:
        coords = await geocode_address(address)
        return await development_potential.analyse_development_potential(
            address, postcode, coords["latitude"], coords["longitude"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _safe_call(coro):
    try:
        return await coro
    except Exception as e:
        log.warning("service_call_failed", error=str(e))
        return {}


def _s(val) -> dict:
    if isinstance(val, (dict,)):
        return val
    return {}


def _i(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _f(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _grade(score: int) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "F"


def _risk_label(score: int) -> str:
    if score >= 70: return "High"
    if score >= 50: return "Medium-High"
    if score >= 35: return "Medium"
    if score >= 20: return "Low-Medium"
    return "Low"


def _liquidity_label(score: int) -> str:
    if score >= 75: return "High"
    if score >= 50: return "Medium"
    if score >= 25: return "Low"
    return "Very Low"


def _calc_stamp_duty(price: int) -> int:
    if not price:
        return 0
    if price <= 250000:
        return 0
    if price <= 925000:
        return int((price - 250000) * 0.05)
    return int(675000 * 0.05 + (price - 925000) * 0.10)


def _calc_acquisition_cost(price: int) -> int:
    if not price:
        return 0
    return int(price * 0.25 + _calc_stamp_duty(price) + 2500)
