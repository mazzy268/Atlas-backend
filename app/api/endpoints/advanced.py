"""
Advanced Features Router
All 8 new institutional-grade endpoints.
Mount this router in your main.py:

    from app.api.endpoints.advanced import router as advanced_router
    app.include_router(advanced_router, tags=["Advanced Intelligence"])
"""
from fastapi import APIRouter, Query, HTTPException
from app.services.advanced import (
    market_heatmap,
    liquidity_engine,
    true_value,
    development_potential,
    infrastructure_impact,
    street_intelligence,
    market_risk,
    deal_scanner,
)
from app.services.geocoder import geocode_address
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


# ── 1. Market Heatmap ─────────────────────────────────────────────────────────

@router.get(
    "/market-heatmap",
    summary="Market opportunity score for a UK location",
)
async def get_market_heatmap(
    location: str = Query(..., description="City, town, or postcode e.g. Manchester or M1"),
    postcode: str | None = Query(None, description="Optional full postcode for more precise data"),
):
    try:
        return await market_heatmap.calculate_heatmap(location, postcode)
    except Exception as e:
        log.error("heatmap_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Liquidity Score ────────────────────────────────────────────────────────

@router.get(
    "/liquidity-score",
    summary="How easy is it to sell a property in this postcode?",
)
async def get_liquidity_score(
    postcode: str = Query(..., description="UK postcode e.g. NE15 6DL"),
):
    try:
        return await liquidity_engine.calculate_liquidity(postcode)
    except Exception as e:
        log.error("liquidity_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. True Value Estimation ──────────────────────────────────────────────────

@router.get(
    "/true-value",
    summary="Multi-model property valuation",
)
async def get_true_value(
    address: str = Query(..., description="Full UK address"),
    postcode: str = Query(..., description="UK postcode"),
):
    try:
        return await true_value.estimate_true_value(address, postcode)
    except Exception as e:
        log.error("true_value_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Development Potential ──────────────────────────────────────────────────

@router.post(
    "/development-potential",
    summary="AI development potential analysis",
)
async def get_development_potential(
    address: str = Query(..., description="Full UK address"),
    postcode: str = Query(..., description="UK postcode"),
):
    try:
        coords = await geocode_address(address)
        return await development_potential.analyse_development_potential(
            address, postcode, coords["latitude"], coords["longitude"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("dev_potential_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Infrastructure Impact ──────────────────────────────────────────────────

@router.get(
    "/infrastructure-impact",
    summary="Nearby infrastructure signals and value impact",
)
async def get_infrastructure_impact(
    latitude: float = Query(...),
    longitude: float = Query(...),
    postcode: str = Query(..., description="UK postcode"),
):
    try:
        return await infrastructure_impact.analyse_infrastructure(latitude, longitude, postcode)
    except Exception as e:
        log.error("infrastructure_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Street Intelligence ────────────────────────────────────────────────────

@router.get(
    "/street-intelligence",
    summary="Street-level investment scoring",
)
async def get_street_intelligence(
    address: str = Query(..., description="Full UK address including street name"),
):
    try:
        coords = await geocode_address(address)
        return await street_intelligence.analyse_street(
            address, coords["latitude"], coords["longitude"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("street_intelligence_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. Market Risk Analysis ───────────────────────────────────────────────────

@router.get(
    "/risk-analysis",
    summary="Comprehensive investment risk scoring",
)
async def get_risk_analysis(
    address: str = Query(..., description="Full UK address"),
    postcode: str = Query(..., description="UK postcode"),
):
    try:
        coords = await geocode_address(address)
        return await market_risk.analyse_risk(
            address, postcode, coords["latitude"], coords["longitude"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("risk_analysis_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── 8. Deal Scanner ───────────────────────────────────────────────────────────

@router.get(
    "/deal-scanner",
    summary="Scan for undervalued and distressed property deals",
)
async def get_deal_scanner(
    postcode: str = Query(..., description="UK postcode to scan e.g. NE15 6DL"),
    address: str | None = Query(None, description="Optional specific address"),
):
    try:
        return await deal_scanner.scan_deals(postcode, address)
    except Exception as e:
        log.error("deal_scanner_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
