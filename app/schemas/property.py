from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime


# ── Requests ──────────────────────────────────────────────────────────────────

class AnalysePropertyRequest(BaseModel):
    address: str = Field(..., min_length=5, examples=["10 Downing Street, London, SW1A 2AA"])
    force_refresh: bool = Field(False, description="Bypass cache and regenerate report")


class PropertySalesRequest(BaseModel):
    postcode: str = Field(..., examples=["SW1A 2AA"])
    limit: int = Field(20, ge=1, le=100)


class CrimeDataRequest(BaseModel):
    latitude: float
    longitude: float
    radius_m: int = Field(1600, ge=100, le=5000)


class DemographicsRequest(BaseModel):
    postcode: str


class FloodRiskRequest(BaseModel):
    latitude: float
    longitude: float


# ── Nested response models ────────────────────────────────────────────────────

class Coordinates(BaseModel):
    latitude: float
    longitude: float
    display_name: str


class InvestmentScoreResult(BaseModel):
    score: int = Field(..., ge=0, le=100)
    grade: str                              # A/B/C/D/F
    reasoning: str
    key_positives: list[str]
    key_risks: list[str]


class StrategyResult(BaseModel):
    recommended_strategies: list[str]
    primary_strategy: str
    reasoning: str
    estimated_yields: dict[str, Any]


class RenovationResult(BaseModel):
    estimated_cost_gbp: int
    after_repair_value_gbp: int
    current_estimated_value_gbp: int
    roi_pct: float
    recommended_works: list[str]
    payback_period_months: int


class FloorplanAnalysis(BaseModel):
    estimated_bedrooms: int
    extension_potential: str
    loft_conversion_viable: bool
    hmo_room_potential: int
    layout_notes: str


class NeighbourhoodIntelligence(BaseModel):
    crime_score: int                        # 0–10 (10 = safest)
    crime_summary: str
    school_rating: str
    best_nearby_school: str
    transport_score: int                    # 0–10
    transport_summary: str
    income_estimate: str
    deprivation_decile: Optional[int]
    overall_desirability: str


class PlanningItem(BaseModel):
    reference: str
    description: str
    status: str
    decision_date: Optional[str]
    distance_m: Optional[float]


class PlanningScannerResult(BaseModel):
    nearby_applications: list[PlanningItem]
    risk_level: str                         # low / medium / high
    development_opportunity: str
    notes: str


class DealFinderResult(BaseModel):
    is_below_market: bool
    estimated_market_value_gbp: Optional[int]
    discount_pct: Optional[float]
    deal_type: str                          # below-market / fair-value / overpriced
    recommendation: str


class PriceGrowthResult(BaseModel):
    current_estimate_gbp: int
    one_year_forecast_gbp: int
    three_year_forecast_gbp: int
    five_year_forecast_gbp: int
    annual_growth_rate_pct: float
    confidence: str                         # low / medium / high
    drivers: list[str]


class RentalYieldResult(BaseModel):
    estimated_monthly_rent_gbp: int
    gross_yield_pct: float
    net_yield_pct: float
    monthly_mortgage_estimate_gbp: int
    monthly_cashflow_gbp: int
    annual_profit_gbp: int
    assumptions: dict[str, Any]


class PortfolioSummary(BaseModel):
    total_properties: int
    total_value_gbp: int
    total_equity_gbp: int
    blended_yield_pct: float
    monthly_cashflow_gbp: int


# ── Full report ───────────────────────────────────────────────────────────────

class PropertyReport(BaseModel):
    address: str
    coordinates: Coordinates
    generated_at: datetime

    # Raw data presence flags
    data_sources: dict[str, bool]

    # 12 AI features
    investment_score: InvestmentScoreResult
    strategy_detector: StrategyResult
    renovation_predictor: RenovationResult
    floorplan_analysis: FloorplanAnalysis
    neighbourhood_intelligence: NeighbourhoodIntelligence
    rental_demand_score: int
    planning_scanner: PlanningScannerResult
    deal_finder: DealFinderResult
    price_growth_predictor: PriceGrowthResult
    rental_yield_simulator: RentalYieldResult
    ai_summary: str

    # Raw data (optional — returned when debug=true)
    raw_data: Optional[dict[str, Any]] = None


# ── Utility responses ─────────────────────────────────────────────────────────

class SaleRecord(BaseModel):
    transaction_id: str
    price_gbp: int
    date: str
    property_type: str
    tenure: str
    new_build: bool


class CrimeCategory(BaseModel):
    category: str
    count: int


class CrimeDataResponse(BaseModel):
    location: Coordinates
    radius_m: int
    total_crimes: int
    by_category: list[CrimeCategory]
    period: str


class DemographicsResponse(BaseModel):
    postcode: str
    area_name: str
    population: Optional[int]
    median_age: Optional[float]
    employment_rate: Optional[float]
    owner_occupied_pct: Optional[float]
    source: str


class FloodRiskResponse(BaseModel):
    location: Coordinates
    flood_zone: str
    risk_level: str
    river_sea_risk: str
    surface_water_risk: str
    reservoir_risk: str
    notes: str
