"""
POST /analyse-property — core endpoint.
Geocodes address, fetches all data, runs AI analysis, returns full report.
Caches results in DB for CACHE_TTL_SECONDS.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.schemas.property import AnalysePropertyRequest, PropertyReport
from app.services.ai_analysis.report_builder import build_report
from app.db.session import get_db
from app.models.database import Property, PropertyReport as DBReport
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)
settings = get_settings()


@router.post("/analyse-property", summary="Analyse a UK property and return full intelligence report")
async def analyse_property(
    request: AnalysePropertyRequest,
    db: AsyncSession = Depends(get_db),
):
    address = request.address.strip()

    # ── Cache check ────────────────────────────────────────────────────────────
    if not request.force_refresh:
        cached = await _get_cached_report(db, address)
        if cached:
            log.info("cache_hit", address=address)
            return {"source": "cache", "report": cached}

    # ── Generate fresh report ──────────────────────────────────────────────────
    try:
        report = await build_report(address)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("report_generation_failed", address=address, error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    # ── Persist to DB ──────────────────────────────────────────────────────────
    await _save_report(db, address, report)

    return {"source": "live", "report": report}


@router.post(
    "/property-assistant",
    summary="Ask the AI assistant a question about a previously analysed property",
)
async def property_assistant(
    address: str,
    question: str,
    db: AsyncSession = Depends(get_db),
):
    from app.services.ai_analysis.openai_client import complete_text
    from app.services.ai_analysis.prompts import ai_assistant_prompt

    cached = await _get_cached_report(db, address)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail="Property not analysed yet. Call POST /analyse-property first.",
        )

    # Build condensed context for the assistant
    context = _build_assistant_context(cached)
    prompt = ai_assistant_prompt(context, question)
    answer = await complete_text(prompt, "ai_assistant")

    return {"address": address, "question": question, "answer": answer}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_cached_report(db: AsyncSession, address: str) -> dict | None:
    """Return cached report if it exists and hasn't expired."""
    stmt = (
        select(DBReport)
        .where(DBReport.address == address)
        .order_by(DBReport.generated_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        return None

    # Check expiry
    if row.expires_at and row.expires_at < datetime.now(timezone.utc).replace(tzinfo=None):
        return None

    return _db_report_to_dict(row)


async def _save_report(db: AsyncSession, address: str, report: dict) -> None:
    """Persist the generated report to the database."""
    try:
        from datetime import datetime as dt
        expires_str = report.get("expires_at")
        expires_at = dt.fromisoformat(expires_str) if expires_str else None

        db_report = DBReport(
            address=address,
            land_registry_data=report.get("raw_data", {}).get("land_registry"),
            epc_data=report.get("raw_data", {}).get("epc"),
            crime_data=report.get("raw_data", {}).get("crime"),
            demographics_data=report.get("raw_data", {}).get("demographics"),
            flood_risk_data=report.get("raw_data", {}).get("flood"),
            planning_data=report.get("raw_data", {}).get("planning"),
            schools_data=report.get("raw_data", {}).get("schools"),
            transport_data=report.get("raw_data", {}).get("transport"),
            investment_score=report.get("investment_score", {}).get("score"),
            investment_score_reasoning=report.get("investment_score", {}).get("reasoning"),
            strategy_detector=report.get("strategy_detector"),
            renovation_predictor=report.get("renovation_predictor"),
            floorplan_analysis=report.get("floorplan_analysis"),
            neighbourhood_intelligence=report.get("neighbourhood_intelligence"),
            rental_demand_score=report.get("rental_demand_score"),
            planning_scanner=report.get("planning_scanner"),
            deal_finder=report.get("deal_finder"),
            price_growth_predictor=report.get("price_growth_predictor"),
            rental_yield_simulator=report.get("rental_yield_simulator"),
            ai_summary=report.get("ai_summary"),
            generated_at=dt.utcnow(),
            expires_at=expires_at,
        )
        db.add(db_report)
        await db.flush()
        log.info("report_saved", address=address, id=str(db_report.id))
    except Exception as e:
        log.error("report_save_failed", address=address, error=str(e))
        # Don't raise — saving failure shouldn't break the response


def _db_report_to_dict(row: DBReport) -> dict:
    return {
        "address": row.address,
        "generated_at": row.generated_at.isoformat() if row.generated_at else None,
        "investment_score": row.investment_score,
        "investment_score_reasoning": row.investment_score_reasoning,
        "strategy_detector": row.strategy_detector,
        "renovation_predictor": row.renovation_predictor,
        "floorplan_analysis": row.floorplan_analysis,
        "neighbourhood_intelligence": row.neighbourhood_intelligence,
        "rental_demand_score": row.rental_demand_score,
        "planning_scanner": row.planning_scanner,
        "deal_finder": row.deal_finder,
        "price_growth_predictor": row.price_growth_predictor,
        "rental_yield_simulator": row.rental_yield_simulator,
        "ai_summary": row.ai_summary,
        "raw_data": {
            "land_registry": row.land_registry_data,
            "epc": row.epc_data,
            "crime": row.crime_data,
            "demographics": row.demographics_data,
            "flood": row.flood_risk_data,
            "planning": row.planning_data,
            "schools": row.schools_data,
            "transport": row.transport_data,
        },
    }


def _build_assistant_context(report: dict) -> str:
    """Flatten the report into a concise text context for the AI assistant."""
    inv = report.get("investment_score") or {}
    strat = report.get("strategy_detector") or {}
    rys = report.get("rental_yield_simulator") or {}
    pgp = report.get("price_growth_predictor") or {}
    nb = report.get("neighbourhood_intelligence") or {}

    return f"""
Address: {report.get('address')}
Investment Score: {inv.get('score')}/100 (Grade {inv.get('grade')})
Primary Strategy: {strat.get('primary_strategy')}
Estimated Value: £{pgp.get('current_estimate_gbp', 'N/A'):,} (if int)
Gross Yield: {rys.get('gross_yield_pct')}%
Monthly Rent Estimate: £{rys.get('estimated_monthly_rent_gbp', 'N/A')}
Monthly Cashflow: £{rys.get('monthly_cashflow_gbp', 'N/A')}
5-Year Forecast: £{pgp.get('five_year_forecast_gbp', 'N/A')}
Neighbourhood: {nb.get('overall_desirability')}
Crime Score: {nb.get('crime_score')}/10
Transport Score: {nb.get('transport_score')}/10
Key Risks: {inv.get('key_risks')}
Key Positives: {inv.get('key_positives')}
AI Summary: {report.get('ai_summary', '')[:300]}
""".strip()
