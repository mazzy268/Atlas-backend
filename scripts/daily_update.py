"""
scripts/daily_update.py
Scheduled data refresh job — refreshes cached reports that are about to expire
and pre-warms the cache for frequently queried properties.

Usage:
    python scripts/daily_update.py

Cron (runs at 2am daily):
    0 2 * * * /path/to/venv/bin/python /path/to/scripts/daily_update.py
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update
from app.models.database import PropertyReport, Base
from app.services.ai_analysis.report_builder import build_report
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
log = get_logger("daily_update")
settings = get_settings()


async def run_daily_update():
    log.info("daily_update_start")
    engine = create_async_engine(settings.database_url)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as db:
        # Find reports expiring in next 6 hours
        expiry_threshold = datetime.utcnow() + timedelta(hours=6)
        stmt = (
            select(PropertyReport)
            .where(PropertyReport.expires_at <= expiry_threshold)
            .order_by(PropertyReport.generated_at.desc())
            .limit(50)       # refresh up to 50 at a time
        )
        result = await db.execute(stmt)
        stale_reports = result.scalars().all()

        log.info("stale_reports_found", count=len(stale_reports))

        refreshed = 0
        failed = 0
        for report in stale_reports:
            try:
                log.info("refreshing_report", address=report.address)
                new_report = await build_report(report.address)

                # Update existing record
                stmt = (
                    update(PropertyReport)
                    .where(PropertyReport.id == report.id)
                    .values(
                        land_registry_data=new_report.get("raw_data", {}).get("land_registry"),
                        epc_data=new_report.get("raw_data", {}).get("epc"),
                        crime_data=new_report.get("raw_data", {}).get("crime"),
                        demographics_data=new_report.get("raw_data", {}).get("demographics"),
                        flood_risk_data=new_report.get("raw_data", {}).get("flood"),
                        planning_data=new_report.get("raw_data", {}).get("planning"),
                        schools_data=new_report.get("raw_data", {}).get("schools"),
                        transport_data=new_report.get("raw_data", {}).get("transport"),
                        investment_score=new_report.get("investment_score", {}).get("score"),
                        strategy_detector=new_report.get("strategy_detector"),
                        renovation_predictor=new_report.get("renovation_predictor"),
                        neighbourhood_intelligence=new_report.get("neighbourhood_intelligence"),
                        rental_demand_score=new_report.get("rental_demand_score"),
                        planning_scanner=new_report.get("planning_scanner"),
                        deal_finder=new_report.get("deal_finder"),
                        price_growth_predictor=new_report.get("price_growth_predictor"),
                        rental_yield_simulator=new_report.get("rental_yield_simulator"),
                        ai_summary=new_report.get("ai_summary"),
                        generated_at=datetime.utcnow(),
                        expires_at=datetime.utcnow() + timedelta(seconds=settings.cache_ttl_seconds),
                    )
                )
                await db.execute(stmt)
                await db.commit()
                refreshed += 1

                # Rate limit — be a good citizen to the external APIs
                await asyncio.sleep(2)

            except Exception as e:
                log.error("refresh_failed", address=report.address, error=str(e))
                failed += 1
                await db.rollback()

    await engine.dispose()
    log.info("daily_update_complete", refreshed=refreshed, failed=failed)
    print(f"✓ Daily update complete — {refreshed} refreshed, {failed} failed")


if __name__ == "__main__":
    asyncio.run(run_daily_update())
