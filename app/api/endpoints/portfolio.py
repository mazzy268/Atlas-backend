"""
Portfolio tracker endpoints.
GET  /portfolio/{user_id}          — list all holdings
POST /portfolio/{user_id}/holding  — add a property to portfolio
GET  /portfolio/{user_id}/summary  — aggregated portfolio analytics
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.db.session import get_db
from app.models.database import Portfolio, PortfolioHolding
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


class AddHoldingRequest(BaseModel):
    property_address: str
    purchase_price_gbp: int
    purchase_date: Optional[str] = None
    strategy: Optional[str] = "BTL"
    monthly_rent_gbp: Optional[int] = None
    mortgage_payment_gbp: Optional[int] = None
    notes: Optional[str] = None


@router.get("/portfolio/{user_id}", summary="List all portfolio holdings for a user")
async def get_portfolio(user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        return {"user_id": user_id, "holdings": [], "total_properties": 0}

    holdings_stmt = select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio.id)
    holdings_result = await db.execute(holdings_stmt)
    holdings = holdings_result.scalars().all()

    return {
        "user_id": user_id,
        "portfolio_id": str(portfolio.id),
        "portfolio_name": portfolio.name,
        "holdings": [_holding_to_dict(h) for h in holdings],
        "total_properties": len(holdings),
    }


@router.post("/portfolio/{user_id}/holding", summary="Add a property to a user's portfolio")
async def add_holding(
    user_id: str,
    request: AddHoldingRequest,
    db: AsyncSession = Depends(get_db),
):
    # Get or create portfolio
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        portfolio = Portfolio(user_id=user_id, name=f"{user_id}'s Portfolio")
        db.add(portfolio)
        await db.flush()

    holding = PortfolioHolding(
        portfolio_id=portfolio.id,
        purchase_price=request.purchase_price_gbp * 100,   # store in pence
        purchase_date=datetime.fromisoformat(request.purchase_date) if request.purchase_date else None,
        strategy=request.strategy,
        monthly_rent=request.monthly_rent_gbp * 100 if request.monthly_rent_gbp else None,
        mortgage_payment=request.mortgage_payment_gbp * 100 if request.mortgage_payment_gbp else None,
        notes=request.notes,
    )
    db.add(holding)
    await db.flush()

    log.info("holding_added", user_id=user_id, address=request.property_address)
    return {"message": "Holding added", "holding_id": str(holding.id)}


@router.get("/portfolio/{user_id}/summary", summary="Aggregated portfolio metrics")
async def get_portfolio_summary(user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Portfolio).where(Portfolio.user_id == user_id)
    result = await db.execute(stmt)
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holdings_stmt = select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio.id)
    holdings_result = await db.execute(holdings_stmt)
    holdings = holdings_result.scalars().all()

    total_purchase = sum(h.purchase_price or 0 for h in holdings)
    total_rent = sum(h.monthly_rent or 0 for h in holdings)
    total_mortgage = sum(h.mortgage_payment or 0 for h in holdings)
    monthly_cashflow = total_rent - total_mortgage

    return {
        "user_id": user_id,
        "total_properties": len(holdings),
        "total_purchase_value_gbp": total_purchase // 100,
        "total_monthly_rent_gbp": total_rent // 100,
        "total_monthly_mortgage_gbp": total_mortgage // 100,
        "monthly_cashflow_gbp": monthly_cashflow // 100,
        "annual_cashflow_gbp": (monthly_cashflow * 12) // 100,
        "blended_gross_yield_pct": round(
            (total_rent * 12 / total_purchase * 100) if total_purchase else 0, 2
        ),
        "strategies": list({h.strategy for h in holdings if h.strategy}),
    }


def _holding_to_dict(h: PortfolioHolding) -> dict:
    return {
        "id": str(h.id),
        "strategy": h.strategy,
        "purchase_price_gbp": (h.purchase_price or 0) // 100,
        "purchase_date": h.purchase_date.isoformat() if h.purchase_date else None,
        "monthly_rent_gbp": (h.monthly_rent or 0) // 100,
        "mortgage_payment_gbp": (h.mortgage_payment or 0) // 100,
        "monthly_cashflow_gbp": ((h.monthly_rent or 0) - (h.mortgage_payment or 0)) // 100,
        "notes": h.notes,
    }
