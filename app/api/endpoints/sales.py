"""GET /property-sales"""
from fastapi import APIRouter, Query, HTTPException
from app.services.data_fetchers import land_registry
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/property-sales", summary="Historical sold prices for a postcode")
async def get_property_sales(
    postcode: str = Query(..., description="UK postcode e.g. SW1A 2AA"),
    limit: int = Query(20, ge=1, le=100),
):
    try:
        data = await land_registry.fetch(postcode, limit=limit)
        return data
    except Exception as e:
        log.error("sales_endpoint_failed", postcode=postcode, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
