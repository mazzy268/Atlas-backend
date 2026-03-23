"""GET /demographics"""
from fastapi import APIRouter, Query, HTTPException
from app.services.data_fetchers import demographics as demo_fetcher
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/demographics", summary="ONS demographic data for a postcode")
async def get_demographics(
    postcode: str = Query(..., description="UK postcode e.g. SW1A 2AA"),
):
    try:
        data = await demo_fetcher.fetch(postcode)
        return data
    except Exception as e:
        log.error("demographics_endpoint_failed", postcode=postcode, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
