"""GET /flood-risk"""
from fastapi import APIRouter, Query, HTTPException
from app.services.data_fetchers import flood_risk as flood_fetcher
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/flood-risk", summary="EA flood risk data for a lat/lng")
async def get_flood_risk(
    latitude: float = Query(...),
    longitude: float = Query(...),
):
    try:
        data = await flood_fetcher.fetch(latitude, longitude)
        return data
    except Exception as e:
        log.error("flood_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
