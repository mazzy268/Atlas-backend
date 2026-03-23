"""GET /crime-data"""
from fastapi import APIRouter, Query, HTTPException
from app.services.data_fetchers import crime as crime_fetcher
from app.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/crime-data", summary="Crime statistics near a lat/lng")
async def get_crime_data(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius_m: int = Query(1600, ge=100, le=5000),
):
    try:
        data = await crime_fetcher.fetch(latitude, longitude, radius_m)
        return data
    except Exception as e:
        log.error("crime_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
