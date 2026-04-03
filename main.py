"""
Atlas Property Intelligence API — FastAPI entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.api.endpoints import property, sales, crime, demographics, flood, portfolio

settings = get_settings()
configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("atlas_api_starting", env=settings.app_env)
    yield
    log.info("atlas_api_shutdown")


app = FastAPI(
    title="Atlas Property Intelligence API",
    description=(
        "Analyse any UK property address and receive a full AI-powered investment report "
        "combining 8 data sources and 12 analytical features."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
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

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(property.router, tags=["Property Analysis"])
app.include_router(sales.router, tags=["Data"])
app.include_router(crime.router, tags=["Data"])
app.include_router(demographics.router, tags=["Data"])
app.include_router(flood.router, tags=["Data"])
app.include_router(portfolio.router, tags=["Portfolio"])


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0", "env": settings.app_env}


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
