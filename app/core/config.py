from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/atlas"

    # AI
    openai_api_key: str = ""
    groq_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # EPC API
    epc_api_key: str = ""
    epc_api_email: str = ""

    # OS Places (optional)
    os_places_api_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cache_ttl_seconds: int = 86400

    # Nominatim — must send a valid User-Agent per OSM policy
    nominatim_user_agent: str = "AtlasPropertyIntelligence/1.0"

    # Radius defaults (metres) for spatial queries
    crime_radius_m: int = 1600       # ~1 mile
    schools_radius_m: int = 2000
    transport_radius_m: int = 800
    planning_radius_m: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
