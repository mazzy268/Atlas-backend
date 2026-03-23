import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.ext.asyncio import create_async_engine
from app.models.database import Base
from app.core.config import get_settings

settings = get_settings()


async def init_db():
    print("Connecting to database...")
    engine = create_async_engine(settings.database_url, echo=False)

    async with engine.begin() as conn:
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)

    await engine.dispose()
    print("Done! Database tables created successfully.")


if __name__ == "__main__":
    asyncio.run(init_db())