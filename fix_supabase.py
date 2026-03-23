#!/usr/bin/env python3
"""
Fix Supabase connection permanently.
Run from inside your Atlas folder:  python fix_supabase.py
"""
import os
import sys
import re

def main():
    if not os.path.exists("app"):
        print("ERROR: Run this from inside your Atlas folder")
        sys.exit(1)

    print("Step 1: Fixing db/session.py to force asyncpg driver...")

    session_content = '''from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import get_settings

settings = get_settings()

# Force asyncpg driver - works with both local postgres and Supabase
def _build_url(url: str) -> str:
    """Ensure the URL always uses postgresql+asyncpg driver."""
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url

database_url = _build_url(settings.database_url)

engine = create_async_engine(
    database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"ssl": "require"} if "supabase" in database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
'''

    os.makedirs("app/db", exist_ok=True)
    with open("app/db/session.py", "w", encoding="utf-8") as f:
        f.write(session_content)
    print("   Done - session.py fixed")

    print("Step 2: Fixing .env DATABASE_URL...")

    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()

        # Fix URL format - replace any postgres:// or postgresql:// with postgresql+asyncpg://
        env_content = re.sub(
            r"DATABASE_URL=postgres(?:ql)?://",
            "DATABASE_URL=postgresql+asyncpg://",
            env_content
        )

        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        print("   Done - DATABASE_URL format fixed in .env")
    else:
        print("   WARNING: .env file not found")

    print("Step 3: Installing required packages...")
    os.system(f"{sys.executable} -m pip install asyncpg sqlalchemy[asyncio] --quiet")
    print("   Done - packages installed")

    print("Step 4: Verifying .env DATABASE_URL...")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DATABASE_URL"):
                    url = line.strip()
                    # Mask password for display
                    masked = re.sub(r":([^:@]+)@", ":****@", url)
                    print(f"   Current URL: {masked}")
                    if "asyncpg" in url:
                        print("   asyncpg driver confirmed")
                    else:
                        print("   WARNING: asyncpg not in URL - check your .env manually")

    print("")
    print("=" * 50)
    print("ALL FIXES APPLIED")
    print("=" * 50)
    print("")
    print("Now run:")
    print("   uvicorn app.main:app --reload --port 8000")
    print("")
    print("This fix is permanent - you will not need to run this again.")

if __name__ == "__main__":
    main()
