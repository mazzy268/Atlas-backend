#!/usr/bin/env python3
"""
Atlas Fix Script - Run this from inside your Atlas folder
Fixes the AI client to use Groq (free) and clears the cache
Usage: python fix_atlas.py
"""
import os
import sys

def main():
    # Check we are in Atlas folder
    if not os.path.exists("app"):
        print("ERROR: Please run this from inside your Atlas folder")
        print("Open cmd inside Atlas folder and type: python fix_atlas.py")
        sys.exit(1)

    print("Step 1: Writing new AI client using Groq...")
    
    groq_client = '''import json
import re
from groq import AsyncGroq
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def complete_json(prompt: str, feature_name: str = "unknown") -> dict:
    if not settings.groq_api_key:
        return {"error": "Groq API key not configured", "feature": feature_name}
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert UK property analyst AI. "
                        "Always respond with valid JSON only. "
                        "No markdown, no code blocks, no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content
        result = _safe_parse_json(raw)
        log.info("ai_feature_complete", feature=feature_name)
        return result
    except Exception as e:
        log.error("ai_completion_failed", feature=feature_name, error=str(e))
        return {"error": str(e), "feature": feature_name}


async def complete_text(prompt: str, feature_name: str = "summary") -> str:
    if not settings.groq_api_key:
        return "AI summary unavailable - Groq API key not configured."
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are Atlas, an expert UK property investment assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("ai_text_failed", feature=feature_name, error=str(e))
        return f"Summary generation failed: {e}"


def _safe_parse_json(raw: str) -> dict:
    if not raw:
        return {}
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\\{.*\\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"parse_error": "Could not parse AI response", "raw": raw[:500]}
'''

    os.makedirs("app/services/ai_analysis", exist_ok=True)
    with open("app/services/ai_analysis/openai_client.py", "w", encoding="utf-8") as f:
        f.write(groq_client)
    print("   Done - AI client updated to Groq")

    print("Step 2: Updating config to include groq_api_key...")
    
    config_path = "app/core/config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        config = f.read()
    
    if "groq_api_key" not in config:
        config = config.replace(
            'openai_api_key: str = ""',
            'openai_api_key: str = ""\n    groq_api_key: str = ""'
        )
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config)
        print("   Done - groq_api_key added to config")
    else:
        print("   Already has groq_api_key - skipping")

    print("Step 3: Checking .env file for GROQ_API_KEY...")
    
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()
        
        if "GROQ_API_KEY" not in env_content:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write("\nGROQ_API_KEY=PASTE_YOUR_GROQ_KEY_HERE\n")
            print("   Added GROQ_API_KEY placeholder to .env")
            print("   >>> YOU MUST edit .env and replace PASTE_YOUR_GROQ_KEY_HERE with your real key")
        else:
            print("   GROQ_API_KEY already in .env")
    
    print("Step 4: Clearing old cached reports from database...")
    
    clear_cache_script = '''import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath("."))

async def clear_cache():
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy import text
        from app.core.config import get_settings
        settings = get_settings()
        engine = create_async_engine(settings.database_url, echo=False)
        async with engine.begin() as conn:
            result = await conn.execute(text("DELETE FROM property_reports"))
            print(f"   Cleared {result.rowcount} cached reports from database")
        await engine.dispose()
    except Exception as e:
        print(f"   Cache clear skipped: {e}")

asyncio.run(clear_cache())
'''
    
    with open("_temp_clear_cache.py", "w", encoding="utf-8") as f:
        f.write(clear_cache_script)
    
    os.system(f"{sys.executable} _temp_clear_cache.py")
    os.remove("_temp_clear_cache.py")

    print("")
    print("=" * 50)
    print("ALL FIXES APPLIED")
    print("=" * 50)
    print("")
    print("NOW DO THIS:")
    print("")
    print("1. Get your FREE Groq API key:")
    print("   Go to: console.groq.com")
    print("   Sign up -> API Keys -> Create API Key -> Copy it")
    print("")
    print("2. Open your .env file:")
    print("   notepad .env")
    print("   Find: GROQ_API_KEY=PASTE_YOUR_GROQ_KEY_HERE")
    print("   Replace with your real key")
    print("   Save and close")
    print("")
    print("3. Install Groq library:")
    print("   pip install groq")
    print("")
    print("4. Restart the API:")
    print("   uvicorn app.main:app --reload --port 8000")
    print("")
    print("5. Go to http://localhost:8000/docs")
    print("   Try POST /analyse-property with force_refresh: true")
    print("")

if __name__ == "__main__":
    main()
