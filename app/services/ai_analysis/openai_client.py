import json
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
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"parse_error": "Could not parse AI response", "raw": raw[:500]}
