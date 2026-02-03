from typing import Generator

from app.core.database import SessionLocal
from app.core import config
from app.providers.llm_provider import LLMProvider
from app.providers.microsoft_voice_live_provider import MicrosoftVoiceLiveProvider
from app.services.model_selector import model_selector


def get_db() -> Generator:
    """Return a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_llm_provider() -> LLMProvider:
    """Return the active LLM provider."""
    if config.USE_MICROSOFT_VOICE_LIVE:
        return MicrosoftVoiceLiveProvider()
    # Default: use the existing dummy provider
    return model_selector.select("dummy")
