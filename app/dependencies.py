from typing import Generator

from app.core.database import SessionLocal
from app.core import config
from app.providers.llm_provider import LLMProvider
from app.providers.disabled_speech_provider import DisabledSpeechProvider
from app.providers.microsoft_voice_live_provider import MicrosoftVoiceLiveProvider
from app.providers.speech_provider import SpeechProvider
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
    return model_selector.select(config.LLM_PROVIDER, config.LLM_MODEL)


def get_speech_provider() -> SpeechProvider:
    """Return the active Speech (STT/TTS) provider."""

    if config.USE_MICROSOFT_VOICE_LIVE:
        return MicrosoftVoiceLiveProvider()

    return DisabledSpeechProvider()
