import os

from dotenv import load_dotenv


# Load environment variables from .env if present
load_dotenv()


# Minimal app configuration
APP_NAME = os.getenv("APP_NAME", "Bot Backend")
ENV = os.getenv("ENV", "dev")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Invalid database connection string")

# Microsoft Voice Live configuration (optional)
MICROSOFT_VOICE_LIVE_API_KEY: str | None = os.getenv("MICROSOFT_VOICE_LIVE_API_KEY")
MICROSOFT_VOICE_LIVE_REGION: str | None = os.getenv("MICROSOFT_VOICE_LIVE_REGION")
MICROSOFT_VOICE_LIVE_VOICE: str | None = os.getenv("MICROSOFT_VOICE_LIVE_VOICE")
MICROSOFT_VOICE_LIVE_BASE_URL: str | None = os.getenv("MICROSOFT_VOICE_LIVE_BASE_URL")

# Optional explicit STT/TTS endpoints. If not set, they can be derived from region.
MICROSOFT_VOICE_LIVE_TTS_URL: str | None = os.getenv("MICROSOFT_VOICE_LIVE_TTS_URL")
MICROSOFT_VOICE_LIVE_STT_URL: str | None = os.getenv("MICROSOFT_VOICE_LIVE_STT_URL")

USE_MICROSOFT_VOICE_LIVE: bool = os.getenv(
    "USE_MICROSOFT_VOICE_LIVE", "false"
).lower() in {"1", "true", "yes"}
