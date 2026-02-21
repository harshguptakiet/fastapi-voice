import os

from dotenv import load_dotenv


# Load environment variables from .env if present
load_dotenv()


# Minimal app configuration
APP_NAME = os.getenv("APP_NAME", "Bot Backend")
ENV = os.getenv("ENV", "dev")

# LLM configuration (provider-agnostic)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "dummy").strip() or "dummy"
LLM_MODEL: str | None = (os.getenv("LLM_MODEL") or None)

# Optional OpenAI-style configuration (used only by the OpenAI provider adapter)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Invalid database connection string")

# Microsoft Voice Live configuration (optional)
MICROSOFT_VOICE_LIVE_API_KEY: str | None = os.getenv("MICROSOFT_VOICE_LIVE_API_KEY")
MICROSOFT_VOICE_LIVE_REGION: str | None = os.getenv("MICROSOFT_VOICE_LIVE_REGION")
MICROSOFT_VOICE_LIVE_VOICE: str | None = os.getenv("MICROSOFT_VOICE_LIVE_VOICE")
MICROSOFT_VOICE_LIVE_BASE_URL: str | None = os.getenv("MICROSOFT_VOICE_LIVE_BASE_URL")

# Required only for /voice/stream when enabled.
MICROSOFT_VOICE_LIVE_REALTIME_API_VERSION: str | None = os.getenv(
    "MICROSOFT_VOICE_LIVE_REALTIME_API_VERSION"
)

MICROSOFT_VOICE_LIVE_REALTIME_MODEL: str = os.getenv(
    "MICROSOFT_VOICE_LIVE_REALTIME_MODEL", "gpt-4.1"
)

# Prefer header auth; enable query auth only if required.
MICROSOFT_VOICE_LIVE_AUTH_IN_QUERY: bool = os.getenv(
    "MICROSOFT_VOICE_LIVE_AUTH_IN_QUERY", "false"
).lower() in {"1", "true", "yes"}

# Voice websocket bridge toggle.
ENABLE_VOICE_STREAM_WS: bool = os.getenv(
    "ENABLE_VOICE_STREAM_WS", "false"
).lower() in {"1", "true", "yes"}

# Optional explicit STT/TTS endpoints. If not set, they can be derived from region.
MICROSOFT_VOICE_LIVE_TTS_URL: str | None = os.getenv("MICROSOFT_VOICE_LIVE_TTS_URL")
MICROSOFT_VOICE_LIVE_STT_URL: str | None = os.getenv("MICROSOFT_VOICE_LIVE_STT_URL")

USE_MICROSOFT_VOICE_LIVE: bool = os.getenv(
    "USE_MICROSOFT_VOICE_LIVE", "false"
).lower() in {"1", "true", "yes"}
