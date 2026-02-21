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

# Anthropic configuration
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL: str | None = os.getenv("ANTHROPIC_MODEL")

# Gemini configuration
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
GEMINI_MODEL: str | None = os.getenv("GEMINI_MODEL")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Invalid database connection string")

# Object storage configuration
OBJECT_STORAGE_PROVIDER: str = os.getenv("OBJECT_STORAGE_PROVIDER", "local")
OBJECT_STORAGE_BUCKET: str | None = os.getenv("OBJECT_STORAGE_BUCKET")
OBJECT_STORAGE_REGION: str | None = os.getenv("OBJECT_STORAGE_REGION")
OBJECT_STORAGE_ENDPOINT_URL: str | None = os.getenv("OBJECT_STORAGE_ENDPOINT_URL")
OBJECT_STORAGE_ACCESS_KEY: str | None = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
OBJECT_STORAGE_SECRET_KEY: str | None = os.getenv("OBJECT_STORAGE_SECRET_KEY")
LOCAL_DOCUMENT_STORAGE_DIR: str = os.getenv("LOCAL_DOCUMENT_STORAGE_DIR", "storage/documents")

# Deepgram (STT) + ElevenLabs (TTS) configuration.
DEEPGRAM_API_KEY: str | None = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_STT_URL: str = os.getenv("DEEPGRAM_STT_URL", "https://api.deepgram.com/v1/listen")
DEEPGRAM_MODEL: str = os.getenv("DEEPGRAM_MODEL", "nova-2")
DEEPGRAM_LANGUAGE: str = os.getenv("DEEPGRAM_LANGUAGE", "en-US")

ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_BASE_URL: str = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io/v1")
ELEVENLABS_VOICE_ID: str | None = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

USE_DEEPGRAM_ELEVENLABS: bool = os.getenv(
    "USE_DEEPGRAM_ELEVENLABS", "false"
).lower() in {"1", "true", "yes"}
