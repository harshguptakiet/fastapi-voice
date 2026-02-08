from __future__ import annotations

from app.core import config


def validate_configuration() -> None:
    """Fail fast on invalid or incomplete configuration."""

    llm_provider = (config.LLM_PROVIDER or "").strip().lower()
    llm_model = (config.LLM_MODEL or None)

    if llm_provider not in {"dummy", "test", "openai", "offline", "local"}:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")

    if llm_provider == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        # model optional; OpenAI provider will default if missing.

    if llm_provider in {"offline", "local"}:
        # Explicitly fail until an offline adapter is implemented.
        raise RuntimeError(
            "LLM_PROVIDER=offline is not implemented yet. "
            "Select LLM_PROVIDER=dummy or openai."
        )

    if config.USE_MICROSOFT_VOICE_LIVE:
        if not config.MICROSOFT_VOICE_LIVE_API_KEY:
            raise RuntimeError(
                "USE_MICROSOFT_VOICE_LIVE=true requires MICROSOFT_VOICE_LIVE_API_KEY"
            )
        if not config.MICROSOFT_VOICE_LIVE_STT_URL:
            raise RuntimeError(
                "USE_MICROSOFT_VOICE_LIVE=true requires MICROSOFT_VOICE_LIVE_STT_URL"
            )

        if config.ENABLE_VOICE_STREAM_WS and not config.MICROSOFT_VOICE_LIVE_BASE_URL:
            raise RuntimeError(
                "ENABLE_VOICE_STREAM_WS=true requires MICROSOFT_VOICE_LIVE_BASE_URL"
            )

        if config.ENABLE_VOICE_STREAM_WS and not config.MICROSOFT_VOICE_LIVE_REALTIME_API_VERSION:
            raise RuntimeError(
                "ENABLE_VOICE_STREAM_WS=true requires MICROSOFT_VOICE_LIVE_REALTIME_API_VERSION"
            )

    # Silence unused variable warnings while keeping intent explicit.
    _ = llm_model
