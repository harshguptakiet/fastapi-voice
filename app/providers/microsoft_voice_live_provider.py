from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from app.core import config
from app.providers.llm_provider import LLMProvider


class MicrosoftVoiceLiveError(RuntimeError):
    """Raised when a Microsoft Voice Live API call fails."""


class MicrosoftVoiceLiveProvider(LLMProvider):
    """LLMProvider backed by Microsoft Voice Live.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        # Configuration comes from .env through app.core.config
        self.api_key = api_key or config.MICROSOFT_VOICE_LIVE_API_KEY
        self.region = region or config.MICROSOFT_VOICE_LIVE_REGION
        self.base_url = base_url or config.MICROSOFT_VOICE_LIVE_BASE_URL

        if not self.api_key:
            raise MicrosoftVoiceLiveError("MICROSOFT_VOICE_LIVE_API_KEY is not set")
        if not self.base_url:
            raise MicrosoftVoiceLiveError(
                "MICROSOFT_VOICE_LIVE_BASE_URL is not set; configure it in .env"
            )

    # ------------------------------------------------------------------
    # LLMProvider interface methods (required by the rest of the app)
    # ------------------------------------------------------------------

    async def generate(self, prompt: str) -> str:
      
        """Generate a text response for the given prompt."""

        raise MicrosoftVoiceLiveError(
            "Text generation is not configured for MicrosoftVoiceLiveProvider. "
            "Use voice-specific methods instead when they are implemented."
        )

    async def stream(self, prompt: str, on_token: Callable[[str], Any]) -> None:
        """Stream tokens for a text response."""

        raise MicrosoftVoiceLiveError(
            "Streaming text is not configured for MicrosoftVoiceLiveProvider. "
            "Use voice-specific methods instead when they are implemented."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return basic auth headers."""

        return {
            # TODO: confirm the exact header name required by Voice Live
            "Ocp-Apim-Subscription-Key": self.api_key,
        }

    async def _simple_health_check(self) -> bool:
        """Ping the base URL to check connectivity."""

        url = self.base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            raise MicrosoftVoiceLiveError(f"Health check failed: {exc}") from exc

        # Many services will return 401/403 here, which is fine: it proves
        # the endpoint is reachable even if auth is not yet fully wired.
        return resp.status_code < 500
