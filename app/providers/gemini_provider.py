from __future__ import annotations

from typing import Any, Callable, Optional

import httpx

from app.core import config
from app.providers.llm_provider import LLMProvider


class GeminiProviderError(RuntimeError):
    pass


class GeminiProvider(LLMProvider):
    """Minimal Gemini generateContent API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.base_url = (base_url or config.GEMINI_BASE_URL).rstrip("/")
        self.model = self._normalize_model(model or config.GEMINI_MODEL or "google/gemini-flash")

        if not self.api_key:
            raise GeminiProviderError("GEMINI_API_KEY is not set")

    def _normalize_model(self, model: str) -> str:
        key = (model or "").strip().lower()
        mapping = {
            "google/gemini-flash": "gemini-2.0-flash",
            "google/gemini-flash-lite": "gemini-2.0-flash-lite",
        }
        if key in mapping:
            return mapping[key]
        if key.startswith("google/"):
            return key.split("/", 1)[1]
        return model

    async def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }

        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(url, params=params, json=payload)

        if resp.status_code >= 400:
            raise GeminiProviderError(
                f"Gemini request failed (status={resp.status_code}): {resp.text}"
            )

        data = resp.json()
        try:
            candidates = data.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("Missing candidates")
            first = candidates[0] if isinstance(candidates[0], dict) else {}
            content = first.get("content") if isinstance(first, dict) else {}
            parts = content.get("parts") if isinstance(content, dict) else []
            texts: list[str] = []
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str):
                            texts.append(text)
            return "".join(texts)
        except Exception as exc:
            raise GeminiProviderError("Unexpected Gemini response format") from exc

    async def stream(self, prompt: str, on_token: Callable[[str], Any]) -> None:
        text = await self.generate(prompt)
        for chunk in text.split():
            result = on_token(chunk + " ")
            if hasattr(result, "__await__"):
                await result
