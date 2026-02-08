from __future__ import annotations

from typing import Any, Callable

from app.providers.llm_provider import LLMProvider
from app.providers.offline_provider import OfflineProvider
from app.providers.openai_provider import OpenAIProvider


class DummyProvider(LLMProvider):
    """Simple local provider for testing."""

    async def generate(self, prompt: str) -> str:
        return f"Dummy response for: {prompt}"

    async def stream(self, prompt: str, on_token: Callable[[str], Any]):
        for ch in "dummy stream response":
            result = on_token(ch)
            if hasattr(result, "__await__"):
                await result  # support async callbacks


class ModelSelector:
    def select(self, provider: str, model: str | None = None) -> LLMProvider:
        """Return an LLMProvider by provider name."""

        key = (provider or "dummy").strip().lower()

        if key in {"dummy", "test"}:
            return DummyProvider()
        if key in {"openai"}:
            return OpenAIProvider(model=model)
        if key in {"offline", "local"}:
            return OfflineProvider()

        raise ValueError(f"Unsupported LLM provider: {provider}")


model_selector = ModelSelector()
