from __future__ import annotations

from app.core import config
from .model_selector import model_selector


class LLMHandler:
    def _select_provider(
        self,
        provider: str | None,
        *,
        llm_model: str | None,
        legacy_model: str | None,
    ):
        # Back-compat: `model` used to mean provider name.
        provider_name = (provider or legacy_model or config.LLM_PROVIDER)
        model_name = (llm_model or config.LLM_MODEL)
        return model_selector.select(provider_name, model_name)

    async def generate_response(
        self,
        text: str,
        provider: str | None = None,
        *,
        llm_model: str | None = None,
        model: str | None = None,
    ):
        selected = self._select_provider(
            provider,
            llm_model=llm_model,
            legacy_model=model,
        )
        return await selected.generate(text)

    async def stream_response(
        self,
        text: str,
        provider: str | None = None,
        on_token=None,
        *,
        llm_model: str | None = None,
        model: str | None = None,
    ):
        selected = self._select_provider(
            provider,
            llm_model=llm_model,
            legacy_model=model,
        )
        return await selected.stream(text, on_token)
