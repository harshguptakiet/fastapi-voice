from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core import config
from app.dependencies import get_llm_provider
from app.providers.llm_provider import LLMProvider
from app.schemas.llm import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMHealthResponse,
    LLMModelsResponse,
)
from app.services.model_selector import model_selector


router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/health", response_model=LLMHealthResponse)
async def llm_health(provider: LLMProvider = Depends(get_llm_provider)) -> LLMHealthResponse:
    # We can't do a full upstream health check generically.
    # If provider instantiation succeeded, report ok.
    return LLMHealthResponse(status="ok", provider=provider.__class__.__name__)


@router.get("/models", response_model=LLMModelsResponse)
async def llm_models() -> LLMModelsResponse:
    # Expose supported providers/adapters.
    return LLMModelsResponse(
        provider="configured",
        models=model_selector.list_supported_providers(),
    )


@router.post("/generate", response_model=LLMGenerateResponse)
async def llm_generate(body: LLMGenerateRequest) -> LLMGenerateResponse:
    try:
        provider_name = body.provider or config.LLM_PROVIDER
        provider_id = model_selector.normalize_provider(provider_name)
        selected = model_selector.select(provider_name or "", body.llm_model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    text = await selected.generate(body.prompt)
    return LLMGenerateResponse(
        provider=provider_id,
        model=body.llm_model,
        text=text,
        raw=None,
    )


@router.post("/stream")
async def llm_stream(body: LLMGenerateRequest) -> StreamingResponse:
    try:
        provider_name = body.provider or config.LLM_PROVIDER
        selected = model_selector.select(provider_name, body.llm_model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_token(tok: str) -> Any:
        await queue.put(tok)

    async def run_stream() -> None:
        try:
            await selected.stream(body.prompt, on_token)
        finally:
            await queue.put(None)

    async def event_iter() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run_stream())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {item}\n\n".encode("utf-8")
        finally:
            task.cancel()

    return StreamingResponse(event_iter(), media_type="text/event-stream")
