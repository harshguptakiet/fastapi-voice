from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    provider: str | None = Field(None, description="Override LLM provider")
    llm_model: str | None = Field(None, description="Override model name")


class LLMGenerateResponse(BaseModel):
    provider: str = Field(..., description="Selected provider")
    model: str | None = Field(None, description="Selected model")
    text: str = Field("", description="LLM output")
    raw: dict[str, Any] | None = Field(None, description="Optional provider raw payload")


class LLMHealthResponse(BaseModel):
    status: Literal["ok", "unhealthy"]
    provider: str


class LLMModelInfo(BaseModel):
    id: str
    label: str | None = None


class LLMModelsResponse(BaseModel):
    provider: str
    models: list[LLMModelInfo] = Field(default_factory=list)
