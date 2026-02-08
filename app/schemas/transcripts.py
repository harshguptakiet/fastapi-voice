from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.voice import NormalizedTranscript


class TranscriptNormalizeRequest(BaseModel):
    provider: str | None = Field(None, description="Optional provider hint")
    request_id: str | None = None
    language: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TranscriptNormalizeResponse(BaseModel):
    transcript: NormalizedTranscript
