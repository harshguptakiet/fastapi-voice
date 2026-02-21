from __future__ import annotations

from pydantic import BaseModel, Field


class TextInteractionRequest(BaseModel):
    """Text-only interaction request.

    The interaction gateway accepts normalized client inputs and returns
    a normalized interaction object without invoking any LLM.
    """

    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, description="User-provided text input")
    language: str | None = Field(None, description="Optional BCP-47 language code")
