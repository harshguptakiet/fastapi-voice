from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class NormalizedInteractionInput(BaseModel):
    """Normalized interaction input.

    This is the stable contract between clients (voice/text) and the
    conversation/orchestration layer.
    """

    session_id: str = Field(..., description="Client/session correlation id")
    input_type: Literal["voice", "text"]

    # Pointer to original input (e.g., blob url, recording id, request id)
    raw_input_ref: Optional[str] = None

    # Text to feed the orchestrator/LLM layer
    normalized_text: str = Field("", description="Normalized text representation")

    language: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
