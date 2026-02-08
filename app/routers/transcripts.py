from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.schemas.transcripts import TranscriptNormalizeRequest, TranscriptNormalizeResponse
from app.schemas.voice import NormalizedTranscript


router = APIRouter(prefix="/transcripts", tags=["transcripts"])


@router.post("/normalize", response_model=TranscriptNormalizeResponse)
async def normalize_transcript(body: TranscriptNormalizeRequest) -> TranscriptNormalizeResponse:
    # Generic normalizer that understands a few common speech payload shapes.
    raw = body.raw or {}
    provider = (body.provider or raw.get("provider") or "unknown").strip() or "unknown"
    rid = body.request_id or raw.get("request_id") or str(uuid.uuid4())
    language = body.language or raw.get("language")

    text = (
        raw.get("DisplayText")
        or raw.get("displayText")
        or raw.get("text")
        or raw.get("Text")
        or ""
    )
    if isinstance(text, str):
        text = text.strip()
    else:
        text = ""

    transcript = NormalizedTranscript(
        request_id=rid,
        provider=provider,
        text=text,
        language=language,
        confidence=None,
        segments=[],
        raw=raw,
    )

    return TranscriptNormalizeResponse(transcript=transcript)
