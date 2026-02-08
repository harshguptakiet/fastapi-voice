from __future__ import annotations

from fastapi import APIRouter

from app.schemas.interaction import NormalizedInteractionInput
from app.schemas.interaction_request import TextInteractionRequest


router = APIRouter(prefix="/interactions", tags=["interactions"])


@router.post("", response_model=NormalizedInteractionInput)
async def create_interaction(body: TextInteractionRequest) -> NormalizedInteractionInput:
    # Boundary endpoint: normalize input and return it.
    return NormalizedInteractionInput(
        session_id=body.session_id,
        input_type="text",
        raw_input_ref=None,
        normalized_text=body.text.strip(),
        language=body.language,
    )
