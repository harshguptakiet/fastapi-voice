from __future__ import annotations

from fastapi import APIRouter

from app.schemas.interaction import NormalizedInteractionInput
from app.schemas.interaction_request import TextInteractionRequest


from app.services.orchestrator import orchestrator


router = APIRouter(prefix="/interactions", tags=["interactions"])


@router.post("", response_model=dict)
async def create_interaction(body: TextInteractionRequest) -> dict:
    # 1. Create normalized input
    interaction = NormalizedInteractionInput(
        session_id=body.session_id,
        input_type="text",
        raw_input_ref=None,
        normalized_text=body.text.strip(),
        language=body.language,
    )

    # 2. Process with orchestrator
    response_text = await orchestrator.process_interaction(interaction)

    # 3. Return both the normalization and the response
    return {
        "interaction": interaction.model_dump(),
        "response_text": response_text
    }

