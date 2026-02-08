from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.schemas.sessions import (
    SessionAddMessageRequest,
    SessionCreateResponse,
    SessionMessagesResponse,
    SessionStateResponse,
)
from app.services.context_service import context


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    session_id = str(uuid.uuid4())
    context.set(session_id, {})
    return SessionCreateResponse(session_id=session_id)


@router.get("/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str) -> SessionStateResponse:
    state = context.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateResponse(session_id=session_id, state=state)


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    context.reset(session_id)
    return {"status": "ok"}


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
async def list_messages(session_id: str) -> SessionMessagesResponse:
    messages = context.get_messages(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionMessagesResponse(session_id=session_id, messages=messages)


@router.post("/{session_id}/messages", response_model=SessionMessagesResponse)
async def add_message(session_id: str, body: SessionAddMessageRequest) -> SessionMessagesResponse:
    ok = context.add_message(session_id, role=body.role, content=body.content)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionMessagesResponse(session_id=session_id, messages=context.get_messages(session_id) or [])
