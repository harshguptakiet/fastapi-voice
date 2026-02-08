from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionStateResponse(BaseModel):
    session_id: str
    state: dict = Field(default_factory=dict)


class SessionMessage(BaseModel):
    role: str = Field(..., description="user|assistant|system")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionAddMessageRequest(BaseModel):
    role: str = Field(..., description="user|assistant|system")
    content: str = Field(..., min_length=1)
