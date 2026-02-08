from __future__ import annotations

from datetime import datetime
from typing import Any


class ContextService:

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        # Ensure session container always exists.
        existing = self.sessions.get(session_id) or {}
        existing.update(data)
        existing.setdefault("messages", [])
        self.sessions[session_id] = existing

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def get_messages(self, session_id: str):
        sess = self.sessions.get(session_id)
        if not sess:
            return None
        return list(sess.get("messages") or [])

    def add_message(self, session_id: str, *, role: str, content: str) -> bool:
        sess = self.sessions.get(session_id)
        if not sess:
            return False

        messages = sess.setdefault("messages", [])
        messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
            }
        )
        return True


context = ContextService()
