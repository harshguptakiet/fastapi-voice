class ContextService:

    def __init__(self):
        self.sessions = {}

    def get(self, session_id: str):
        return self.sessions.get(session_id, {})

    def set(self, session_id: str, data: dict):
        self.sessions[session_id] = data

    def reset(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]


context = ContextService()
