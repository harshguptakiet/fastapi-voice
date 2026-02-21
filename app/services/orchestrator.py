from __future__ import annotations

import logging
from typing import Any, Optional

from app.schemas.interaction import NormalizedInteractionInput
from app.services.context_service import context
from app.services.llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class ConversationOrchestrator:
    def __init__(self, llm_handler: Optional[LLMHandler] = None):
        self.llm_handler = llm_handler or LLMHandler()

    async def process_interaction(
        self, 
        interaction: NormalizedInteractionInput,
        provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> str:
        session_id = interaction.session_id
        text = interaction.normalized_text
        
        # 1. Ensure session exists
        if not context.exists(session_id):
            context.set(session_id, {})
        
        # 2. Detect basic intents
        intent = self._detect_intent(text)
        logger.info(f"Detected intent: {intent} for session {session_id}")
        
        # 3. Track session state
        context.update_state(session_id, "language", interaction.language or "en")
        
        # 4. Add user message to history
        context.add_message(session_id, role="user", content=text)
        
        # 5. Get conversation context
        history = context.get_messages(session_id) or []
        
        # 6. Generate response logic (AI Reasoning Logic)
        # In a real scenario, we might use the intent to branch logic.
        # For now, we use the LLM to generate the final response.
        
        # Format a prompt with history
        prompt = self._build_prompt(history, session_id)
        
        response_text = await self.llm_handler.generate_response(
            prompt,
            provider=provider,
            llm_model=llm_model
        )
        
        # 7. Update state with last response and inferred topic (simple)
        context.update_state(session_id, "last_response", response_text)
        if len(text.split()) > 3:
            # Very simple topic inference
            context.update_state(session_id, "current_topic", text[:30] + "...")
        
        # 8. Add assistant message to history
        context.add_message(session_id, role="assistant", content=response_text)
        
        return response_text

    def _detect_intent(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["help", "what can you do", "support"]):
            return "help"
        if any(w in text for w in ["stop", "cancel", "bye", "exit"]):
            return "exit"
        if "?" in text or any(text.startswith(w) for w in ["who", "what", "where", "when", "why", "how"]):
            return "question"
        return "statement"

    def _build_prompt(self, history: list[dict[str, Any]], session_id: str) -> str:
        # Get persona from state
        sess = context.get(session_id) or {}
        persona = sess.get("persona", "default")
        
        system_prompt = f"You are a helpful assistant. Persona: {persona}. "
        if sess.get("current_topic"):
            system_prompt += f"Current topic is {sess.get('current_topic')}. "
            
        full_prompt = system_prompt + "\n\n"
        for msg in history[-5:]: # Last 5 messages for context
            full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        full_prompt += "Assistant:"
        return full_prompt

orchestrator = ConversationOrchestrator()
