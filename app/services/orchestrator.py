from __future__ import annotations

import logging
import re
from typing import Any, Optional

from app.schemas.interaction import NormalizedInteractionInput
from app.services.context_service import context
from app.services.llm_handler import LLMHandler

logger = logging.getLogger(__name__)

VOICE_AGENT_POLICY = (
    "You are a real-time conversational voice assistant designed for EdTech applications. "
    "Your purpose is to provide fast, clear, spoken-style responses optimized for text-to-speech output. "
    "CRITICAL RESPONSE RULES: Keep responses between 1–3 short sentences. Maximum 120 tokens. "
    "Prefer simple, natural spoken language. Use proper punctuation for natural speech rhythm. "
    "Avoid long explanations unless explicitly requested. Do not use markdown, formatting symbols, emojis, bullet points, or special characters. "
    "Do not repeat the user's question. Do not add disclaimers. Do not mention being an AI. "
    "Avoid filler phrases like Sure, Of course, or That's a great question. Avoid over-explaining. End responses naturally. "
    "EMOTION CONTROL: First decide one emotion for delivery using one of: calm, excited, empathetic, confident, cheerful, serious, reassuring, playful, urgent. "
    "Generate one internal emotion label on the first line in this exact form: [emotion: <type>]. "
    "Then provide the spoken response text. "
    "EDTECH MODE BEHAVIOR: If explaining a concept, break it into simple, digestible parts. Prefer clarity over depth. "
    "Encourage learning without sounding overly formal. If the student seems confused, use reassuring tone. "
    "If answering a factual question, be confident and precise. If asked for deeper explanation, expand but stay under 3 sentences unless explicitly asked to elaborate."
)

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
        emotion, cleaned_response = self._extract_emotion_and_clean(response_text)
        
        # 7. Update state with last response and inferred topic (simple)
        context.update_state(session_id, "last_response", cleaned_response)
        context.update_state(session_id, "last_emotion", emotion or "calm")
        if len(text.split()) > 3:
            # Very simple topic inference
            context.update_state(session_id, "current_topic", text[:30] + "...")
        
        # 8. Add assistant message to history
        context.add_message(session_id, role="assistant", content=cleaned_response)
        
        return cleaned_response

    def get_last_emotion(self, session_id: str) -> str:
        sess = context.get(session_id) or {}
        emotion = sess.get("last_emotion")
        if isinstance(emotion, str) and emotion.strip():
            return emotion.strip().lower()
        return "calm"

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
        
        system_prompt = f"{VOICE_AGENT_POLICY} Persona: {persona}. "
        if sess.get("current_topic"):
            system_prompt += f"Current topic is {sess.get('current_topic')}. "
            
        full_prompt = system_prompt + "\n\n"
        for msg in history[-5:]: # Last 5 messages for context
            full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        full_prompt += "Assistant:"
        return full_prompt

    def _extract_emotion_and_clean(self, text: str) -> tuple[str | None, str]:
        if not isinstance(text, str):
            return (None, "")
        raw = text.strip()
        m = re.match(r"^\[emotion:\s*([a-zA-Z-]+)\]\s*", raw, flags=re.IGNORECASE)
        if not m:
            return (None, raw)
        emotion = m.group(1).strip().lower()
        cleaned = raw[m.end():].strip()
        return (emotion, cleaned)

orchestrator = ConversationOrchestrator()
