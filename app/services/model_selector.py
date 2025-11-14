from typing import Awaitable, Callable, Any
from app.providers.llm_provider import LLMProvider

class DummyProvider(LLMProvider):
    """Temporary placeholder until a real API is selected"""

    async def generate(self, prompt: str) -> str:
        return f"Dummy response for: {prompt}"

    async def stream(self, prompt: str, on_token: Callable[[str], Any]):
        for ch in "dummy stream response":
            result = on_token(ch)
            if hasattr(result, "__await__"):
                await result  # support async callbacks


class ModelSelector:

    def select(self, model: str) -> LLMProvider:
        # Only dummy for now
        return DummyProvider()


model_selector = ModelSelector()
