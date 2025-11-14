from .model_selector import model_selector

class LLMHandler:

    async def generate_response(self, text: str, model: str = "dummy"):
        provider = model_selector.select(model)
        return await provider.generate(text)

    async def stream_response(self, text: str, model: str, on_token):
        provider = model_selector.select(model)
        return await provider.stream(text, on_token)
