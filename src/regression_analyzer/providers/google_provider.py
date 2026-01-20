"""Google Gemini API provider."""

from typing import Optional

from .base import BaseLLMProvider, LLMProvider, LLMResult


class GoogleProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    AVAILABLE_MODELS = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro"
    ]
    DEFAULT_MODEL = "gemini-1.5-pro-latest"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize Google Gemini provider."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

        genai.configure(api_key=api_key)
        self._genai = genai
        self._default_model = model or self.DEFAULT_MODEL

    @property
    def provider_type(self) -> LLMProvider:
        return LLMProvider.GOOGLE

    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt via Google Gemini API."""
        model_name = model or self._default_model

        import asyncio
        loop = asyncio.get_event_loop()

        def _generate():
            gen_model = self._genai.GenerativeModel(model_name)
            response = gen_model.generate_content(prompt)
            return response

        response = await loop.run_in_executor(None, _generate)

        usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
            }

        return LLMResult(
            content=response.text if response.text else "",
            model=model_name,
            provider=self.provider_type,
            usage=usage
        )

    def get_available_models(self) -> list[str]:
        return self.AVAILABLE_MODELS.copy()

    def get_default_model(self) -> str:
        return self._default_model
