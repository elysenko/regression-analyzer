"""OpenAI API provider."""

from typing import Optional

from .base import BaseLLMProvider, LLMProvider, LLMResult


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize OpenAI provider."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._default_model = model or self.DEFAULT_MODEL

    @property
    def provider_type(self) -> LLMProvider:
        return LLMProvider.OPENAI

    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt via OpenAI API."""
        model = model or self._default_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return LLMResult(
            content=choice.message.content or "",
            model=model,
            provider=self.provider_type,
            usage=usage,
            raw_response=response.model_dump()
        )

    def get_available_models(self) -> list[str]:
        return self.AVAILABLE_MODELS.copy()

    def get_default_model(self) -> str:
        return self._default_model
