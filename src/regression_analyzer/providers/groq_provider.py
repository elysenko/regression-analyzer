"""Groq API provider."""

from typing import Optional

from .base import BaseLLMProvider, LLMProvider, LLMResult


class GroqProvider(BaseLLMProvider):
    """Groq API provider (fast inference for open models)."""

    AVAILABLE_MODELS = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]
    DEFAULT_MODEL = "llama-3.1-70b-versatile"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize Groq provider."""
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError("groq package required. Install with: pip install groq")

        self.client = AsyncGroq(api_key=api_key)
        self._default_model = model or self.DEFAULT_MODEL

    @property
    def provider_type(self) -> LLMProvider:
        return LLMProvider.GROQ

    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt via Groq API."""
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
