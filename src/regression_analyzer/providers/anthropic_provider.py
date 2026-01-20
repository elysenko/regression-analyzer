"""Anthropic API provider with instructor support for structured outputs."""

from typing import Optional, Type, TypeVar, List

from pydantic import BaseModel

from .base import BaseLLMProvider, LLMProvider, LLMResult

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""

    AVAILABLE_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229"
    ]
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize Anthropic provider."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self._default_model = model or self.DEFAULT_MODEL

        self._instructor_client = None
        try:
            import instructor
            self._instructor_client = instructor.from_anthropic(self.client)
        except ImportError:
            pass

    @property
    def provider_type(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC

    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt via Anthropic API."""
        model = model or self._default_model

        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }

        return LLMResult(
            content=content,
            model=model,
            provider=self.provider_type,
            usage=usage,
            raw_response=response.model_dump()
        )

    def get_available_models(self) -> list[str]:
        return self.AVAILABLE_MODELS.copy()

    def get_default_model(self) -> str:
        return self._default_model

    async def run_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> T:
        """Execute prompt and return structured response using instructor."""
        if self._instructor_client is None:
            return await super().run_structured(prompt, response_model, model)

        model = model or self._default_model

        response = await self._instructor_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
        )

        return response

    async def run_structured_list(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> List[T]:
        """Execute prompt and return list of structured responses using instructor."""
        if self._instructor_client is None:
            return await super().run_structured_list(prompt, response_model, model)

        model = model or self._default_model

        response = await self._instructor_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            response_model=List[response_model],
        )

        return response if response else []
