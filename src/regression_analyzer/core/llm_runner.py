"""Multi-provider LLM runner with model routing."""

import os
from typing import Optional, Type, TypeVar, List

from pydantic import BaseModel

from ..providers import (
    BaseLLMProvider,
    LLMProvider,
    LLMResult,
    ClaudeCodeProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    GroqProvider,
)

T = TypeVar("T", bound=BaseModel)


class LLMRunner:
    """Multi-provider LLM wrapper with model routing.

    Routes prompts to the appropriate provider based on model name or
    default provider configured via environment variables.

    Environment Variables:
        LLM_PROVIDER: Default provider (claude-code, openai, anthropic, google, groq)
        OPENAI_API_KEY: Enables OpenAI provider
        ANTHROPIC_API_KEY: Enables Anthropic provider
        GOOGLE_GEMINI_API_KEY: Enables Google provider
        GROQ_API_KEY: Enables Groq provider
    """

    def __init__(self):
        """Initialize LLM runner and configure providers from environment."""
        self.providers: dict[LLMProvider, BaseLLMProvider] = {}
        self.default_provider = LLMProvider.CLAUDE_CODE
        self._init_providers_from_env()

    def _init_providers_from_env(self):
        """Initialize available providers based on .env configuration."""
        # Always register Claude Code (uses Max subscription, no API key needed)
        self.providers[LLMProvider.CLAUDE_CODE] = ClaudeCodeProvider()

        # Register API-based providers if keys are available
        if os.getenv("OPENAI_API_KEY"):
            self.providers[LLMProvider.OPENAI] = OpenAIProvider(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers[LLMProvider.ANTHROPIC] = AnthropicProvider(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        if os.getenv("GOOGLE_GEMINI_API_KEY"):
            self.providers[LLMProvider.GOOGLE] = GoogleProvider(
                api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
            )

        if os.getenv("GROQ_API_KEY"):
            self.providers[LLMProvider.GROQ] = GroqProvider(
                api_key=os.getenv("GROQ_API_KEY")
            )

        # Override default provider if specified
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            try:
                self.default_provider = LLMProvider(env_provider)
                if self.default_provider not in self.providers:
                    raise ValueError(
                        f"Provider {env_provider} specified but not available. "
                        f"Check API key configuration."
                    )
            except ValueError as e:
                valid_providers = [p.value for p in LLMProvider]
                raise ValueError(
                    f"Invalid LLM_PROVIDER: {env_provider}. "
                    f"Valid options: {valid_providers}"
                ) from e

    def get_provider_for_model(self, model: str) -> BaseLLMProvider:
        """Route model name to appropriate provider."""
        if model.startswith("gpt-"):
            if LLMProvider.OPENAI not in self.providers:
                raise ValueError(
                    f"Model {model} requires OpenAI provider. Set OPENAI_API_KEY."
                )
            return self.providers[LLMProvider.OPENAI]

        elif model.startswith("claude-"):
            if LLMProvider.ANTHROPIC in self.providers:
                return self.providers[LLMProvider.ANTHROPIC]
            return self.providers[LLMProvider.CLAUDE_CODE]

        elif model.startswith("gemini-"):
            if LLMProvider.GOOGLE not in self.providers:
                raise ValueError(
                    f"Model {model} requires Google provider. Set GOOGLE_GEMINI_API_KEY."
                )
            return self.providers[LLMProvider.GOOGLE]

        elif model.startswith("llama") or model.startswith("mixtral"):
            if LLMProvider.GROQ not in self.providers:
                raise ValueError(
                    f"Model {model} requires Groq provider. Set GROQ_API_KEY."
                )
            return self.providers[LLMProvider.GROQ]

        elif model in ["sonnet", "opus"]:
            return self.providers[LLMProvider.CLAUDE_CODE]

        return self.providers[self.default_provider]

    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt using appropriate provider."""
        if model:
            provider = self.get_provider_for_model(model)
        else:
            provider = self.providers[self.default_provider]

        return await provider.run(prompt, model)

    async def run_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> T:
        """Execute prompt and return structured response."""
        if model:
            provider = self.get_provider_for_model(model)
        else:
            provider = self.providers[self.default_provider]

        return await provider.run_structured(prompt, response_model, model)

    async def run_structured_list(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> List[T]:
        """Execute prompt and return list of structured responses."""
        if model:
            provider = self.get_provider_for_model(model)
        else:
            provider = self.providers[self.default_provider]

        return await provider.run_structured_list(prompt, response_model, model)

    def get_available_models(self) -> list[str]:
        """Return all available models across all configured providers."""
        models = []
        for provider in self.providers.values():
            models.extend(provider.get_available_models())
        return models

    def get_available_providers(self) -> list[str]:
        """Return list of configured provider names."""
        return [p.value for p in self.providers.keys()]

    def get_default_provider(self) -> str:
        """Return the default provider name."""
        return self.default_provider.value
