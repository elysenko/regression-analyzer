"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeVar, Type
import json

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE_CODE = "claude-code"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"


@dataclass
class LLMResult:
    """Result from an LLM call."""
    content: str
    model: Optional[str] = None
    provider: Optional[LLMProvider] = None
    usage: dict = field(default_factory=dict)
    raw_response: Optional[dict] = None
    tool_calls: list = field(default_factory=list)
    files_written: list = field(default_factory=list)

    @classmethod
    def from_json(cls, json_str: str) -> "LLMResult":
        """Parse LLMResult from JSON string."""
        data = json.loads(json_str)
        return cls(
            content=data.get("result", data.get("content", "")),
            model=data.get("model"),
            usage=data.get("usage", {}),
            raw_response=data
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider.value if self.provider else None,
            "usage": self.usage,
            "tool_calls": self.tool_calls,
            "files_written": self.files_written
        }


def _clean_json_keys(obj):
    """Recursively clean JSON object keys by stripping embedded quote characters."""
    if isinstance(obj, dict):
        return {
            k.strip('"\'').strip(): _clean_json_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_clean_json_keys(item) for item in obj]
    return obj


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @property
    @abstractmethod
    def provider_type(self) -> LLMProvider:
        """Return the provider type enum."""
        pass

    @abstractmethod
    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt and return result."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Return list of available models for this provider."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    async def run_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> T:
        """Execute prompt and return structured response."""
        try:
            from llm_output_parser import parse_json
        except ImportError:
            import json as json_module
            def parse_json(text, **kwargs):
                # Simple fallback: extract JSON from text
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json_module.loads(text[start:end])
                return None

        result = await self.run(prompt, model)

        try:
            data = parse_json(result.content, strict=False, allow_incomplete=True)
            if data is None:
                raise ValueError("Failed to parse JSON from response")

            data = _clean_json_keys(data)

            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            return response_model.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to parse response into {response_model.__name__}: {e}") from e

    async def run_structured_list(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> list[T]:
        """Execute prompt and return list of structured responses."""
        try:
            from llm_output_parser import parse_json
        except ImportError:
            import json as json_module
            def parse_json(text, **kwargs):
                start = text.find('[')
                end = text.rfind(']') + 1
                if start >= 0 and end > start:
                    return json_module.loads(text[start:end])
                return None

        result = await self.run(prompt, model)

        try:
            data = parse_json(result.content, strict=False, allow_incomplete=True)
            if data is None:
                return []

            data = _clean_json_keys(data)

            if isinstance(data, dict):
                data = [data]

            return [response_model.model_validate(item) for item in data]
        except Exception as e:
            raise ValueError(f"Failed to parse response into list of {response_model.__name__}: {e}") from e
