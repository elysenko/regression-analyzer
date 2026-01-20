"""Claude Code CLI provider - uses Claude Max subscription."""

import asyncio
import json
from typing import Optional, List, Dict, Any

from .base import BaseLLMProvider, LLMProvider, LLMResult


class ClaudeCodeProvider(BaseLLMProvider):
    """Provider using Claude Code CLI (leverages Claude Max subscription)."""

    AVAILABLE_MODELS = ["sonnet", "opus"]
    DEFAULT_MODEL = "sonnet"

    def __init__(self, model: Optional[str] = None):
        """Initialize Claude Code provider."""
        self._default_model = model or self.DEFAULT_MODEL

    @property
    def provider_type(self) -> LLMProvider:
        return LLMProvider.CLAUDE_CODE

    async def run(self, prompt: str, model: Optional[str] = None) -> LLMResult:
        """Execute prompt via Claude Code CLI."""
        model = model or self._default_model

        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            "--model", model
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Claude Code CLI failed: {error_msg}")

        try:
            data = json.loads(stdout.decode())
            return LLMResult(
                content=data.get("result", ""),
                model=model,
                provider=self.provider_type,
                usage=data.get("usage", {}),
                raw_response=data
            )
        except json.JSONDecodeError:
            return LLMResult(
                content=stdout.decode(),
                model=model,
                provider=self.provider_type
            )

    def get_available_models(self) -> list[str]:
        """Return available Claude models."""
        return self.AVAILABLE_MODELS.copy()

    def get_default_model(self) -> str:
        """Return the default model."""
        return self._default_model
