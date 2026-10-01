"""Thin abstraction over the Anthropic Messages API with forced tool-use for
structured output, retry/backoff, and basic cost/latency observability.

Isolating the raw API call here (rather than scattering it through the agent
logic) means the provider could be swapped or a fallback model added without
touching any business logic in ComplianceAgent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class LLMProviderError(Exception):
    """Raised when the LLM provider fails to produce a usable structured response."""


@dataclass
class StructuredLLMResult:
    data: Dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float


class AnthropicProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.anthropic_model
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)),
    )
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        tool_description: str,
        tool_schema: Dict[str, Any],
        max_tokens: int = 1024,
    ) -> StructuredLLMResult:
        start = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": tool_name,
                        "description": tool_description,
                        "input_schema": tool_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(f"Anthropic API returned an error: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000

        tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use_block is None:
            raise LLMProviderError("Model did not return the expected tool_use block.")

        usage = response.usage
        logger.info(
            "LLM call complete: model=%s input_tokens=%d output_tokens=%d latency_ms=%.1f",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            latency_ms,
        )

        return StructuredLLMResult(
            data=tool_use_block.input,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        )