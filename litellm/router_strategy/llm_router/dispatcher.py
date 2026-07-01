"""Dispatcher client abstraction for the LLM router.

The dispatcher is the LLM that picks the best candidate model for each prompt.
It is a small ``Protocol`` so tests can inject a stub instead of monkeypatching
``litellm.acompletion``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from litellm._logging import verbose_router_logger


@runtime_checkable
class DispatcherClient(Protocol):
    async def choose_model(self, context: str, prompt: str) -> str:
        """Return the raw text response of the dispatcher LLM for this prompt.

        Callers are responsible for parsing the returned text into JSON.
        """
        ...


class LiteLLMDispatcher:
    """Default dispatcher that calls ``litellm.acompletion`` with the configured
    dispatcher model. Constructed from ``LLMRouterConfig``."""

    def __init__(
        self,
        dispatcher_model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._dispatcher_model = dispatcher_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def choose_model(self, context: str, prompt: str) -> str:
        import litellm
        from litellm.types.utils import ModelResponse

        response = await litellm.acompletion(
            model=self._dispatcher_model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        if not isinstance(response, ModelResponse):
            raise TypeError(f"LLMRouter dispatcher expected a ModelResponse, got {type(response).__name__}")
        content = response.choices[0].message.content
        verbose_router_logger.debug(
            "LLMRouter dispatcher %s returned: %s",
            self._dispatcher_model,
            content,
        )
        return content or ""
