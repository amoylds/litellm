"""LLM router strategy.

For each incoming prompt, ask a configurable "dispatcher" LLM to choose the
best model from the candidate pool, informed by per-model capability metadata
and a quality-vs-speed preference. Falls back to a deterministic heuristic when
no dispatcher is configured or when dispatch fails. A TTL decision cache avoids
re-querying the dispatcher for repeated identical prompts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
)
from litellm.router_strategy.llm_router.config import (
    DECISION_CACHE_SWEEP_THRESHOLD,
    DISPATCHER_PROMPT_TEMPLATE,
    LLM_ROUTER_DECISION_KEY,
)
from litellm.router_strategy.llm_router.dispatcher import (
    DispatcherClient,
    LiteLLMDispatcher,
)
from litellm.router_strategy.llm_router.heuristic import pick_model_heuristic
from litellm.router_strategy.llm_router.prompt_analysis import (
    PromptAnalysis,
    analyze_prompt,
    is_json_object_response_format,
    messages_have_images,
    tools_requested,
)
from litellm.router_strategy.llm_router.scoring import pick_model_prompt_aware
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import LLMRouterCapabilities, LLMRouterConfig

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


def _format_capabilities(caps: LLMRouterCapabilities) -> str:
    parts = tuple(
        s
        for s in (
            f"quality_score={caps.quality_score}" if caps.quality_score is not None else "",
            f"speed_score={caps.speed_score}" if caps.speed_score is not None else "",
            f"context_window={caps.context_window}" if caps.context_window is not None else "",
            f"strengths={','.join(caps.strengths)}" if caps.strengths else "",
        )
        if s
    )
    return ", ".join(parts) if parts else "no declared capabilities"


class _DispatchResult(BaseModel):
    model: str
    reasoning: str = ""


def _strip_json_fence(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith("```"):
        return trimmed
    end = 3
    while end < len(trimmed) and trimmed[end] == "`":
        end += 1
    trimmed = trimmed[end:]
    if trimmed[:4].lower() == "json":
        trimmed = trimmed[4:]
    rstripped = trimmed.rstrip()
    if rstripped.endswith("```"):
        trimmed = rstripped[: rstripped.rfind("```")]
    return trimmed.strip()


def _parse_dispatch_response(raw: str) -> str | None:
    if not raw:
        return None
    cleaned = _strip_json_fence(raw)
    try:
        result = _DispatchResult.model_validate_json(cleaned)
    except (ValueError, TypeError):
        return None
    return result.model.strip() or None


def _build_candidate_context(
    candidates: Mapping[str, LLMRouterCapabilities],
    cost_map: Mapping[str, float],
    quality_preference: float,
) -> str:
    lines = tuple(
        f"- {name}: {_format_capabilities(caps)}"
        + (f", cost_per_token={cost_map.get(name)}" if cost_map.get(name) is not None else "")
        for name, caps in candidates.items()
    )
    return DISPATCHER_PROMPT_TEMPLATE.format(
        quality_preference=quality_preference,
        candidates="\n".join(lines),
    )


class LLMRouter(CustomLogger):
    """LLM-dispatch model router registered via ``model: auto_router/llm_router``.

    ``model_to_capabilities`` and ``model_to_cost`` are derived from the OTHER
    deployments declared in ``available_models`` (mirrors the adaptive router's
    dependency-injected ``model_to_prefs`` / ``model_to_cost``)."""

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: Router,
        config: LLMRouterConfig,
        model_to_capabilities: Mapping[str, LLMRouterCapabilities],
        model_to_cost: Mapping[str, float],
        dispatcher: DispatcherClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance
        self.config = config
        self._candidates: Mapping[str, LLMRouterCapabilities] = dict(model_to_capabilities)
        self._cost_map: Mapping[str, float] = dict(model_to_cost)
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._context_template: str = _build_candidate_context(
            self._candidates, self._cost_map, config.quality_preference
        )
        self._dispatcher: DispatcherClient | None = dispatcher
        if self._dispatcher is None and config.dispatcher_model is not None:
            self._dispatcher = LiteLLMDispatcher(
                dispatcher_model=config.dispatcher_model,
                temperature=config.dispatcher_temperature,
                max_tokens=config.dispatcher_max_tokens,
            )

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, Any],
        messages: list[dict[str, Any]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse | None:
        from litellm.types.router import PreRoutingHookResponse

        if messages is None:
            return None

        user_text = get_last_user_message(cast(list[AllMessageValues], messages)) or ""
        if not user_text:
            return self._respond(self._heuristic_model(None), messages, request_kwargs, "no_user_message")

        analysis = self._analyze(user_text, messages, request_kwargs)
        chosen, via = await self._route(user_text, analysis)
        self._stash_decision(request_kwargs, chosen, via)
        return PreRoutingHookResponse(model=chosen, messages=messages)

    def _analyze(
        self,
        user_text: str,
        messages: object,
        request_kwargs: Mapping[str, object],
    ) -> PromptAnalysis | None:
        if not self.config.prompt_analysis_enabled:
            return None
        return analyze_prompt(
            user_text,
            has_images=messages_have_images(messages),
            has_tools=tools_requested(request_kwargs.get("tools")),
            has_json_mode=is_json_object_response_format(request_kwargs.get("response_format")),
        )

    async def _route(self, user_text: str, analysis: PromptAnalysis | None) -> tuple[str, str]:
        key = self._cache_key(user_text, analysis)
        cached = self._cache_lookup(key)
        if cached is not None:
            return cached, "cache"

        chosen, via = await self._dispatch(user_text)
        if chosen is None:
            chosen = self._heuristic_model(analysis)
            via = "heuristic"
        self._cache_store(key, chosen)
        return chosen, via

    async def _dispatch(self, user_text: str) -> tuple[str | None, str]:
        if self._dispatcher is None:
            return None, "heuristic"
        try:
            raw = await self._dispatcher.choose_model(self._context_template, user_text)
        except Exception as e:  # noqa: BLE001  # dispatcher is best-effort: any failure must fall back to the heuristic
            verbose_router_logger.warning("LLMRouter dispatcher call failed: %s", e)
            return None, "heuristic"

        chosen = _parse_dispatch_response(raw)
        if chosen is None or chosen not in self._candidates:
            verbose_router_logger.warning("LLMRouter dispatcher returned invalid choice: %s", raw)
            return None, "heuristic"
        return chosen, "dispatcher"

    def _heuristic_model(self, analysis: PromptAnalysis | None) -> str:
        if self._candidates:
            if analysis is not None:
                return pick_model_prompt_aware(
                    self._candidates, self._cost_map, self.config.quality_preference, analysis
                )
            return pick_model_heuristic(self._candidates, self._cost_map, self.config.quality_preference)
        default = self.config.default_model
        if default is not None:
            return default
        raise ValueError(f"LLMRouter[{self.model_name}] has no candidates and no default_model")

    def _respond(
        self,
        model: str,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
        via: str,
    ) -> PreRoutingHookResponse:
        from litellm.types.router import PreRoutingHookResponse

        self._stash_decision(request_kwargs, model, via)
        return PreRoutingHookResponse(model=model, messages=messages)

    def _stash_decision(
        self,
        request_kwargs: dict[str, Any] | None,
        chosen: str,
        via: str,
    ) -> None:
        if request_kwargs is None:
            return
        metadata = request_kwargs.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata[LLM_ROUTER_DECISION_KEY] = {
                "router_model_name": self.model_name,
                "routed_model": chosen,
                "routed_via": via,
            }

    def _cache_key(self, user_text: str, analysis: PromptAnalysis | None) -> str:
        candidate_names = tuple(sorted(self._candidates.keys()))
        capability_flags = (analysis.vision, analysis.tools, analysis.complexity) if analysis is not None else None
        payload = json.dumps(
            {
                "prompt": user_text,
                "candidates": candidate_names,
                "preference": self.config.quality_preference,
                "capabilities": capability_flags,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_lookup(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, model = entry
        if expires_at <= time.monotonic():
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return model

    def _cache_store(self, key: str, model: str) -> None:
        if self.config.cache_ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + self.config.cache_ttl_seconds
        self._cache[key] = (expires_at, model)
        self._cache.move_to_end(key)
        if len(self._cache) >= DECISION_CACHE_SWEEP_THRESHOLD:
            self._evict_expired_cache()

    def _evict_expired_cache(self) -> None:
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._cache.items() if exp <= now]
        for k in expired:
            self._cache.pop(k, None)
