"""Regression tests for the LLMRouter strategy."""

from typing import List
from unittest.mock import MagicMock

import pytest

from litellm.router_strategy.llm_router.dispatcher import DispatcherClient
from litellm.router_strategy.llm_router.llm_router import LLMRouter
from litellm.types.router import LLMRouterCapabilities, LLMRouterConfig


def _caps(
    quality_score: float | None = None,
    speed_score: float | None = None,
    context_window: int | None = None,
    strengths: List[str] | None = None,
) -> LLMRouterCapabilities:
    return LLMRouterCapabilities(
        quality_score=quality_score,
        speed_score=speed_score,
        context_window=context_window,
        strengths=strengths or [],
    )


def _config(**overrides) -> LLMRouterConfig:
    base = {
        "available_models": ["fast", "smart"],
        "dispatcher_model": "gpt-4o-mini",
        "quality_preference": 0.5,
        "cache_ttl_seconds": 300,
    }
    base.update(overrides)
    return LLMRouterConfig(**base)


def _candidates() -> dict:
    return {
        "fast": _caps(quality_score=0.3, speed_score=0.9),
        "smart": _caps(quality_score=0.95, speed_score=0.2),
    }


def _costs() -> dict:
    return {"fast": 0.00000015, "smart": 0.0000025}


class StubDispatcher:
    """Records calls and returns a canned raw response."""

    def __init__(self, response: str = "", raises: BaseException | None = None) -> None:
        self._response = response
        self._raises = raises
        self.call_count = 0
        self.prompts: list[str] = []
        self.contexts: list[str] = []

    async def choose_model(self, context: str, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        self.contexts.append(context)
        if self._raises is not None:
            raise self._raises
        return self._response


def _make_router(
    dispatcher: DispatcherClient | None = None,
    config: LLMRouterConfig | None = None,
    candidates: dict | None = None,
    costs: dict | None = None,
) -> LLMRouter:
    return LLMRouter(
        model_name="smart-router",
        litellm_router_instance=MagicMock(),
        config=config or _config(),
        model_to_capabilities=candidates or _candidates(),
        model_to_cost=costs or _costs(),
        dispatcher=dispatcher,
    )


class TestDispatcherPath:
    @pytest.mark.asyncio
    async def test_dispatcher_returns_valid_candidate(self):
        stub = StubDispatcher(response='{"model": "smart", "reasoning": "needs reasoning"}')
        router = _make_router(dispatcher=stub)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "Explain quantum tunneling"}]
        )
        assert result is not None
        assert result.model == "smart"
        assert stub.call_count == 1

    @pytest.mark.asyncio
    async def test_dispatcher_returns_non_candidate_falls_back(self):
        stub = StubDispatcher(response='{"model": "unknown-model", "reasoning": "x"}')
        router = _make_router(dispatcher=stub)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )
        assert result is not None
        assert result.model in {"fast", "smart"}
        assert result.model != "unknown-model"

    @pytest.mark.asyncio
    async def test_dispatcher_raises_falls_back(self):
        stub = StubDispatcher(raises=RuntimeError("dispatcher down"))
        router = _make_router(dispatcher=stub)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )
        assert result is not None
        assert result.model in {"fast", "smart"}

    @pytest.mark.asyncio
    async def test_dispatcher_unparseable_response_falls_back(self):
        stub = StubDispatcher(response="the best model is smart obviously")
        router = _make_router(dispatcher=stub)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )
        assert result is not None
        assert result.model in {"fast", "smart"}


class TestDispatcherPrompt:
    @pytest.mark.asyncio
    async def test_context_has_no_literal_prompt_placeholder(self):
        # Regression: the dispatcher system message must not contain the literal
        # "{prompt}" placeholder. The user text arrives in the user-role message.
        stub = StubDispatcher(response='{"model": "smart"}')
        router = _make_router(dispatcher=stub)
        await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
        )
        context = stub.contexts[0]
        assert "{prompt}" not in context
        assert "Candidates:" in context
        assert "fast" in context and "smart" in context

    @pytest.mark.asyncio
    async def test_context_is_precomputed_once(self):
        # Regression: the candidate context is fixed at construction and should be
        # reused across dispatch calls rather than rebuilt each time.
        stub = StubDispatcher(response='{"model": "smart"}')
        router = _make_router(dispatcher=stub)
        await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "first"}]
        )
        await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "second"}]
        )
        assert stub.contexts[0] == stub.contexts[1]


class TestFenceStripping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"model": "smart"}', "smart"),
            ('```json\n{"model": "smart"}```', "smart"),
            ('```\n{"model": "smart"}\n```', "smart"),
            # Regression: a model name containing a backtick must survive fence
            # stripping (strip("`") would corrupt the JSON content).
            ('{"model": "fast`tick"}', "fast`tick"),
        ],
    )
    def test_parse_dispatch_response_handles_fences(self, raw: str, expected: str):
        from litellm.router_strategy.llm_router.llm_router import _parse_dispatch_response

        assert _parse_dispatch_response(raw) == expected


class TestConfigValidation:
    def test_available_models_rejects_empty_list(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMRouterConfig(available_models=[])


class TestHeuristicFallback:
    @pytest.mark.asyncio
    async def test_no_dispatcher_uses_heuristic_no_llm_call(self):
        config = _config(dispatcher_model=None)
        router = _make_router(dispatcher=None, config=config)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )
        assert result is not None
        assert result.model in {"fast", "smart"}

    @pytest.mark.asyncio
    async def test_quality_preference_one_picks_highest_quality(self):
        config = _config(dispatcher_model=None, quality_preference=1.0)
        router = _make_router(dispatcher=None, config=config)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )
        assert result is not None
        assert result.model == "smart"

    @pytest.mark.asyncio
    async def test_quality_preference_zero_picks_fastest(self):
        config = _config(dispatcher_model=None, quality_preference=0.0)
        router = _make_router(dispatcher=None, config=config)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )
        assert result is not None
        assert result.model == "fast"


class TestDecisionCache:
    @pytest.mark.asyncio
    async def test_identical_prompt_hits_cache_dispatcher_called_once(self):
        stub = StubDispatcher(response='{"model": "smart"}')
        router = _make_router(dispatcher=stub)
        messages = [{"role": "user", "content": "same prompt"}]
        first = await router.async_pre_routing_hook(model="smart-router", request_kwargs={}, messages=messages)
        second = await router.async_pre_routing_hook(model="smart-router", request_kwargs={}, messages=messages)
        assert first is not None and second is not None
        assert first.model == second.model == "smart"
        assert stub.call_count == 1

    @pytest.mark.asyncio
    async def test_different_prompt_does_not_hit_cache(self):
        stub = StubDispatcher(response='{"model": "smart"}')
        router = _make_router(dispatcher=stub)
        await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "prompt one"}]
        )
        await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "user", "content": "prompt two"}]
        )
        assert stub.call_count == 2


class TestPreRoutingResponse:
    @pytest.mark.asyncio
    async def test_returns_pre_routing_hook_response_with_chosen_model(self):
        stub = StubDispatcher(response='{"model": "fast"}')
        router = _make_router(dispatcher=stub)
        messages = [{"role": "user", "content": "hello"}]
        result = await router.async_pre_routing_hook(model="smart-router", request_kwargs={}, messages=messages)
        assert result is not None
        assert result.model == "fast"
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_decision_stashed_on_metadata(self):
        stub = StubDispatcher(response='{"model": "smart"}')
        router = _make_router(dispatcher=stub)
        request_kwargs: dict = {}
        await router.async_pre_routing_hook(
            model="smart-router", request_kwargs=request_kwargs, messages=[{"role": "user", "content": "hi"}]
        )
        decision = request_kwargs["metadata"]["llm_router_decision"]
        assert decision["router_model_name"] == "smart-router"
        assert decision["routed_model"] == "smart"
        assert decision["routed_via"] == "dispatcher"

    @pytest.mark.asyncio
    async def test_no_messages_returns_none(self):
        router = _make_router(dispatcher=StubDispatcher())
        result = await router.async_pre_routing_hook(model="smart-router", request_kwargs={}, messages=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_user_message_falls_back(self):
        stub = StubDispatcher()
        router = _make_router(dispatcher=stub)
        result = await router.async_pre_routing_hook(
            model="smart-router", request_kwargs={}, messages=[{"role": "system", "content": "be helpful"}]
        )
        assert result is not None
        assert result.model in {"fast", "smart"}
        assert stub.call_count == 0


class TestRouterWiring:
    def test_is_llm_router_deployment_true(self):
        from litellm import Router
        from litellm.types.router import LiteLLM_Params

        router = Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
        params = LiteLLM_Params(model="auto_router/llm_router/my-router")
        assert router._is_llm_router_deployment(params) is True
        assert router._is_auto_router_deployment(params) is False

    def test_is_llm_router_deployment_false_for_other_prefix(self):
        from litellm import Router
        from litellm.types.router import LiteLLM_Params

        router = Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
        assert router._is_llm_router_deployment(LiteLLM_Params(model="auto_router/complexity_router")) is False
        assert router._is_llm_router_deployment(LiteLLM_Params(model="openai/gpt-4o")) is False

    def test_init_llm_router_deployment(self):
        from litellm import Router
        from litellm.types.router import Deployment, LiteLLM_Params

        router = Router(
            model_list=[
                {
                    "model_name": "fast",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "input_cost_per_token": 0.00000015},
                    "model_info": {"llm_router_capabilities": {"quality_score": 0.3, "speed_score": 0.9}},
                },
                {
                    "model_name": "smart",
                    "litellm_params": {"model": "openai/gpt-4o", "input_cost_per_token": 0.0000025},
                    "model_info": {"llm_router_capabilities": {"quality_score": 0.95, "speed_score": 0.2}},
                },
            ]
        )
        deployment = Deployment(
            model_name="auto_router/llm_router/test-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/llm_router/test-router",
                llm_router_config={
                    "available_models": ["fast", "smart"],
                    "dispatcher_model": "gpt-4o-mini",
                },
            ),
            model_info={"id": "test-id"},
        )
        router.init_llm_router_deployment(deployment)
        assert "auto_router/llm_router/test-router" in router.llm_routers
        llm_router = router.llm_routers["auto_router/llm_router/test-router"]
        assert set(llm_router._candidates.keys()) == {"fast", "smart"}
        assert llm_router._candidates["smart"].quality_score == 0.95
        assert llm_router._cost_map["fast"] == 0.00000015

    def test_set_model_list_finalizes_llm_router(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "fast",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                    "model_info": {"llm_router_capabilities": {"quality_score": 0.3, "speed_score": 0.9}},
                },
                {
                    "model_name": "smart",
                    "litellm_params": {"model": "openai/gpt-4o"},
                    "model_info": {"llm_router_capabilities": {"quality_score": 0.95, "speed_score": 0.2}},
                },
                {
                    "model_name": "smart-router",
                    "litellm_params": {
                        "model": "auto_router/llm_router",
                        "llm_router_config": {
                            "available_models": ["fast", "smart"],
                            "dispatcher_model": "gpt-4o-mini",
                        },
                    },
                },
            ]
        )
        assert "smart-router" in router.llm_routers
