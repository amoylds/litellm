"""Tests for the prompt-aware deterministic model scorer."""

from typing import List, Optional

import pytest

from litellm.router_strategy.llm_router.prompt_analysis import analyze_prompt
from litellm.router_strategy.llm_router.scoring import (
    affinity_from_strengths,
    pick_model_prompt_aware,
)
from litellm.types.router import LLMRouterCapabilities


def _caps(
    quality_score: Optional[float] = None,
    speed_score: Optional[float] = None,
    strengths: Optional[List[str]] = None,
) -> LLMRouterCapabilities:
    return LLMRouterCapabilities(
        quality_score=quality_score,
        speed_score=speed_score,
        strengths=strengths or [],
    )


class TestAffinity:
    def test_coding_strength_lights_up_only_coding(self):
        aff = affinity_from_strengths(("code_generation",))
        assert aff.coding == 1.0
        assert aff.reasoning == 0.1
        assert aff.creativity == 0.1
        assert aff.factual == 0.1

    def test_vision_strength_lights_up_vision(self):
        aff = affinity_from_strengths(("vision", "ocr"))
        assert aff.vision == 1.0
        assert aff.coding == 0.1

    def test_no_strengths_all_floor(self):
        aff = affinity_from_strengths(())
        assert (aff.coding, aff.reasoning, aff.creativity, aff.factual) == (0.1, 0.1, 0.1, 0.1)


class TestPromptAwareSelection:
    def test_coding_prompt_prefers_coder_over_stronger_generalist(self):
        candidates = {
            "coder": _caps(quality_score=0.3, speed_score=0.5, strengths=["code_generation"]),
            "smart": _caps(quality_score=0.9, speed_score=0.5, strengths=["general_knowledge"]),
        }
        costs = {"coder": 0.0, "smart": 0.0}
        analysis = analyze_prompt("debug this python function and fix the algorithm bug in my api code")
        assert pick_model_prompt_aware(candidates, costs, 0.5, analysis) == "coder"

    def test_factual_prompt_prefers_factual_model(self):
        candidates = {
            "coder": _caps(quality_score=0.9, speed_score=0.5, strengths=["code_generation"]),
            "smart": _caps(quality_score=0.3, speed_score=0.5, strengths=["general_knowledge"]),
        }
        costs = {"coder": 0.0, "smart": 0.0}
        analysis = analyze_prompt("what is the capital of france and explain its history")
        assert pick_model_prompt_aware(candidates, costs, 0.5, analysis) == "smart"

    def test_vision_prompt_requires_vision_model_even_if_lower_quality(self):
        candidates = {
            "eyes": _caps(quality_score=0.4, speed_score=0.5, strengths=["vision"]),
            "brain": _caps(quality_score=0.95, speed_score=0.5, strengths=["reasoning"]),
        }
        costs = {"eyes": 0.0, "brain": 0.0}
        analysis = analyze_prompt("what is in this picture", has_images=True)
        assert pick_model_prompt_aware(candidates, costs, 1.0, analysis) == "eyes"

    def test_tools_prompt_requires_tool_model(self):
        candidates = {
            "agentic": _caps(quality_score=0.4, speed_score=0.5, strengths=["tool_use"]),
            "chatty": _caps(quality_score=0.95, speed_score=0.5, strengths=["general_knowledge"]),
        }
        costs = {"agentic": 0.0, "chatty": 0.0}
        analysis = analyze_prompt("book me a flight", has_tools=True)
        assert pick_model_prompt_aware(candidates, costs, 1.0, analysis) == "agentic"

    def test_unsatisfiable_vision_requirement_still_returns_a_model(self):
        candidates = {
            "a": _caps(quality_score=0.9, speed_score=0.5, strengths=["reasoning"]),
            "b": _caps(quality_score=0.3, speed_score=0.5, strengths=["general_knowledge"]),
        }
        costs = {"a": 0.0, "b": 0.0}
        analysis = analyze_prompt("what is in this picture", has_images=True)
        assert pick_model_prompt_aware(candidates, costs, 1.0, analysis) in {"a", "b"}

    def test_ties_break_by_cost_then_name(self):
        candidates = {
            "zeta": _caps(quality_score=0.5, speed_score=0.5, strengths=["general_knowledge"]),
            "alpha": _caps(quality_score=0.5, speed_score=0.5, strengths=["general_knowledge"]),
        }
        analysis = analyze_prompt("what is the capital of france")
        cheaper_alpha = pick_model_prompt_aware(
            candidates, {"zeta": 0.001, "alpha": 0.0001}, 0.5, analysis
        )
        assert cheaper_alpha == "alpha"
        equal_cost = pick_model_prompt_aware(candidates, {"zeta": 0.0, "alpha": 0.0}, 0.5, analysis)
        assert equal_cost == "alpha"

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            pick_model_prompt_aware({}, {}, 0.5, analyze_prompt("hi"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
