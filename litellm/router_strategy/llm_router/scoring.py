"""Prompt-aware deterministic scorer for the LLM router.

Given a :class:`PromptAnalysis` plus each candidate's declared
``llm_router_capabilities``, pick the model whose declared strengths best match
what the prompt needs, biased by the quality-vs-speed preference. This is
SmarterRouter's "automatically pick the right model for each prompt" idea done
locally with no LLM call: a coding prompt goes to the model that declares a
coding strength, a hard prompt is nudged toward higher-quality models, and image
or tool prompts are restricted to models that declare those capabilities.

Pure and fully typed; all state is built with comprehensions wrapped in
``tuple()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from litellm.router_strategy.llm_router.prompt_analysis import PromptAnalysis
from litellm.types.router import LLMRouterCapabilities

_GENERALIST_FLOOR: float = 0.1
_SPECIALIST_SCORE: float = 1.0
_CAPABILITY_THRESHOLD: float = 0.5

_CATEGORY_WEIGHT: float = 0.5
_QUALITY_SPEED_WEIGHT: float = 0.35
_COMPLEXITY_WEIGHT: float = 0.15

_CATEGORY_STRENGTH_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "coding": ("cod", "program", "algorithm", "debug", "sql", "script", "engineer"),
    "reasoning": ("reason", "logic", "math", "analy", "problem", "solv", "think"),
    "creativity": ("creativ", "writ", "story", "poem", "brainstorm", "narrat", "roleplay"),
    "factual": ("fact", "qa", "lookup", "knowledge", "summar", "general", "chat", "retriev"),
    "vision": ("vision", "image", "multimodal", "ocr", "visual"),
    "tools": ("tool", "function", "agent", "structured"),
}


@dataclass(frozen=True, slots=True)
class CategoryAffinity:
    """How strongly a model's declared strengths lean into each category. A
    category with no matching strength keeps the generalist floor so every model
    stays a viable generalist."""

    coding: float
    reasoning: float
    creativity: float
    factual: float
    vision: float
    tools: float


def _affinity_for(category: str, strengths: tuple[str, ...]) -> float:
    keywords = _CATEGORY_STRENGTH_KEYWORDS[category]
    matched = any(kw in strength for strength in strengths for kw in keywords)
    return _SPECIALIST_SCORE if matched else _GENERALIST_FLOOR


def affinity_from_strengths(strengths: tuple[str, ...]) -> CategoryAffinity:
    lowered = tuple(s.lower() for s in strengths)
    return CategoryAffinity(
        coding=_affinity_for("coding", lowered),
        reasoning=_affinity_for("reasoning", lowered),
        creativity=_affinity_for("creativity", lowered),
        factual=_affinity_for("factual", lowered),
        vision=_affinity_for("vision", lowered),
        tools=_affinity_for("tools", lowered),
    )


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        return tuple(1.0 for _ in values)
    return tuple((v - lo) / span for v in values)


def _midpoint(v: float | None) -> float:
    return v if v is not None else 0.5


def _category_match(analysis: PromptAnalysis, affinity: CategoryAffinity) -> float:
    return (
        analysis.reasoning * affinity.reasoning
        + analysis.coding * affinity.coding
        + analysis.creativity * affinity.creativity
        + analysis.factual * affinity.factual
    )


def _meets_requirements(analysis: PromptAnalysis, affinity: CategoryAffinity) -> bool:
    if analysis.vision >= 1.0 and affinity.vision < _CAPABILITY_THRESHOLD:
        return False
    if analysis.tools >= 1.0 and affinity.tools < _CAPABILITY_THRESHOLD:
        return False
    return True


def candidate_meets_requirements(analysis: PromptAnalysis, capabilities: LLMRouterCapabilities) -> bool:
    """Whether ``capabilities`` satisfies the vision/tools requirements ``analysis``
    demands. Used to reject a dispatcher pick that would violate a capability the
    prompt needs, so the LLM path honors the same guarantees as the local scorer."""
    return _meets_requirements(analysis, affinity_from_strengths(tuple(capabilities.strengths)))


def pick_model_prompt_aware(
    candidates: Mapping[str, LLMRouterCapabilities],
    cost_map: Mapping[str, float],
    quality_preference: float,
    analysis: PromptAnalysis,
) -> str:
    """Pick the candidate that best fits ``analysis``. Models that declare a
    required capability (vision/tools) are preferred exclusively when the prompt
    needs it, unless none qualify. Ties break by cheapest cost then name. Raises
    ``ValueError`` when ``candidates`` is empty."""
    if not candidates:
        raise ValueError("pick_model_prompt_aware requires at least one candidate")

    affinities = {name: affinity_from_strengths(tuple(caps.strengths)) for name, caps in candidates.items()}

    eligible = tuple(n for n in candidates if _meets_requirements(analysis, affinities[n]))
    names = eligible or tuple(candidates.keys())

    norm_category = _normalize(tuple(_category_match(analysis, affinities[n]) for n in names))
    norm_quality = _normalize(tuple(_midpoint(candidates[n].quality_score) for n in names))
    norm_speed = _normalize(tuple(_midpoint(candidates[n].speed_score) for n in names))

    quality_speed = tuple(
        quality_preference * norm_quality[i] + (1.0 - quality_preference) * norm_speed[i] for i in range(len(names))
    )

    ranked = sorted(
        (
            (
                names[i],
                _CATEGORY_WEIGHT * norm_category[i]
                + _QUALITY_SPEED_WEIGHT * quality_speed[i]
                + _COMPLEXITY_WEIGHT * analysis.complexity * norm_quality[i],
                cost_map.get(names[i], 0.0),
                names[i],
            )
            for i in range(len(names))
        ),
        key=lambda t: (-t[1], t[2], t[3]),
    )
    return ranked[0][0]
