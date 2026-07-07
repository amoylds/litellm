"""Deterministic heuristic fallback for the LLM router.

Used when no dispatcher is configured, the dispatcher call fails, its response
is unparseable, or the chosen model is not a candidate. Pure, fully typed, no
mutation.
"""

from __future__ import annotations

from collections.abc import Mapping

from litellm.types.router import LLMRouterCapabilities


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


def pick_model_heuristic(
    candidates: Mapping[str, LLMRouterCapabilities],
    cost_map: Mapping[str, float],
    quality_preference: float,
) -> str:
    """Pick the best candidate by a quality_preference-weighted score over
    normalized declared capability. Ties break by cheapest cost then by name.
    Raises ``ValueError`` when ``candidates`` is empty."""
    if not candidates:
        raise ValueError("pick_model_heuristic requires at least one candidate")

    names = tuple(candidates.keys())
    norm_q = _normalize(tuple(_midpoint(candidates[n].quality_score) for n in names))
    norm_s = _normalize(tuple(_midpoint(candidates[n].speed_score) for n in names))

    ranked = sorted(
        (
            (
                names[i],
                quality_preference * norm_q[i] + (1.0 - quality_preference) * norm_s[i],
                cost_map.get(names[i], 0.0),
                names[i],
            )
            for i in range(len(names))
        ),
        key=lambda t: (-t[1], t[2], t[3]),
    )
    return ranked[0][0]
