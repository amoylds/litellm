"""Content-aware prompt analysis for the LLM router.

Ports SmarterRouter's deterministic prompt categorization so the router can pick
a model from the prompt's content alone, with zero extra LLM calls. Each prompt
is scored across task categories (reasoning, coding, creativity, factual) plus a
meta ``complexity`` signal and ``vision`` / ``tools`` capability requirements.
The scores feed the prompt-aware heuristic in ``scoring.py``.

Pure and fully typed: every score is computed as an expression and the frozen
result is built in one shot, so there is no seed-then-mutate state.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter, ValidationError

TASK_CATEGORIES: tuple[str, ...] = ("reasoning", "coding", "creativity", "factual")

_REASONING_KEYWORDS: tuple[str, ...] = (
    "calculate",
    "logic",
    "solve",
    "reason",
    "prove",
    "math",
    "sequence",
    "pattern",
    "if then",
    "therefore",
    "because",
    "derive",
    "speed",
    "velocity",
    "distance",
    "how much",
    "how many",
    "result",
)
_CODING_KEYWORDS: tuple[str, ...] = (
    "code",
    "function",
    "implement",
    "algorithm",
    "program",
    "python",
    "javascript",
    "java",
    "sql",
    "debug",
    "api",
    "class",
    "def ",
    "return",
    "import",
    "write code",
    "bug",
    "fix",
    "script",
    "json",
    "xml",
    "yaml",
    "parse",
    "schema",
)
_CREATIVE_KEYWORDS: tuple[str, ...] = (
    "story",
    "write",
    "poem",
    "creative",
    "imagine",
    "describe",
    "invent",
    "fantasy",
    "narrative",
    "character",
    "scene",
    "song",
    "haiku",
    "lyrics",
    "joke",
    "humor",
)
_FACTUAL_KEYWORDS: tuple[str, ...] = (
    "what is",
    "who is",
    "when did",
    "where is",
    "define",
    "explain",
    "fact",
    "history",
    "capital",
    "year",
    "date",
    "list",
    "tell me about",
    "summary",
    "summarize",
)
_COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "complex",
    "expert",
    "detailed",
    "comprehensive",
    "optimized",
    "architecture",
    "distributed",
    "performance",
    "scalable",
    "deep dive",
    "advanced",
    "professional",
    "senior",
    "production-ready",
    "implement",
    "algorithm",
    "data structure",
    "tree",
    "graph",
    "recursive",
    "unit test",
    "type hint",
    "generics",
    "async",
    "concurrent",
    "nuance",
    "subtle",
    "imply",
    "hidden meaning",
    "step-by-step",
    "reasoning chain",
)
_CODING_COMPLEXITY_INDICATORS: tuple[str, ...] = (
    "with",
    "include",
    "and",
    "also",
    "plus",
    "additionally",
    "operations",
    "methods",
    "functions",
    "classes",
    "interface",
    "inheritance",
    "generic",
    "template",
    "exception",
    "handle",
    "error",
    "security",
    "thread",
)
_CODE_INDICATORS: tuple[str, ...] = (
    "```",
    "def ",
    "function ",
    "const ",
    "let ",
    "var ",
    "class ",
)


@dataclass(frozen=True, slots=True)
class PromptAnalysis:
    """Per-prompt category weights. ``vision`` and ``tools`` are 0.0/1.0
    capability requirements; the rest are unbounded-above affinity scores."""

    reasoning: float
    coding: float
    creativity: float
    factual: float
    complexity: float
    vision: float
    tools: float


def _keyword_score(text: str, keywords: tuple[str, ...], weight: float) -> float:
    return weight * sum(1 for kw in keywords if kw in text)


def _length_complexity(prompt: str) -> float:
    length = len(prompt)
    return (0.2 if length > 500 else 0.0) + (0.3 if length > 1500 else 0.0)


def _structure_complexity(prompt: str) -> float:
    return (0.1 if prompt.count("?") > 2 else 0.0) + (0.1 if prompt.count("\n") > 5 else 0.0)


def _coding_complexity(prompt_lower: str, coding: float) -> float:
    if coding <= 0.5:
        return 0.0
    hits = sum(1 for ind in _CODING_COMPLEXITY_INDICATORS if ind in prompt_lower)
    if hits >= 3:
        return 0.3
    if hits >= 2:
        return 0.15
    return 0.0


def analyze_prompt(
    prompt: str,
    *,
    has_images: bool = False,
    has_tools: bool = False,
    has_json_mode: bool = False,
) -> PromptAnalysis:
    """Score ``prompt`` across task categories plus complexity, mirroring
    SmarterRouter's ``_analyze_prompt``. Capability flags come from the request
    envelope (image parts, ``tools``, JSON response format) rather than the text.
    """
    lower = prompt.lower()

    reasoning = _keyword_score(lower, _REASONING_KEYWORDS, 0.3)
    creativity = _keyword_score(lower, _CREATIVE_KEYWORDS, 0.35)
    factual = _keyword_score(lower, _FACTUAL_KEYWORDS, 0.3)

    coding_from_caps = (0.2 if has_tools else 0.0) + (0.3 if has_json_mode else 0.0)
    coding_text = _keyword_score(lower, _CODING_KEYWORDS, 0.4) + coding_from_caps
    has_code_block = any(ind in prompt for ind in _CODE_INDICATORS)
    coding = 1.0 if has_code_block else coding_text

    complexity = min(
        _length_complexity(prompt)
        + _structure_complexity(prompt)
        + _keyword_score(lower, _COMPLEXITY_KEYWORDS, 0.15)
        + _coding_complexity(lower, coding)
        + (0.3 if has_tools else 0.0)
        + (0.1 if has_json_mode else 0.0),
        1.0,
    )

    vision = 1.0 if has_images else 0.0
    tools = 1.0 if has_tools else 0.0

    non_meta = (reasoning, coding, creativity, factual, vision, tools)
    factual_final = 0.5 if max(*non_meta, complexity) == 0.0 else factual

    return PromptAnalysis(
        reasoning=reasoning,
        coding=coding,
        creativity=creativity,
        factual=factual_final,
        complexity=complexity,
        vision=vision,
        tools=tools,
    )


class _TypedPart(BaseModel):
    type: str = ""


class _TypedMessage(BaseModel):
    content: object = None


_MESSAGE_LIST = TypeAdapter(list[_TypedMessage])
_PART_LIST = TypeAdapter(list[_TypedPart])
_RESPONSE_FORMAT = TypeAdapter(_TypedPart)
_TOOL_LIST = TypeAdapter(list[object])


def _content_has_image(content: object) -> bool:
    if not isinstance(content, list):
        return False
    try:
        parts = _PART_LIST.validate_python(content)
    except ValidationError:
        return False
    return any(part.type == "image_url" for part in parts)


def messages_have_images(messages: object) -> bool:
    """True when any message carries an ``image_url`` content part. Accepts the
    raw request messages and validates them into a typed shape before inspection.
    """
    try:
        typed = _MESSAGE_LIST.validate_python(messages)
    except ValidationError:
        return False
    return any(_content_has_image(msg.content) for msg in typed)


def is_json_object_response_format(response_format: object) -> bool:
    """True when ``response_format`` requests ``{"type": "json_object"}``."""
    try:
        parsed = _RESPONSE_FORMAT.validate_python(response_format)
    except ValidationError:
        return False
    return parsed.type == "json_object"


def tools_requested(tools: object) -> bool:
    """True when the request declares a non-empty ``tools`` list."""
    try:
        parsed = _TOOL_LIST.validate_python(tools)
    except ValidationError:
        return False
    return len(parsed) > 0
