"""Constants and the dispatcher prompt template for the LLM router strategy."""

DEFAULT_QUALITY_PREFERENCE: float = 0.5
DEFAULT_CACHE_TTL_SECONDS: int = 300
DEFAULT_DISPATCHER_TEMPERATURE: float = 0.0
DEFAULT_DISPATCHER_MAX_TOKENS: int = 200

DECISION_CACHE_SWEEP_THRESHOLD: int = 1024

LLM_ROUTER_DECISION_KEY: str = "llm_router_decision"

DISPATCHER_PROMPT_TEMPLATE: str = """\
You are a model router. Pick the single best model for the user's prompt from \
the candidate list below, given the caller's quality-vs-speed preference. The \
user's prompt arrives in the user-role message of this request, so do not echo \
it back; just choose the best candidate for it.

Quality preference: {quality_preference} (0.0 = prefer speed/low cost, \
1.0 = prefer highest quality).

Candidates:
{candidates}

Respond with ONLY a JSON object of the form:
{{"model": "<candidate name>", "reasoning": "<one short sentence>"}}
The "model" value MUST be one of the candidate names, exactly as written.\
"""
