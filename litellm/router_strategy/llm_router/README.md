# LLM Router

A routing strategy where an LLM picks the best model for each prompt. For every
incoming request, a configurable "dispatcher" LLM is shown the candidate models
plus their declared capability/benchmark metadata and a quality-vs-speed
preference, and returns the single model it judges best for that prompt.

This mirrors the "AI automatically picks the right model for each prompt" idea
while slotting into litellm's existing `auto_router/<name>` registration path,
so it needs no changes to the routing plumbing.

## How it works

For each request:

1. Extract the last user message.
2. If a `dispatcher_model` is configured, build a compact context string of the
   candidates (name, declared capability scores, context window, strengths,
   cost) and send a structured prompt to the dispatcher LLM asking it to return
   JSON `{"model": <name>, "reasoning": <text>}`. The chosen model is validated
   against the candidate set.
3. The `quality_preference` knob (0.0 = prefer speed/low cost, 1.0 = prefer
   highest quality) is included in the dispatcher prompt context so the
   decision and the bias live in one place.
4. On any failure (no dispatcher configured, LLM call error, unparseable
   response, chosen model not a candidate), fall back to a deterministic,
   content-aware heuristic (see below). No LLM call, no cost.
5. Each routing decision is cached in-memory by a hash of (prompt, candidate
   set, preference, capability requirements) for `cache_ttl_seconds`, so
   repeated identical prompts skip the dispatcher call.

## Content-aware heuristic

Ported from SmarterRouter, the heuristic analyzes the prompt itself so the
router can pick the right model with zero extra LLM calls; setting
`dispatcher_model` to null makes this the primary, fully local and free path.

Each prompt is scored across task categories (reasoning, coding, creativity,
factual) from keyword and structure signals, plus a meta `complexity` score and
`vision` / `tools` capability requirements read from the request envelope
(image content parts, `tools`, and a `{"type": "json_object"}` response format).
Each candidate's declared `strengths` are matched to those same categories, and
the model whose strengths best fit what the prompt needs wins, blended with the
`quality_preference`-weighted `quality_score` / `speed_score` and a
complexity-driven nudge toward higher-quality models. When a prompt needs vision
or tools, only candidates that declare that capability are considered (unless
none qualify). Ties break by cheapest cost then name. So a coding prompt goes to
the model that declares a coding strength, a factual lookup goes to the
factual/general model, and an image prompt goes to a vision model even if a
text-only model scores higher on quality. Set `prompt_analysis_enabled: false`
to revert to the prompt-blind `quality_score` / `speed_score` scorer.

The decision is stashed on `request_kwargs["metadata"]["llm_router_decision"]`
as `{"router_model_name", "routed_model", "routed_via"}` where `routed_via` is
`"dispatcher"`, `"heuristic"`, or `"cache"`.

## Capability data

Capability/benchmark data is declared statically per model in `model_info` under
`llm_router_capabilities`, validated by the `LLMRouterCapabilities` Pydantic
model. There is no external API fetch; everything is admin-controlled. All
fields are optional:

- `quality_score` (0.0-1.0)
- `speed_score` (0.0-1.0)
- `context_window` (tokens)
- `strengths` (list of free-text strings, surfaced to the dispatcher and matched
  to prompt categories by the content-aware heuristic; keywords like `code`,
  `reason`/`math`, `creative`/`writing`, `fact`/`qa`, `vision`/`image`, and
  `tool`/`function`/`agent` map to the coding, reasoning, creativity, factual,
  vision, and tools categories)

Per-model cost is read from the deployment's `input_cost_per_token`.

## Config example

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      input_cost_per_token: 0.00000015
    model_info:
      llm_router_capabilities:
        quality_score: 0.4
        speed_score: 0.95
        context_window: 128000
        strengths: ["factual_lookup", "simple_qa"]

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      input_cost_per_token: 0.0000025
    model_info:
      llm_router_capabilities:
        quality_score: 0.95
        speed_score: 0.3
        context_window: 128000
        strengths: ["reasoning", "code_generation", "analysis"]

  - model_name: smart-router
    litellm_params:
      model: auto_router/llm_router
      llm_router_default_model: gpt-4o-mini
      llm_router_config:
        available_models: ["gpt-4o-mini", "gpt-4o"]
        dispatcher_model: gpt-4o-mini
        quality_preference: 0.6
        cache_ttl_seconds: 300
        dispatcher_temperature: 0.0
        dispatcher_max_tokens: 200
        prompt_analysis_enabled: true
```

## Comparison with the other routers

- `complexity_router`: rule-based complexity scoring into tiers, zero API
  calls. `llm_router` instead asks an LLM to make the per-prompt choice.
- `adaptive_router`: rule-based request-type classification plus a bandit that
  learns from post-call signals. `llm_router` uses upfront capability metadata
  and an LLM judgment rather than learned feedback.
- `quality_router`: maps a complexity tier to a manually-declared quality tier
  per model. `llm_router` uses richer per-model capability scores and an LLM
  dispatcher as the primary path.

## Fallback chain

dispatcher LLM (primary) -> content-aware heuristic scorer -> `default_model` ->
first candidate.
