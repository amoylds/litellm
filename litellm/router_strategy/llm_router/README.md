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
   response, chosen model not a candidate), fall back to a deterministic
   heuristic: pick by `quality_preference`-weighted score over normalized
   declared `quality_score` / `speed_score`, tie-break by cheapest cost then
   name.
5. Each routing decision is cached in-memory by a hash of (prompt, candidate
   set, preference) for `cache_ttl_seconds`, so repeated identical prompts skip
   the dispatcher call.

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
- `strengths` (list of free-text strings, surfaced to the dispatcher)

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

dispatcher LLM (primary) -> heuristic scorer -> `default_model` -> first
candidate.
