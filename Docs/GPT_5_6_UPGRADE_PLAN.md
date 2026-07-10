# GPT-5.6 Upgrade Plan (Handoff)

> **STATUS: IMPLEMENTED 2026-07-09** (same day, later session). All decisions below
> were executed as stated; the live verification checklist at the bottom was run
> against the real API and its results are recorded in
> "Verification results (2026-07-09)". Additions beyond this plan: a one-time DB
> migration swapping every user's default to gpt-5.6-sol + medium reasoning
> (`database.py::_migrate_gpt56`, sentinel `gpt56_migrated`), and an every-startup
> normalizer clamping stored models/efforts to the supported set.

Research handoff from a separate session (2026-07-09, GPT-5.6 GA launch day). This doc
contains everything needed to implement the upgrade: verified API facts, the decided
model mapping, and the concrete per-file change list. All API details below were
verified against OpenAI's official model pages and the `openai-python` v2.45.0 SDK
source (released 2026-07-09, the GPT-5.6 support release).

## Decisions (already made — implement as stated)

| Role | Current | New |
|---|---|---|
| Primary/default model (`GPT_MODEL` in `.env.example`) | `gpt-5.5` | `gpt-5.6-sol` |
| Settings-modal selectable models | `gpt-5.5` only | `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5` |
| Utility model (`UTILITY_MODEL`) | `gpt-5-mini` | `gpt-5.6-luna` |

Keep `gpt-5.5` fully working — it stays as a user-selectable option.
`gpt-5-mini` support can be dropped once utilities are on Luna (no user-facing surface).

## GPT-5.6 family — verified facts

Three tiers, released GA 2026-07-09. All share: **1,050,000-token context window,
128,000 max output tokens, knowledge cutoff February 16, 2026**, text+image input /
text output, Responses API + Chat Completions, streaming, function calling,
structured outputs.

| Model ID | Tier | Input /1M | Cached /1M | Output /1M |
|---|---|---|---|---|
| `gpt-5.6-sol` | Flagship | $5.00 | $0.50 | $30.00 |
| `gpt-5.6-terra` | Balanced (~GPT-5.5 quality) | $2.50 | $0.25 | $15.00 |
| `gpt-5.6-luna` | Fast/cheap | $1.00 | $0.10 | $6.00 |

- Bare `gpt-5.6` is an alias routing to Sol. Use the explicit `-sol` slug in config.
- Sol is priced identically to gpt-5.5 (same $5/$30) — the default swap is cost-neutral.
- Long-context surcharge unchanged from 5.5: prompts **> 272K input tokens bill at
  2× input / 1.5× output**. Applies to all three tiers.
- Prompt-cache economics: reads keep the 90% discount; **cache writes now bill at
  1.25× the input rate**; minimum cache life 30 minutes.
- Cost note for utilities: Luna is ~3–4× gpt-5-mini per token ($1/$6 vs $0.25/$2),
  but absolute utility spend is small; the volume-sensitive path is the wake
  classifier (runs per channel message). Quality is a full generation above mini —
  parity is not a concern.

## Reasoning effort — the critical changes

GPT-5.6 effort ladder: **`none`, `low`, `medium`, `high`, `xhigh`, `max`**. Default `medium`.

1. **`minimal` is NOT supported on any 5.6 model** (it remains valid on gpt-5/gpt-5-mini).
   Any code path that can send `minimal` to a 5.6 model will 400. Known paths:
   - `UTILITY_REASONING_EFFORT` defaults to `"minimal"` (`config.py`). With
     `UTILITY_MODEL=gpt-5.6-luna` this must become `none` (preferred for
     classifiers — zero reasoning tokens, cheapest output) or `low`.
   - `slack_client/event_handlers/settings.py` has minimal-handling logic
     (web-search upgrade minimal→low, restore-from-stored). Stored user settings
     may contain `minimal` — clamp/migrate to `low` (or `none`) when the selected
     model is 5.6.
2. **`max` is new, and Sol-only per all launch coverage.** OpenAI's guide lists the
   ladder family-wide without per-tier restrictions, so this is not 100% settled —
   safest: only offer `max` in the modal when model == `gpt-5.6-sol`, and verify
   empirically once (one cheap API call per tier) before finalizing the option
   lists for Terra/Luna.
3. `xhigh` exists on 5.6 (already in our modal options). OpenAI migration guidance:
   users on `xhigh` should compare against `max`; also test one effort level lower
   than current — 5.6 tends to hold quality with fewer reasoning tokens.
4. **`text.verbosity` is unchanged** (`low`/`medium`/`high`, verified in SDK types).
   Existing verbosity plumbing carries over as-is.
5. **temperature/top_p at `effort=none`:** gpt-5.5 allows temperature/top_p when
   `reasoning_effort=none` (current hybrid handling in
   `openai_client/api/responses.py`). **Whether 5.6 keeps this is undocumented.**
   Do NOT extend that branch to `gpt-5.6*` until verified with a live call; until
   then send `temperature=1.0` and no `top_p` for 5.6 at all efforts. If the test
   shows 5.6 accepts sampling params at `none`, mirror the 5.5 branch.

## New API surface (openai-python v2.45.0, verified in SDK source)

- `reasoning.effort` literal now includes `"max"` (full literal:
  `none | minimal | low | medium | high | xhigh | max`; per-model support varies).
- `reasoning.mode`: `"standard" | "pro"` — request-level "think harder" toggle,
  works on any 5.6 model, defaults to medium effort if effort unset. Optional —
  not needed for this upgrade, but could be a future settings toggle.
- `reasoning.context`: `"auto" | "current_turn" | "all_turns"` — controls which
  reasoning items are re-rendered on later turns. Only relevant for
  `previous_response_id` chaining; **we're stateless (`store=False`, full history
  in `input`), so omit it.**
- **`prompt_cache_retention` is deprecated** → replaced by
  `prompt_cache_options: {"mode": ..., "ttl": "30m"}` (`"30m"` is currently the
  only supported ttl). We currently send `prompt_cache_retention="24h"` for 5.x
  reasoning models in `openai_client/api/responses.py` (~line 90). For 5.6+,
  OpenAI auto-places one implicit cache breakpoint by default — **the simplest
  correct move is to send nothing for 5.6 models** (implicit caching just works),
  keep `prompt_cache_retention` for gpt-5.5 only, or migrate both to
  `prompt_cache_options`. Explicit breakpoints (`prompt_cache_breakpoint` on
  content blocks, up to 4 writes/request) are available but unnecessary for us.
- "Ultra mode" (marketing) = **Responses API multi-agent beta**:
  `client.beta.responses.create(model="gpt-5.6-sol", multi_agent={"enabled": True},
  betas=["responses_multi_agent=v1"])`. Sol-only, spawns ~4 parallel subagents,
  token spend increases by design. **Out of scope — do not implement.**
- Programmatic tool calling (JS-in-V8 tool) — new tool type, out of scope.

## Per-file change list

1. **`requirements.in`**: bump `openai>=2.0.0` → `openai>=2.45.0`, then `make lock`.
   Commit both `requirements.in` and `requirements.txt`.
2. **`config.py`**:
   - Supported-models comment (~line 29) and `MODEL_CUTOFFS` (~line 32): add the
     three 5.6 models with cutoff `"February 16, 2026"` (keep gpt-5.5 =
     "August 31, 2025").
   - `gpt_model` default (~line 54): `gpt-5.6-sol`.
   - `utility_model` default: `gpt-5.6-luna`; `utility_reasoning_effort` default
     `"minimal"` → `"none"` (or `"low"`).
   - `get_model_token_limit` (~line 358): `gpt-5.6*` gets the same 1.05M-total
     branch as `gpt-5.5` (same window, same 130K reserve).
   - Consider a small per-model effort validator/clamp helper (e.g.
     `clamp_effort(model, effort)`: `minimal`→`low` on 5.6; `max`→`xhigh` on
     non-Sol) so bad stored settings can never reach the API.
3. **`openai_client/api/responses.py`** (both text and streaming param builders):
   - `is_reasoning_model = "chat" not in model` heuristic still works (no 5.6
     chat variant) — no change needed.
   - The `startswith("gpt-5.4") or startswith("gpt-5.5")` effort-none
     temperature/top_p branch: leave 5.6 out pending the live test above.
   - Prompt caching block (`model in ["gpt-5.1", ...]` list): don't send
     `prompt_cache_retention` for 5.6 models (see API-surface section). The
     hardcoded model list only contains removed models plus gpt-5.5 — simplify.
   - Apply the effort clamp before building `reasoning": {"effort": ...}`.
4. **`settings_modal.py`**:
   - Model picker (~lines 348–356): replace the single option with four —
     Sol ("GPT-5.6 Sol"), Terra ("GPT-5.6 Terra"), Luna ("GPT-5.6 Luna"),
     GPT-5.5. Remove the force-to-gpt-5.5 normalization at ~lines 52–56
     (replace with "force to gpt-5.6-sol if stored model not in supported list").
   - Reasoning options (~lines 381–387: none/low/medium/high/xhigh): append
     `max` **only when the selected model is `gpt-5.6-sol`** (pending the
     empirical check; if Terra/Luna accept max too, offer it everywhere). The
     modal already reshapes on model change (`image_model` action pattern in
     `slack_client/event_handlers/settings.py` ~line 586) — model select needs
     the same reshape so the effort list updates per model.
   - Validation fallback: if stored effort not in the new option list (e.g.
     `max` after switching Sol→Terra, or legacy `minimal`), fall back per the
     clamp rules, not blindly to `none`.
5. **`slack_client/event_handlers/settings.py`**: update the minimal/web-search
   special-casing (~lines 705–727) — for 5.6 models `minimal` can never be
   stored/restored; map to `low` (web search on) or `none`.
6. **`.env.example`**: `GPT_MODEL=gpt-5.6-sol`, `UTILITY_MODEL=gpt-5.6-luna`,
   update the inline comments (they currently say "only supported chat model" /
   "gpt-5-mini only"). Check `WEB_SEARCH_MODEL` comment still makes sense.
7. **`CLAUDE.md`**: Model-Specific Parameters section + Responses API bullets
   currently describe a gpt-5.5 + gpt-5-mini world — update to the new pairing
   after implementation.
8. **Tests**: update fixtures/asserts pinned to `gpt-5.5`/`gpt-5-mini` defaults;
   add cases for the effort clamp (minimal→ none/low on 5.6, max rejected/
   remapped on non-Sol) and for the modal option lists per model.

## Verification checklist (one-time live checks, cheap)

- [x] `gpt-5.6-sol` + `reasoning.effort=max` → 200
- [x] `gpt-5.6-terra` / `gpt-5.6-luna` + `max` → confirm 400 (→ keep Sol-only) or 200 (→ offer everywhere)
- [x] any 5.6 model + `minimal` → confirm 400 (expected)
- [x] `gpt-5.6-sol` + `effort=none` + `temperature`/`top_p` → 200 or 400 (decides responses.py branch)
- [x] 5.6 request with no cache params → response shows cached tokens on second call (implicit caching works)
- [ ] `make test` passed post-implementation; live dev-bot smoke test in #chatgpt-bot-test still pending

## Verification results (2026-07-09, live API)

| Check | Result | Decision taken |
|---|---|---|
| `gpt-5.6-sol` + `max` | **200** | `max` offered |
| `gpt-5.6-terra` + `max` | **200** | `max` offered on ALL tiers (launch coverage saying Sol-only was wrong) |
| `gpt-5.6-luna` + `max` | **200** | same |
| `gpt-5.6-sol`/`-luna` + `minimal` | **400** ("Unsupported value: 'minimal'… Supported: none, low, medium, high, xhigh…") | `clamp_effort()` maps minimal→none on 5.6; note the error message's supported-list omits `max` even though `max` returns 200 — trust the probe, not the error copy |
| `gpt-5.6-sol` + `effort=none` + `temperature=0.7`/`top_p=0.9` | **200** (also verified on luna) | 5.5's hybrid temperature branch extended to `gpt-5.6*` |
| Implicit caching (two identical ~1.5K-token calls, zero cache params) | call 1 `cached_tokens=0`, call 2 **`cached_tokens=1509`** | 5.6 sends NO `prompt_cache_retention`; `prompt_cache_key` still sent for shard routing |
| `gpt-5.6-terra` + `effort=none` | **200** | ladder incl. `none` uniform across tiers |

## Sources

- https://openai.com/index/gpt-5-6/ and https://openai.com/index/previewing-gpt-5-6-sol/
- https://developers.openai.com/api/docs/models/gpt-5.6-sol (also /gpt-5.6-terra, /gpt-5.6-luna, /gpt-5.5, /gpt-5-mini)
- https://developers.openai.com/api/docs/guides/latest-model (Using GPT-5.6 / migration guidance)
- https://developers.openai.com/api/docs/guides/reasoning
- openai-python v2.45.0 source: `types/shared/reasoning_effort.py`, `types/shared/reasoning.py`, `types/shared/chat_model.py`, `types/responses/response_create_params.py`, `examples/responses/multi_agent_streaming.py`
- GPT-5.6 Preview System Card: https://deploymentsafety.openai.com/gpt-5-6-preview
