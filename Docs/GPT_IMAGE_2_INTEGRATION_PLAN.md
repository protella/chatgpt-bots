# GPT-Image-2 Integration Plan

## 1. Model Research — `gpt-image-2`

> **Status update (2026-05-08)**: API confirmed live. Re-research surfaced **breaking changes** vs gpt-image-1 — see "Breaking changes" below. Plan updated accordingly.

### Release
- **Date**: April 21, 2026 — "ChatGPT Images 2.0" livestream unveil
- **Web access**: April 22, 2026
- **Developer API**: Live as of early May 2026
- **Batch API**: Supported (50% discount)

### Identity
- **Primary model ID**: `gpt-image-2`
- **Dated snapshot**: `gpt-image-2-2026-04-21`
- **Mini variant**: No `gpt-image-2-mini` exists. `gpt-image-1-mini` remains as cheaper v1 option.
- **Architecture**: Native multimodal in GPT architecture (not standalone diffusion)
- **Highlights**:
  - Agentic reasoning — plans + researches image structure before drawing
  - ~99% typography accuracy, strong multilingual (CJK, Hindi, Bengali, Arabic)
  - ~2× speed vs gpt-image-1
  - Up to 8 consistent panels per prompt (characters / layout / palette coherence)
  - Optional web search integration for fact-checking

### Endpoints (unchanged from v1)
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- Batch API supported

### ⚠️ Breaking Changes vs gpt-image-1

| Change | Detail | Code impact |
|---|---|---|
| `background: transparent` **removed** | Only `auto` and `opaque` supported on v2. Transparent backgrounds NOT generated. | Filter param before send; if user picked transparent + v2 model, fall back to `auto` |
| `input_fidelity` **removed (auto)** | Model handles fidelity internally; sending the param may be ignored or error | Drop param when model == v2 |
| `output_format` default changed | Some sources report default = `webp` (was `png` in v1) | Be explicit — always send `output_format` rather than rely on default |
| Pricing model | Token-based; output rate $30/M (down from $32 for v1.5) | No code change, log token usage |

### Parameter Shape (gpt-image-2)
| Param | Values | Notes |
|---|---|---|
| `model` | `"gpt-image-2"` | |
| `prompt` | string, up to ~32K tokens | |
| `size` | `1024x1024`, `1024x1536`, `1536x1024`, `auto`; flexible up to long-edge ~3840px | Both edges multiples of 16; aspect ratio < 3:1; total pixels 655,360–8,294,400 |
| `quality` | `low`, `medium`, `high`, `auto` | Same as v1 |
| `n` | 1–8 | Multi-panel consistency new |
| `output_format` | `png`, `jpeg`, `webp` | Be explicit |
| `output_compression` | 0–100 | JPEG/WebP only |
| `background` | `auto`, `opaque` ONLY (no `transparent`) | ⚠️ |
| `moderation` | `auto` (default), `low` | |
| `mask` | file | Edits endpoint |
| `input_fidelity` | n/a — auto | ⚠️ Don't send for v2 |

**Possible new param** (sources conflict): `quality_mode` = `instant` | `thinking` (instant default; thinking ~2–3× cost). Not confirmed in OpenAI dev docs page; treat as optional advanced feature, ignore for now.

**Unsupported**: streaming, function calling, structured outputs, fine-tuning, predicted outputs.

### Response Shape
Unchanged. `response.data[0].b64_json` populated. URL fallback still supported. URLs expire ~2 hours.

### Pricing (token-based)
- Input text: $5 / M tokens
- Output text: $10 / M tokens
- Input image (edits): $8 / M tokens, $2 / M cached
- Output image: $30 / M tokens
- Per-image approx (1024×1024): low ~$0.006, medium ~$0.053, high ~$0.211

### Rate Limits (Images Per Minute)
| Tier | IPM |
|---|---|
| 1 | 5 |
| 2 | 20 |
| 3 | 50 |
| 4 | 150 |
| 5 | 250 |

### Verdict
**Mostly compatible — but NOT pure drop-in.** Swap model string + add conditional param filtering for `background=transparent` and `input_fidelity`. Existing happy-path calls work unchanged. Implementation must guard those two fields when model == v2.

---

## 2. Current Code State

### Where model ID flows today
- `config.py:55` — `image_model` loaded from `GPT_IMAGE_MODEL` env var, default `"gpt-image-1"`
- `config.py:231` — `get_thread_config()` already surfaces `image_model` in returned dict
- `config.py:286–292` — user-preferences mapping does **NOT** currently include `image_model`
- `openai_client/api/images.py:57` (`generate_image`) — reads `config.image_model` **directly**, bypassing thread_config
- `openai_client/api/images.py:431` (`edit_image`) — same direct read
- `message_processor/handlers/image_gen.py:122,190` — builds thread_config, passes `size`/`quality`/`background` but NOT `model`
- `message_processor/handlers/image_edit.py:262,292,603,646` — four `edit_image` call sites, none pass `model`

### Settings UI pattern to mirror
- `settings_modal.py:229–254` — `model_block` / `model_select` static_select for chat model
- `settings_modal.py:941–1000` — existing `image_size_block`, `image_quality_block`, `image_background_block` pattern
- `settings_modal.py:1177–1193` — extraction handler for image settings

### Persistence
- `database.py:~203` — `user_preferences` schema with `image_size`, `image_quality`, `image_background`, `input_fidelity`, `vision_detail`
- `database.py:~1882` — upsert logic for user_preferences

---

## 3. Implementation Plan

### Phase 1 — Config + env
**File: `config.py`**
- Line 55: change default from `"gpt-image-1"` → `"gpt-image-2"` (via `GPT_IMAGE_MODEL` env var)
- Lines 286–292 (inside `get_thread_config` user-prefs mapping): add
  ```python
  if user_prefs.get('image_model'):
      user_config['image_model'] = user_prefs['image_model']
  ```
- Docstring at line 76: drop "gpt-image-1.5" reference

**File: `.env.example`**
- Set `GPT_IMAGE_MODEL=gpt-image-2`
- Add comment noting both models supported, user override via `/settings`

### Phase 2 — Database schema
**File: `database.py`**
- Add column to `user_preferences` schema (~line 203):
  ```sql
  image_model TEXT DEFAULT 'gpt-image-2'
  ```
- Migration: `ALTER TABLE user_preferences ADD COLUMN image_model TEXT DEFAULT 'gpt-image-2'` wrapped in try/except for existing DBs
- Extend `get_user_preferences()` SELECT to include `image_model`
- Extend `upsert_user_preferences()` INSERT/UPDATE to include `image_model`

### Phase 3 — API client pass-through (with v2 param guards)
**File: `openai_client/api/images.py`**

`generate_image()`:
- Signature: add `model: Optional[str] = None`
- Resolve `effective_model = model or config.image_model`
- Line 57: `"model": effective_model`
- **NEW**: if `effective_model.startswith("gpt-image-2")` and `background == "transparent"` → coerce to `"auto"` and log warning (`"transparent background not supported on gpt-image-2, falling back to auto"`)

`edit_image()`:
- Signature: add `model: Optional[str] = None`
- Resolve `effective_model = model or config.image_model`
- Line 431: `"model": effective_model`
- **NEW**: if `effective_model.startswith("gpt-image-2")`:
  - skip `input_fidelity` from params dict (model auto-handles)
  - apply same `background=transparent` → `auto` coercion
- Update docstrings (lines 24, 377): drop "gpt-image-1.5"; say "OpenAI image model (gpt-image-1, gpt-image-1-mini, gpt-image-2)"

**Helper to add** (top of `images.py`):
```python
def _is_v2(model_id: str) -> bool:
    return model_id.startswith("gpt-image-2")
```

### Phase 4 — Handler wiring
**File: `message_processor/handlers/image_gen.py`**
- Lines 122–129 (streaming branch): add `model=thread_config.get("image_model")`
- Lines 190–197 (non-streaming branch): same

**File: `message_processor/handlers/image_edit.py`**
- Lines 262, 292, 603, 646 (four `edit_image` call sites): add `model=thread_config.get("image_model")` to each
- Confirm `thread_config` built upstream in each code path (already uses `config.get_thread_config`)

### Phase 5 — Settings modal dropdown
**File: `settings_modal.py`**

**Add block** (mirror `model_block` pattern lines 229–254, place in common/image settings section near line 940):
```python
blocks.append({
    "type": "section",
    "block_id": "image_model_block",
    "text": {
        "type": "mrkdwn",
        "text": "*Image Model*\nChoose image generation model"
    },
    "accessory": {
        "type": "static_select",
        "action_id": "image_model",
        "placeholder": {"type": "plain_text", "text": "Select image model"},
        "initial_option": {
            "text": {"type": "plain_text", "text": self._get_image_model_display_name(selected_image_model)},
            "value": selected_image_model
        },
        "options": [
            {"text": {"type": "plain_text", "text": "GPT Image 2 (latest)"}, "value": "gpt-image-2"},
            {"text": {"type": "plain_text", "text": "GPT Image 1"}, "value": "gpt-image-1"}
        ]
    }
})
```

**Add helper** `_get_image_model_display_name(model_id)`:
- Maps `gpt-image-2` → "GPT Image 2 (latest)", `gpt-image-1` → "GPT Image 1"

**UI hint for v2 limitations**: When `selected_image_model == "gpt-image-2"`, hide or disable the `transparent` option in the background dropdown (or show context note: "Transparent backgrounds available on GPT Image 1 only").

**Extraction handler** (~line 1193, append after `image_background` extraction):
```python
image_model_block = values.get('image_model_block', {})
if image_model_block:
    selected = image_model_block['image_model'].get('selected_option')
    if selected:
        extracted['image_model'] = selected['value']
```

**Settings read** (wherever `settings.get('image_size', ...)` pattern appears): add `selected_image_model = settings.get('image_model', config.image_model)` near top of the builder.

### Phase 6 — Verification
- `make test` — unit tests pass
- Manual test path:
  1. Run slackbot
  2. Open `/settings` — confirm Image Model dropdown appears with default = `gpt-image-2`
  3. Select `gpt-image-1`, save, generate image → verify API call uses `gpt-image-1` (log inspection)
  4. Switch to `gpt-image-2`, save, generate → verify `gpt-image-2`
  5. Edit an existing image → same verification path
- Fallback check: if API returns `model_not_found` for `gpt-image-2` (access not yet granted), surface friendly error + suggest switching to v1 in settings

---

## 4. Rollout Risk + Mitigation

**Risk 1** (resolved): API access. Confirmed live as of 2026-05-08.

**Risk 2**: `background=transparent` rejected by v2.
- **Mitigation**: Param-guard in `images.py` (Phase 3) coerces to `auto` + logs. UI also hides transparent option when v2 selected.

**Risk 3**: `input_fidelity` ignored / errors on v2.
- **Mitigation**: Param-guard drops field when model is v2. v2 auto-handles fidelity.

**Risk 4**: `output_format` default may differ (webp vs png).
- **Mitigation**: Always send `output_format` explicitly — never rely on server default. Existing `edit_image` already does this; verify `generate_image` does too.

**Risk 5**: Existing DB rows lack `image_model` column.
- **Mitigation**: Schema migration with `ALTER TABLE … ADD COLUMN … DEFAULT 'gpt-image-2'` — wrap in try/except for idempotency.

**Risk 6**: Cost surprise — v2 high-quality at $0.211/image, output tokens $30/M.
- **Mitigation**: No code change. Flag in CHANGELOG. Users already control quality via dropdown.

**Risk 7**: Size constraint changes (multiple-of-16, total pixels 655K–8.3M).
- **Mitigation**: Existing dropdown sizes (`1024x1024`, `1024x1536`, `1536x1024`) all comply. No code change unless adding new size options.

**Risk 8**: `quality_mode` (instant/thinking) — sources conflict on existence.
- **Mitigation**: Don't expose. Default behavior = instant. Revisit if users request thinking mode.

---

## 5. Files Touched Summary

| File | Change |
|---|---|
| `config.py` | Default env → `gpt-image-2`; add user-pref mapping for `image_model` |
| `.env.example` | `GPT_IMAGE_MODEL=gpt-image-2` |
| `database.py` | Schema + migration + getter/setter for `image_model` column |
| `openai_client/api/images.py` | Add `model` kwarg to `generate_image` + `edit_image` |
| `message_processor/handlers/image_gen.py` | Pass `model=thread_config.get("image_model")` at 2 call sites |
| `message_processor/handlers/image_edit.py` | Pass `model=thread_config.get("image_model")` at 4 call sites |
| `settings_modal.py` | New dropdown block + extraction + display helper |
| `CHANGELOG.md` | New version entry noting v2 support + toggle |

---

## 6. Sources

- [Introducing ChatGPT Images 2.0 — OpenAI](https://openai.com/index/introducing-chatgpt-images-2-0/)
- [GPT Image 2 Model — OpenAI API docs](https://developers.openai.com/api/docs/models/gpt-image-2)
- [Changelog — OpenAI API](https://developers.openai.com/api/docs/changelog)
- [ChatGPT Images 2.0: Full Developer Breakdown — buildfastwithai](https://www.buildfastwithai.com/blogs/chatgpt-images-2-0-gpt-image-2-2026)
- [GPT-image-2 officially released — Apiyi](https://help.apiyi.com/en/gpt-image-2-official-launch-beginner-complete-guide-en.html)
- [GPT Image 2 API Guide — WaveSpeed](https://wavespeed.ai/blog/posts/gpt-image-2-api-guide/) (param constraints, transparent-bg unsupported, input_fidelity automatic)
- [gpt-image-2 API Developer Guide — TokenMix](https://tokenmix.ai/blog/gpt-image-2-api-developer-guide-2026) (pricing, possible quality_mode)
- [GPT Image 2 — Replicate proxy listing](https://replicate.com/openai/gpt-image-2) (parameter schema mirror)
- [OpenAI Highlights New ChatGPT Images 2.0 Model and API Availability — TipRanks](https://www.tipranks.com/news/private-companies/openai-highlights-new-chatgpt-images-2-0-model-and-api-availability)
