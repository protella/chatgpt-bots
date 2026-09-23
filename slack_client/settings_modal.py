"""
User Settings Modal for Slack Bot
Handles the interactive settings configuration interface
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from message_processor.image_service import SHAPE_TIER_SIZES
from config import config
from logger import LoggerMixin
import json
import re
import uuid


# The shape × tier grid lives in image_service (it resolves model-chosen shapes at call time
# too); this is the same object, not a copy.
_SHAPE_TIER_SIZES = SHAPE_TIER_SIZES

def _aspect_label(w: int, h: int) -> Optional[str]:
    """The aspect ratio a size is known by, or None when it has no readable name.

    Only the ratios people actually say out loud [OWNER 2026-09-09: "1:1, 4:3, 16:9, etc."]: a
    grid cell answers with its shape key (1920x1088 is the `16:9` cell even though the /16 snap
    makes it 1.76:1 exactly), and any other size snaps to the nearest named ratio when it is
    within a few percent of one. Nothing is ever reduced by arithmetic — `17:11` is not a name."""
    key = f"{w}x{h}"
    for shape, row in _SHAPE_TIER_SIZES.items():
        if key in row.values():
            return shape
    actual = w / h
    best: Optional[str] = None
    best_err = 0.03  # 3%: 1920x1088 is 1.6% off 16:9, 1200x800 is exactly 3:2
    for name in _FRIENDLY_RATIOS:
        a, b = (int(t) for t in name.split(":"))
        err = abs(actual - a / b) / (a / b)
        if err < best_err:
            best, best_err = name, err
    return best


# The ratios a person recognises, both orientations. Order matters only for tie-breaks, which
# the 3% tolerance makes practically impossible between neighbours this far apart.
_FRIENDLY_RATIOS: Tuple[str, ...] = (
    "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "16:10", "10:16", "5:4", "4:5",
    "21:9", "9:21", "2:1", "1:2", "3:1", "1:3",
)


# Display order for the shape select. `auto` first because it is the "let the model decide" answer.
_IMAGE_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("1:1", "Square (1:1)"),
    ("3:2", "Landscape (3:2)"),
    ("2:3", "Portrait (2:3)"),
    ("16:9", "Widescreen (16:9)"),
    ("9:16", "Tall (9:16)"),
    ("3:1", "Panorama (3:1)"),
    ("1:3", "Skyscraper (1:3)"),
)

# No pixel counts in the labels — the resolution rides the context line underneath instead.
_IMAGE_TIERS: Tuple[Tuple[str, str], ...] = (
    ("standard", "Standard"),
    ("large", "Large — more detail"),
    # PULLED 2026-09-09 by the owner, with the `max` column of `image_service.SHAPE_TIER_SIZES`
    # and the 4K half of the custom-size envelope. OpenAI's image guide says "Resolutions above
    # 2560x1440 are experimental"
    # (https://developers.openai.com/api/docs/guides/image-generation), and at those sizes
    # `quality=high` renders visible mesh artifacts (measured 2026-09-09). Restore when the docs
    # drop the experimental label:
    #     ("max", "Maximum — 4K"),
)

# gpt-image-1 takes only the three named sizes, which are exactly the `standard` column of three
# shapes. It gets no tier select at all, so the other four shapes have nothing to resolve to.
_V1_SHAPES: Tuple[str, ...] = ("auto", "1:1", "3:2", "2:3")

_WXH_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")


class SettingsModal(LoggerMixin):
    """Manages the user settings modal interface"""
    
    def __init__(self, db):
        """Initialize with database connection"""
        self.db = db
        self.logger_name = "SettingsModal"
    
    async def build_settings_modal(self, user_id: str, trigger_id: str,
                            current_settings: Optional[Dict] = None,
                            is_new_user: bool = False,
                            thread_id: Optional[str] = None,
                            in_thread: bool = False,
                            scope: Optional[str] = None,
                            pending_message: Optional[Dict] = None,
                            user_memory_value: Optional[str] = None,
                            user_mem_seed: Optional[List] = None,
                            user_memory_forget_all: bool = False) -> Dict:
        """
        Build the complete settings modal.

        Args:
            user_id: Slack user ID
            trigger_id: Slack trigger ID for modal
            current_settings: Current user settings
            is_new_user: Whether this is a new user's first setup
            thread_id: Thread ID if opened from within a thread
            in_thread: Whether modal was opened from within a thread
            scope: Selected scope ('thread' or 'global')
            pending_message: Pending message to process after settings save (for new users)
            user_memory_value / user_mem_seed / user_memory_forget_all: the personal-memory box's
                state on a RE-RENDER (a model change, a scope toggle — anything that rebuilds the
                view with views_update). Both None means a FRESH open: the value and the
                open-time seed ``[[id, hash], ...]`` are computed from the user's rows here. On a
                re-render the caller hands both back verbatim, so an in-flight edit survives and
                the seed stays anchored to the rows the user first saw — the same bargain the
                channel modal makes, except the seed rides the `modal_sessions` row instead of
                `private_metadata`, which is already at its size limit here.

        Returns:
            Modal view dictionary for Slack API
        """
        if not current_settings:
            current_settings = await self.db.get_user_preferences_async(user_id)
            if not current_settings:
                # Get user's email from users table
                user_data = await self.db.get_or_create_user_async(user_id)
                email = user_data.get('email') if user_data else None
                current_settings = await self.db.create_default_user_preferences_async(user_id, email)
        
        # Determine which model is selected. Coerce any stale/dropped model value
        # (e.g. an old thread override) to gpt-5.6-sol so the picker's initial_option
        # is always a valid option.
        from config import SUPPORTED_CHAT_MODELS
        selected_model = current_settings.get('model', config.gpt_model)
        if selected_model not in SUPPORTED_CHAT_MODELS:
            selected_model = 'gpt-5.6-sol'
        
        # Determine default scope if not provided
        if scope is None:
            # New users should always default to global settings
            if is_new_user:
                scope = 'global'
            else:
                scope = 'thread' if in_thread else 'global'
        
        # Personal memory (T2). Off → no section, no seed, nothing to reconcile on submit. A DB
        # hiccup degrades to an empty box rather than blocking the modal, and an empty SEED is
        # what stops that from reading as "the user deleted everything" on save.
        user_memory: Optional[Dict[str, Any]] = None
        if config.enable_user_memory:
            hidden_count = 0
            try:
                rows = list(await self.db.get_user_memory_async(user_id) or [])
            except Exception as e:
                self.log_error(f"Failed to load user memory for {user_id}: {e}")
                rows = []
            if user_memory_value is None and user_mem_seed is None:
                user_memory_value, user_mem_seed, hidden_count = self._compute_memory_seed(rows)
            else:
                # Re-render: the value and seed come back verbatim, so "+N more" is derived from
                # what the seed omits (normalize returns "" for whitespace-only, so a bare strip
                # test matches the fresh-open blank drop).
                user_mem_seed = user_mem_seed or []
                user_memory_value = user_memory_value or ""
                non_blank = sum(1 for m in rows if (m.get("content") or "").strip())
                hidden_count = max(0, non_blank - len(user_mem_seed))
            user_memory = {"value": user_memory_value, "hidden": hidden_count,
                           "forget_all": user_memory_forget_all}

        # Build modal blocks
        blocks = self._build_modal_blocks(current_settings, selected_model, is_new_user, in_thread,
                                          scope, user_memory)
        
        # Determine callback ID based on user status
        callback_id = "welcome_settings_modal" if is_new_user else "settings_modal"

        # Determine if we're in dev environment
        is_dev = config.settings_slash_command.endswith("-dev")
        # Slack modal titles have a 24 character limit
        modal_title = "ChatGPT Settings (Dev)" if is_dev else "ChatGPT Bot Settings"

        # Create session for modal state storage
        session_id = str(uuid.uuid4())

        # Build full state to store in DB
        session_state = {
            "settings": current_settings,
            "thread_id": thread_id,
            "in_thread": in_thread,
            "scope": scope
        }

        # Include pending message if provided (for new users)
        if pending_message:
            session_state["pending_message"] = pending_message

        # The personal-memory seed rides the session row, not private_metadata (which holds only
        # the session id here — the user modal's metadata is already at its size limit). Submit
        # reconciles against EXACTLY these rows, so it can never delete one the user never saw.
        if user_memory is not None:
            session_state["user_mem_seed"] = user_mem_seed or []

        # Store session in database
        await self.db.create_modal_session_async(session_id, user_id, session_state, modal_type='settings')

        # Only store session_id in metadata - much smaller!
        metadata = {
            "session_id": session_id
        }

        return {
            "type": "modal",
            "callback_id": callback_id,
            "title": {"type": "plain_text", "text": modal_title},
            "submit": {"type": "plain_text", "text": "Save Settings"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": blocks,
            "private_metadata": json.dumps(metadata)
        }
    
    def build_channel_settings_modal(self, channel_id: str, current_settings: Optional[Dict],
                                     global_default_mode: str,
                                     channel_memories: Optional[List[Dict]] = None,
                                     memory_textarea_value: Optional[str] = None,
                                     mem_seed: Optional[List] = None,
                                     channel_policy: Optional[Dict] = None,
                                     policy_value: Optional[str] = None,
                                     policy_seed: Optional[str] = None) -> Dict:
        """Build the per-channel settings modal (Phase 7).

        `current_settings` is the DB row (or None). A NULL/absent response_mode means the channel
        inherits the global default; that is represented by the "inherit" option here, and the
        submission handler stores None (NULL) for it so the global default keeps applying.

        `channel_memories` (from ``get_channel_memory_async``) drives the memory sections at the
        bottom: channel-scope facts become ONE editable multiline textarea; workspace-scope facts
        render read-only below it. On a FRESH open (`memory_textarea_value`/`mem_seed` both None)
        the textarea value and the open-time seed ``[[id, hash], ...]`` are computed from the rows.
        On a RE-RENDER (model change / views_update) the caller passes both verbatim so an in-flight
        edit survives and the seed stays anchored to the rows the user first saw. `mem_seed` rides in
        `private_metadata` so the submit handler can reconcile exactly those rows. All optional so
        pure builder tests can render without a DB.

        `channel_policy` (from ``get_channel_policy_async``) backs the standing-policy box, and
        its open-time content hash rides in `private_metadata` as `policy_seed` so the submit
        handler can refuse to overwrite a policy someone else changed while this was open — the
        same seed-and-hash bargain the memory textarea makes.
        """
        from message_processor.channel_steering import POLICY_MAX_CHARS
        from message_processor.participation import MODE_TO_LEVEL, VALID_LEVELS

        from config import (SUPPORTED_CHAT_MODELS, SUPPORTED_IMAGE_MODELS,
                            SUPPORTED_VERBOSITIES, clamp_effort, effective_channel_model,
                            effort_ladder)

        cs = current_settings or {}

        # The standing channel policy lives in its own reserved row, NOT in the settings row —
        # it is an operator instruction, and the memory tools, the fact cap and the fallback
        # extractor must all be unable to reach it. On a FRESH open both the box's value and the
        # open-time hash come from that row; on a RE-RENDER the caller hands both back, so an
        # in-flight edit survives and the hash stays anchored to what the user first saw.
        from database import memory_content_hash
        stored_policy = ((channel_policy or {}).get("content") or "").strip()
        if policy_value is None and policy_seed is None:
            policy_value = stored_policy
            policy_seed = memory_content_hash(stored_policy) if stored_policy else ""

        def _select(action_id, label_map, current, inherit_label):
            """Static select with an 'inherit' first option; initial = current or inherit."""
            options = [{"text": {"type": "plain_text", "text": inherit_label}, "value": "inherit"}]
            options += [{"text": {"type": "plain_text", "text": text}, "value": value}
                        for value, text in label_map]
            selected = current if current in {v for v, _ in label_map} else "inherit"
            initial = next(o for o in options if o["value"] == selected)
            return {"type": "static_select", "action_id": action_id,
                    "options": options, "initial_option": initial}

        # Every one of the six capability controls inherits from the BOT'S CONFIGURATION, not from
        # the asker: on a channel turn the capability keys leave the per-user hierarchy entirely
        # (config.CHANNEL_CAPABILITY_KEYS), so an "each person's own setting" label described a
        # fallback that cannot happen here. The placeholder names the value the channel is
        # actually running on, which is the only way to read a select sitting on "inherit".
        def _workspace_default(current) -> str:
            return f"Use the workspace default (currently: {current})"

        def _tri_state(value):
            """A stored capability tri-state as the select's value: 1 → on, 0 → off, NULL → inherit.

            Anything else can only be a legacy row written before the column's CHECK existed. It
            renders as inherit because that is what the resolver does with it — showing it as ON
            would tell the operator the channel is running a setting it is not.
            """
            if value in (0, 1):
                return "on" if value else "off"
            return None

        model_element = _select(
            "channel_model",
            [(m, m) for m in SUPPORTED_CHAT_MODELS],
            cs.get("model"),
            _workspace_default(config.gpt_model),
        )
        # The effort ladder follows the model this channel will ACTUALLY run on, which is the
        # stored one only while it is still selectable — otherwise the workspace default. Reading
        # the raw stored value instead offered the full 5.6 ladder to a channel that inherits its
        # model, so on a workspace running gpt-5.5 the modal presented `max` and the resolver then
        # refused it. Same call as the resolver makes, so the two cannot drift apart again.
        #
        # The inherit label is clamped against that same model for the same reason: the workspace
        # default effort is not necessarily one this channel's model accepts, and naming a value
        # the channel cannot run is the disagreement in a different place.
        effective_model = effective_channel_model(cs.get("model"), config.gpt_model)
        effort_element = _select(
            "channel_reasoning_effort",
            [(e, e) for e in effort_ladder(effective_model)],
            cs.get("reasoning_effort"),
            _workspace_default(clamp_effort(effective_model, config.default_reasoning_effort)),
        )
        verbosity_element = _select(
            "channel_verbosity",
            [(v, v) for v in SUPPORTED_VERBOSITIES],
            cs.get("verbosity"),
            _workspace_default(config.default_verbosity),
        )
        on_off = [("on", "On"), ("off", "Off")]
        web_search_element = _select(
            "channel_enable_web_search",
            on_off,
            _tri_state(cs.get("enable_web_search")),
            _workspace_default("on" if config.enable_web_search else "off"),
        )
        mcp_element = _select(
            "channel_enable_mcp",
            on_off,
            _tri_state(cs.get("enable_mcp")),
            _workspace_default("on" if config.mcp_enabled_default else "off"),
        )
        image_model_element = _select(
            "channel_image_model",
            [(m, m) for m in SUPPORTED_IMAGE_MODELS],
            cs.get("image_model"),
            _workspace_default(config.image_model),
        )

        # Phase F: one Participation select replaces the old response-mode select.
        # Legacy rows with only response_mode map cleanly (off≡off, tag_only≡mentions_only,
        # auto_respond≡on); submission writes BOTH columns in lockstep.
        #
        # Three options, not four. `judicious` and `active` were labelled as restraint dials
        # ("chime in when clearly valuable" vs. "participate more freely"), but the gate they tuned
        # is now one bit: it decides whether the responder WAKES, not how eagerly it talks. Offering
        # two names for one behavior would have sold the user a knob that turns nothing, so they
        # collapse into `on`. The labels below therefore describe WHEN I wake, never how chatty I am.
        global_default_level = MODE_TO_LEVEL.get((global_default_mode or "tag_only").lower(), "mentions_only")
        mode_options = [
            {"text": {"type": "plain_text", "text": f"Use default (currently: {global_default_level})"},
             "value": "inherit"},
            {"text": {"type": "plain_text", "text": "Mentions only"},
             "value": "mentions_only"},
            {"text": {"type": "plain_text", "text": "On — joins in when it can help (recommended)"},
             "value": "on"},
            {"text": {"type": "plain_text", "text": "Off — never responds here"},
             "value": "off"},
        ]
        current_level = cs.get("participation_level")
        if current_level not in VALID_LEVELS:
            # Fall back to the legacy column when only response_mode was ever set.
            current_level = MODE_TO_LEVEL.get(cs.get("response_mode") or "", None)
        selected_value = current_level if current_level in VALID_LEVELS else "inherit"
        initial_mode_option = next(o for o in mode_options if o["value"] == selected_value)

        # Reply placement is a TRI-STATE control (SHOULD-FIX #5): a stored None means "inherit the
        # workspace default", True means "reply at channel level", False means "threads only". The
        # old binary checkbox resolved NULL to today's global default, so merely opening + saving an
        # inheriting channel FROZE that default into an explicit row. These three options map
        # straight back to None / True / False on submit, so an untouched inheriting channel stays
        # NULL (still inheriting) and future global-config changes keep flowing through.
        ric_value = cs.get("reply_in_channel")  # None (inherit) | True | False
        default_placement_text = ("reply at channel level" if config.reply_in_channel_default
                                  else "threads only")
        placement_options = [
            {"text": {"type": "plain_text",
                      "text": f"Inherit workspace default (currently: {default_placement_text})"},
             "value": "inherit"},
            {"text": {"type": "plain_text", "text": "Reply at channel level"}, "value": "channel"},
            {"text": {"type": "plain_text", "text": "Threads only"}, "value": "threads"},
        ]
        if ric_value is None:
            placement_selected = "inherit"
        elif ric_value:
            placement_selected = "channel"
        else:
            placement_selected = "threads"
        reply_element = {
            "type": "static_select", "action_id": "reply_in_channel",
            "options": placement_options,
            "initial_option": next(o for o in placement_options if o["value"] == placement_selected),
        }

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn",
             "text": f"*Channel settings* for <#{channel_id}>"},
             "accessory": {"type": "button", "action_id": "open_user_settings_push",
                           "text": {"type": "plain_text", "text": "👤 My personal settings"}}},
            {"type": "input", "block_id": "participation_block",
             "element": {"type": "static_select", "action_id": "participation_level",
                         "options": mode_options, "initial_option": initial_mode_option},
             "label": {"type": "plain_text", "text": "Participation"},
             "hint": {"type": "plain_text",
                      "text": "'Off' ignores even @mentions — reopen this menu to turn me back on."}},
            {"type": "input", "block_id": "policy_block", "optional": True,
             "element": {"type": "plain_text_input", "action_id": "standing_policy", "multiline": True,
                         "initial_value": policy_value or "", "max_length": POLICY_MAX_CHARS,
                         "placeholder": {"type": "plain_text",
                                         "text": "e.g. Only jump in on deploy failures; otherwise stay quiet."}},
             "label": {"type": "plain_text", "text": "Standing channel policy"},
             "hint": {"type": "plain_text", "text": "Saving replaces the whole policy; empty it to clear."}},
            {"type": "input", "block_id": "reply_in_channel_block", "optional": True,
             "element": reply_element,
             "label": {"type": "plain_text", "text": "Reply placement"},
             "hint": {"type": "plain_text",
                      "text": "'Reply at channel level' still lets me pick a thread when one fits better."}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn",
             "text": "*Shared response settings*\nThese apply to everyone in this channel."}},
            {"type": "input", "block_id": "channel_model_block", "dispatch_action": True,
             "element": model_element,
             "label": {"type": "plain_text", "text": "Model"}},
            {"type": "input", "block_id": "channel_effort_block",
             "element": effort_element,
             "label": {"type": "plain_text", "text": "Reasoning effort"},
             "hint": {"type": "plain_text",
                      "text": "Only efforts this channel's model supports are listed."}},
            {"type": "input", "block_id": "channel_verbosity_block",
             "element": verbosity_element,
             "label": {"type": "plain_text", "text": "Verbosity"}},
            {"type": "input", "block_id": "channel_web_search_block",
             "element": web_search_element,
             "label": {"type": "plain_text", "text": "Web search"}},
            {"type": "input", "block_id": "channel_mcp_block",
             "element": mcp_element,
             "label": {"type": "plain_text", "text": "MCP servers"}},
            {"type": "input", "block_id": "channel_image_model_block",
             "element": image_model_element,
             "label": {"type": "plain_text", "text": "Image model"}},
        ]

        # Channel-memory editor + read-only workspace-shared list. On a fresh open we derive the
        # textarea value and the open-time seed from the DB rows; on a re-render the caller hands both
        # back verbatim so an in-flight edit survives. The seed lists EXACTLY the rows shown in the box
        # and rides in private_metadata so submit reconciles only what the user could actually see.
        memories = channel_memories or []
        # Recorded participation preferences are excluded: they are the gate's own steering, the
        # box REPLACES what it shows, and a row the operator never saw must not be deletable by
        # editing around it. The reserved policy row is not here at all — it has its own field.
        from message_processor.channel_steering import is_ordinary_fact
        channel_rows = [m for m in memories
                        if m.get("scope") == "channel" and is_ordinary_fact(m)]
        workspace_rows = [m for m in memories if m.get("scope") != "channel"]

        if memory_textarea_value is None and mem_seed is None:
            memory_textarea_value, mem_seed, hidden_count = self._compute_memory_seed(channel_rows)
        else:
            mem_seed = mem_seed or []
            memory_textarea_value = memory_textarea_value or ""
            # Seed is carried verbatim on re-render; derive "+N more" from what it omits (normalize
            # returns "" for whitespace-only, so a bare strip test matches the fresh-open blank drop).
            non_blank = sum(1 for m in channel_rows if (m.get("content") or "").strip())
            hidden_count = max(0, non_blank - len(mem_seed))

        blocks.append({"type": "divider"})
        blocks.extend(self._build_channel_memory_blocks(
            memory_textarea_value, hidden_count, workspace_rows))

        return {
            "type": "modal",
            "callback_id": "channel_settings_modal",
            "title": {"type": "plain_text", "text": "Channel Settings"},
            "submit": {"type": "plain_text", "text": "Save"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "private_metadata": json.dumps({"channel_id": channel_id, "mem_seed": mem_seed,
                                            "policy_seed": policy_seed or ""}),
            "blocks": blocks,
        }

    # Read-only workspace-shared memories are one block per item; cap the list so a workspace with a
    # lot of shared facts can't blow Slack's 100-block modal limit. Channel-scope memory is a single
    # textarea now, so it needs no per-item cap — only the 2900-char textarea budget below.
    _MODAL_LIST_CAP = 10

    # Slack plain_text_input's max we build the channel-memory textarea against (value + budget
    # guard). It IS the memory store's budget (config.memory_store_max_chars) — the tools refuse a
    # write that would not fit this box, so the two must be one number — clamped to 2900 because
    # Slack hard-caps the element at 3000 and a bigger MEMORY_STORE_MAX_CHARS must not silently
    # start hiding notes again.
    _SLACK_TEXTAREA_CEILING = 2900

    @property
    def _MEMORY_TEXTAREA_MAX(self) -> int:
        return min(config.memory_store_max_chars, self._SLACK_TEXTAREA_CEILING)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Trim to `limit` chars with an ellipsis so long facts/reasons stay on one line."""
        text = (text or "").strip()
        return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

    def _compute_memory_seed(self, rows: List[Dict]):
        """From memory rows (oldest-first, as the `get_*_memory_async` accessors return them),
        build the textarea `initial_value` and the open-time seed ``[[id, hash], ...]``.

        Shared by the channel modal's channel-memory box and the user modal's personal-memory box:
        both stores hand back ``id``/``content`` rows and both textareas make the same bargain, so
        one budget rule serves both and they cannot drift.

        Each row is normalized and blank rows are dropped. Rows are included oldest-first only while
        the joined value stays within the textarea budget; on the first row that would overflow we
        stop and the rest become the "+N more not shown" remainder. The seed lists EXACTLY the
        included rows — never seed a row that isn't in the box, or submit could "delete" a row the
        user never saw. Returns ``(initial_value, mem_seed, hidden_count)``.

        The overflow path is the LEGACY path now: the memory tools refuse any write that would put
        a store past this same budget, so a store written since 2026-08-20 fits the box whole and
        `hidden_count` is normally zero. It stays for stores that filled up under the old row cap,
        until their owners consolidate them. (The read-only workspace-shared list is unaffected —
        it has its own `_MODAL_LIST_CAP` and is not part of any store's budget.)
        """
        from database import normalize_memory_line, memory_content_hash

        normed = [(m.get("id"), normalize_memory_line(m.get("content") or "")) for m in rows]
        normed = [(mid, content) for mid, content in normed if content]  # drop blanks

        included: List[str] = []
        seed: List = []
        used = 0
        for mid, content in normed:
            addition = len(content) + (1 if included else 0)  # +1 for the joining newline
            if used + addition > self._MEMORY_TEXTAREA_MAX:
                break
            included.append(content)
            seed.append([mid, memory_content_hash(content)])
            used += addition

        return "\n".join(included), seed, len(normed) - len(included)

    def _build_channel_memory_blocks(self, textarea_value: str, hidden_count: int,
                                     workspace_rows: List[Dict]) -> List[Dict]:
        """Blocks for the channel-memory editor plus the read-only workspace-shared list.

        Channel-scope memory is ONE multiline textarea (`block_id="channel_memory_block"`,
        `action_id="channel_memory"`): edit or delete lines and Save to reconcile, blank it out to
        forget everything. Workspace-scope facts are visible context but READ-ONLY from a channel
        (see `message_processor/memory_tools.py` `_visible_row`), so they render without any control.
        """
        memory_input: Dict = {
            "type": "plain_text_input", "action_id": "channel_memory",
            "multiline": True, "max_length": self._MEMORY_TEXTAREA_MAX,
            "placeholder": {"type": "plain_text",
                            "text": "e.g. Deploys go out Thursdays — ping @oncall before merging."},
        }
        # Slack rejects an empty initial_value, so only set it when there's something to seed.
        if textarea_value:
            memory_input["initial_value"] = textarea_value

        blocks: List[Dict] = [
            {"type": "section",
             "text": {"type": "mrkdwn", "text": "*What I remember about this channel*"}},
            {"type": "input", "block_id": "channel_memory_block", "optional": True,
             "element": memory_input,
             "label": {"type": "plain_text", "text": "Channel memory"},
             "hint": {"type": "plain_text",
                      "text": "One note per line; blank it out to forget everything."}},
        ]
        if hidden_count > 0:
            blocks.append({"type": "context", "elements": [
                {"type": "mrkdwn", "text": f"_+{hidden_count} more not shown_"}]})

        # Workspace-scope memories: shown for context, but managed elsewhere — no edit control.
        if workspace_rows:
            cap = self._MODAL_LIST_CAP
            shown = workspace_rows[:cap]
            lines = "\n".join(f"• {self._truncate(m.get('content') or '', 200) or '(empty)'}"
                              for m in shown)
            if len(workspace_rows) > cap:
                lines += f"\n_+{len(workspace_rows) - cap} more_"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"*Workspace-shared memories* (read-only here)\n{lines}"},
            })

        return blocks

    # The personal-memory box's Slack ids, named once so the builder and the submit handler
    # cannot drift about what to read out of `view['state']`.
    USER_MEMORY_BLOCK = "user_memory_block"
    USER_MEMORY_ACTION = "user_memory"
    USER_MEMORY_FORGET_BLOCK = "user_memory_forget_block"
    USER_MEMORY_FORGET_ACTION = "user_memory_forget_all"
    USER_MEMORY_FORGET_VALUE = "forget_all"

    def _build_user_memory_blocks(self, textarea_value: str, hidden_count: int,
                                  forget_all: bool = False) -> List[Dict]:
        """Blocks for the personal-memory editor in the USER settings modal.

        The channel modal's bargain, for one person's own store: one multiline textarea, one note
        per line, edit or delete lines and Save to reconcile against the open-time seed.

        THE CHECKBOX APPEARS ONLY WHEN THE BOX COULD NOT SHOW EVERYTHING. Blanking the box
        already forgets everything that was shown, so with nothing hidden the checkbox would say
        twice what the empty textarea already says. Past `_MEMORY_TEXTAREA_MAX` that stops being
        true: the reconciler deletes only rows it seeded, so blanking a truncated box silently
        keeps the rest — a delete affordance that quietly under-delivers on "forget everything".
        That is the case the checkbox is for, wired to a full-store delete. `list_facts` is where
        anything past the budget can still be read in full.
        """
        memory_input: Dict[str, Any] = {
            "type": "plain_text_input", "action_id": self.USER_MEMORY_ACTION,
            "multiline": True, "max_length": self._MEMORY_TEXTAREA_MAX,
            "placeholder": {
                "type": "plain_text",
                "text": ("e.g. Prefers short answers with the code first.\n\n"
                         "One note per line. Private to your DMs — never shown in channels."),
            },
        }
        # Slack rejects an empty initial_value, so only set it when there's something to seed.
        if textarea_value:
            memory_input["initial_value"] = textarea_value

        blocks: List[Dict[str, Any]] = [
            {"type": "section",
             "text": {"type": "mrkdwn", "text": "*What I remember about you*"}},
            {"type": "input", "block_id": self.USER_MEMORY_BLOCK, "optional": True,
             "element": memory_input,
             "label": {"type": "plain_text", "text": "Personal memory"}},
        ]
        if hidden_count > 0:
            blocks.append({"type": "context", "elements": [
                {"type": "mrkdwn", "text": f"_+{hidden_count} more not shown_"}]})

            forget_option = {
                "text": {"type": "plain_text",
                         "text": "Forget everything, including items not shown"},
                "value": self.USER_MEMORY_FORGET_VALUE,
            }
            forget_element: Dict[str, Any] = {
                "type": "checkboxes", "action_id": self.USER_MEMORY_FORGET_ACTION,
                "options": [forget_option],
            }
            if forget_all:
                forget_element["initial_options"] = [forget_option]
            blocks.append({
                "type": "input", "block_id": self.USER_MEMORY_FORGET_BLOCK, "optional": True,
                "element": forget_element,
                "label": {"type": "plain_text", "text": "Clear personal memory"},
                "hint": {"type": "plain_text",
                         "text": "Deletes every note above and any beyond what fits here. No undo."},
            })
        blocks.append({"type": "divider"})
        return blocks

    def _build_modal_blocks(self, settings: Dict, selected_model: str,
                           is_new_user: bool = False, in_thread: bool = False,
                           scope: Optional[str] = None,
                           user_memory: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Build the modal blocks based on current settings and model selection

        Args:
            settings: Current settings dictionary
            selected_model: Currently selected model
            is_new_user: Whether this is a new user
            in_thread: Whether modal was opened from within a thread
            scope: The selected scope ('thread' or 'global')
            user_memory: Rendered personal-memory state (`value`/`hidden`/`forget_all`), or None
                to omit the section entirely — which is what ENABLE_USER_MEMORY off looks like
        """
        blocks: List[Dict[str, Any]] = []
        
        # Determine default scope if not provided
        if scope is None:
            # New users should always default to global settings
            if is_new_user:
                scope = 'global'
            else:
                scope = 'thread' if in_thread else 'global'
        
        # Welcome message for new users
        if is_new_user:
            blocks.extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Welcome to the AI Assistant!* 👋\nLet's configure your settings. You can accept the defaults or customize them."
                    }
                },
                {"type": "divider"}
            ])
        
        # Add scope selector for existing users only (new users must configure global first)
        scope_options = []
        
        # New users must configure global settings first
        if is_new_user:
            # For new users, don't show scope selector - they must configure global first
            if in_thread:
                blocks.append({
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "📌 _Setting up your global preferences (applies to all conversations). You can customize thread-specific settings later._"}]
                })
                blocks.append({"type": "divider"})
        else:
            # Add thread option if in a thread
            if in_thread:
                scope_options.append({
                    "text": {"type": "plain_text", "text": "💬 This Thread Only"},
                    "value": "thread",
                    "description": {"type": "plain_text", "text": "Settings apply only to this conversation"}
                })
            
            # Always add global option
            scope_options.append({
                "text": {"type": "plain_text", "text": "🌐 Global Settings"},
                "value": "global",
                "description": {"type": "plain_text", "text": "Settings apply to all conversations"}
            })
        
        # Only add scope selector if there are multiple options
        if len(scope_options) > 1:
            self.log_debug(f"Building scope selector - in_thread: {in_thread}, scope: {scope}, options: {[o['value'] for o in scope_options]}")
            
            # Find the matching option for initial selection
            initial_option = None
            for option in scope_options:
                if option['value'] == scope:
                    initial_option = option
                    break
            
            # If no match found, default to first option
            if not initial_option:
                self.log_warning(f"Scope '{scope}' not found in options, defaulting to first option")
                initial_option = scope_options[0]
            
            self.log_debug(f"Selected initial_option value: {initial_option['value']}")
            
            blocks.append({
                "type": "section",
                "block_id": "scope_selector",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Settings Scope*\nChoose where to save these settings:"
                },
                "accessory": {
                    "type": "radio_buttons",
                    "action_id": "settings_scope",
                    "options": scope_options,
                    "initial_option": initial_option
                }
            })
            
            blocks.append({"type": "divider"})
            
            # Add tip about accessing settings
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"💡 *Tip:* For global settings, type `{config.settings_slash_command}` in any channel/DM (not in a thread)"
                }]
            })
        elif not is_new_user:
            # Single scope available - show header indicating which one
            header_text = "Configure Your Global Settings"
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": header_text}
            })
            
            # Add tip about accessing settings when only global is available
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"💡 *Tip:* You can change these settings anytime by typing:\n`{config.settings_slash_command}` in any channel/DM (not in a thread)"
                }]
            })
        # Model selection (always shown)
        from config import SUPPORTED_CHAT_MODELS
        blocks.append({
            "type": "section",
            "block_id": "model_block",
            "text": {
                "type": "mrkdwn",
                "text": "*AI Model*\nChoose your preferred AI model"
            },
            # Radio buttons render inline (no floating dropdown overlay, which Slack
            # clips at the modal edge) and scale fine for a short model list.
            "accessory": {
                "type": "radio_buttons",
                "action_id": "model_select",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_model_display_name(selected_model)},
                    "value": selected_model
                },
                # Built from the same constant `selected_model` is coerced against. Hard-coding
                # this list is how a newly supported model becomes a legal `initial_option` that
                # is absent from `options` — Slack rejects the entire view for that.
                "options": [
                    {"text": {"type": "plain_text", "text": self._get_model_display_name(m)},
                     "value": m}
                    for m in SUPPORTED_CHAT_MODELS
                ]
            }
        })

        # Fast service tier — personal scope only, never written into a thread config.
        blocks.extend(self._fast_tier_blocks(settings, selected_model, scope))

        blocks.append({"type": "divider"})

        # Model-specific settings (reasoning ladder differs: 5.6 family adds `max`)
        blocks.extend(self._add_gpt55_settings(settings, selected_model))
        
        # Add common settings (features and image settings)
        blocks.extend(self._add_common_settings(settings, user_memory))

        return blocks
    
    def _fast_tier_blocks(self, settings: Dict, selected_model: str,
                          scope: Optional[str]) -> List[Dict]:
        """The fast-service-tier control, or the context line that stands in for it.

        Slack has NO disabled form control, so "greyed out" is rendered as *no control at all* —
        a context block saying why. A checkbox that visibly rejects the click is worse, and Slack
        fights it.

        Rendered ONLY in the global/personal scope. `service_tier` is a personal setting: a
        channel is not one person, and thread settings are whole-document replacements
        (`save_thread_config_async` replaces `config_json` outright), so a hidden checkbox in the
        thread modal would silently delete a stored opt-in.
        """
        if scope != 'global':
            return []

        from config import FAST_SERVICE_TIER_MODELS

        def _context(text: str) -> Dict[str, Any]:
            return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}

        # The admin gate. `standard` means no user may turn fast on, whatever their model.
        if config.openai_service_tier != 'fast':
            return [_context("⚡ *Fast responses* — Disabled by your system administrator.")]

        if selected_model not in FAST_SERVICE_TIER_MODELS:
            name = self._get_model_display_name(selected_model)
            return [_context(f"⚡ *Fast responses* — not available on {name}.")]

        option = {
            "text": {"type": "mrkdwn", "text": "⚡ *Fast responses*\nUp to 2x faster."},
            "value": "fast"
        }
        accessory: Dict[str, Any] = {
            "type": "checkboxes",
            "action_id": "service_tier",
            "options": [option]
        }
        # Slack wants the array omitted entirely rather than empty, and the initial option must be
        # the same object that appears in `options`.
        if settings.get('service_tier') == 'fast':
            accessory["initial_options"] = [option]

        return [{
            "type": "section",
            "block_id": "service_tier_block",
            "text": {"type": "mrkdwn", "text": "Response speed:"},
            "accessory": accessory
        }]

    def _add_gpt55_settings(self, settings: Dict, selected_model: str = 'gpt-5.6-sol') -> List[Dict]:
        """Add model-specific settings blocks (reasoning ladder, temp/top_p when reasoning=none).

        The 5.6 family offers the full ladder incl. `max` (verified live on all three
        tiers); gpt-5.5 tops out at `xhigh`."""
        blocks: List[Dict[str, Any]] = []

        # Check if web search is enabled
        if 'enable_web_search' in settings:
            web_search_enabled = bool(settings['enable_web_search'])
        else:
            web_search_enabled = True
        self.log_debug(f"Settings passed to _add_gpt55_settings: enable_web_search={settings.get('enable_web_search')}, evaluated as {web_search_enabled}")

        from config import config, clamp_effort, effort_ladder
        current_reasoning = settings.get('reasoning_effort', 'none')
        self.log_debug(f"Building reasoning options for {selected_model}, current: {current_reasoning}")

        # ONE definition of the ladder, shared with the resolver. This used to read the raw
        # constants and treat anything that was not 5.6 as 5.5, which offered GPT-6 an effort of
        # `none` — a value that 400s. Advertising a choice the submit-time clamp then silently
        # overwrites is the exact disagreement `effort_ladder` exists to end.
        effort_values = effort_ladder(selected_model)
        reasoning_options = [
            {"text": {"type": "plain_text", "text": self._get_reasoning_display(v)}, "value": v}
            for v in effort_values
        ]

        available_values = [opt['value'] for opt in reasoning_options]
        self.log_debug(f"Reasoning options available for {selected_model}: {available_values}, initial: {current_reasoning}")

        # Clamp stale stored values per model rules (e.g. legacy `minimal`, or `max`
        # carried over after switching 5.6 -> 5.5) instead of blindly resetting
        if current_reasoning not in available_values:
            old_reasoning = current_reasoning
            current_reasoning = clamp_effort(selected_model, current_reasoning)
            if current_reasoning not in available_values:
                current_reasoning = clamp_effort(selected_model, config.default_reasoning_effort)
            self.log_warning(f"Current reasoning '{old_reasoning}' not in available options, clamped to '{current_reasoning}'")

        # Build the reasoning block
        reasoning_block: Dict[str, Any] = {
            "type": "section",
            "block_id": "reasoning_block_gpt54",
            "text": {
                "type": "mrkdwn",
                "text": "*Reasoning Level*\nControls depth of analysis and problem-solving"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "reasoning_level_gpt54",
                "options": reasoning_options
            }
        }

        # Add initial_option if we have a valid selection
        if current_reasoning and current_reasoning != 'None' and current_reasoning in available_values:
            reasoning_block["accessory"]["initial_option"] = {
                "text": {"type": "plain_text", "text": self._get_reasoning_display(current_reasoning)},
                "value": current_reasoning
            }
            self.log_debug(f"Set initial_option for reasoning: {current_reasoning}")
        else:
            if available_values:
                # This branch runs when Slack fails to report a selection. `'none'` is not a legal
                # GPT-6 value, so the workspace default (clamped to the model) stands in instead.
                default_value = clamp_effort(selected_model, config.default_reasoning_effort)
                reasoning_block["accessory"]["initial_option"] = {
                    "text": {"type": "plain_text", "text": self._get_reasoning_display(default_value)},
                    "value": default_value
                }
                self.log_debug(f"No valid reasoning selection - set default initial_option: {default_value}")

        blocks.append(reasoning_block)

        # Add note about xhigh reasoning and temperature availability
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "_Extra High reasoning provides maximum accuracy but is slower and more expensive. Temperature/Top P controls are available when reasoning is set to None._"
            }]
        })

        # Response Detail
        blocks.append({
            "type": "section",
            "block_id": "verbosity_block",
            "text": {
                "type": "mrkdwn",
                "text": "*Response Detail*\nControls how detailed responses are"
            },
            "accessory": {
                "type": "radio_buttons",
                "action_id": "verbosity",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_verbosity_display(settings.get('verbosity', config.default_verbosity))},
                    "value": settings.get('verbosity', config.default_verbosity)
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "📝 Concise"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "📄 Standard"}, "value": "medium"},
                    {"text": {"type": "plain_text", "text": "📚 Detailed"}, "value": "high"}
                ]
            }
        })

        # Temperature and Top P - only visible when reasoning=none
        if current_reasoning == 'none':
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": "⚠️ *Important:* Modify either Temperature OR Top P, not both. OpenAI recommends changing only one."
                }]
            })

            blocks.append({
                "type": "input",
                "block_id": "temperature_block",
                "element": {
                    "type": "number_input",
                    "action_id": "temperature",
                    "is_decimal_allowed": True,
                    "min_value": "0.0",
                    "max_value": "2.0",
                    "initial_value": str(settings.get('temperature', 1.0))
                },
                "label": {"type": "plain_text", "text": "Temperature (0.0-2.0)"},
                "hint": {"type": "plain_text", "text": "Controls randomness. Use this OR Top P, not both. Default: 1.0"}
            })

            blocks.append({
                "type": "input",
                "block_id": "top_p_block",
                "element": {
                    "type": "number_input",
                    "action_id": "top_p",
                    "is_decimal_allowed": True,
                    "min_value": "0.0",
                    "max_value": "1.0",
                    "initial_value": str(settings.get('top_p', 1.0))
                },
                "label": {"type": "plain_text", "text": "Top P (0.0-1.0)"},
                "hint": {"type": "plain_text", "text": "Alternative to temperature. Keep at 1.0 if using temperature. Default: 1.0"}
            })

        blocks.append({"type": "divider"})
        return blocks

    @staticmethod
    def image_size_for(shape: str, tier: str) -> str:
        """Resolve a (shape, tier) pair to what gets stored in `image_size`.

        `auto` stores the literal "auto" — the tier is ignored, exactly as it is ignored at
        render time. A shape that is already a WxH is the synthetic "Custom: …" option coming
        back unchanged, and passes straight through.
        """
        if shape == "auto":
            return "auto"
        row = _SHAPE_TIER_SIZES.get(shape)
        if row is None:
            return shape if _WXH_RE.match(shape) else "auto"
        return row.get(tier) or row["standard"]

    @staticmethod
    def shape_tier_for(size: str, shapes: Sequence[str], tiers: Sequence[str],
                       tier: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Reverse-map a stored `image_size` onto the controls CURRENTLY rendered.

        None means the stored value is not producible by those controls — an off-grid legacy
        size, a size the model chose on an earlier turn, or a grid size carried onto
        gpt-image-1 — and the caller injects the synthetic option for it.

        Under Auto the shape is the model's to pick but the TIER is still the person's, so the
        saved `image_tier` is what comes back rather than a hard-coded "standard" — that is the
        bug this parameter exists to fix.

        A legacy `max` — saved before the 4K tier was pulled — maps to `large` BEFORE any
        default is consulted, matching `image_service.user_defaults`. The two disagreeing is a
        real bug, not cosmetics: under `DEFAULT_IMAGE_TIER=standard` the modal showed Standard
        while the renderer used Large, and saving the view wrote the downgrade back. Any other
        tier the rendered controls cannot offer falls back to the configured default (itself put
        through the same `max` mapping) and then to what the controls do offer — on gpt-image-1
        only `standard` is rendered, so that wins. Returning a tier that is not in `tiers` would
        put an initial_option outside its own option list, which Slack rejects for the whole view.
        """
        if size == "auto":
            default_tier = getattr(config, "default_image_tier", "large")
            if default_tier == "max":
                default_tier = "large"
            if tier == "max" and "large" in tiers:
                tier = "large"
            if tier not in tiers:
                tier = default_tier if default_tier in tiers else (
                    tiers[0] if tiers else "standard")
            return ("auto", str(tier))
        for shape in shapes:
            row = _SHAPE_TIER_SIZES.get(shape) or {}
            for tier in tiers:
                if row.get(tier) == size:
                    return (shape, tier)
        return None

    @staticmethod
    def _resolution_line(size: str, tier: Optional[str] = None) -> str:
        """The context line under the size controls: the pixel size and, when it has a name, the
        aspect ratio. No pixel counts — the owner ruled them noise (2026-09-09).

        Under Auto there is no single resolution to state, but the tier still applies to whatever
        shape the model picks, so the line names the tier and shows the 16:9 cell as the example.
        `tier` is None on models that render no tier select at all (gpt-image-1).
        """
        if (size or "") == "auto" and tier:
            label = dict(_IMAGE_TIERS).get(tier, tier).split(" —")[0]
            example = (_SHAPE_TIER_SIZES.get("16:9") or {}).get(tier)
            if example:
                ex_w, ex_h = example.split("x")
                return (f"_Resolution: chosen per image at {label} "
                        f"(e.g. {ex_w} × {ex_h} for 16:9)_")
            return f"_Resolution: chosen per image at {label}_"
        match = _WXH_RE.match(size or "")
        if not match:
            return "_Resolution: chosen by the model_"
        w, h = int(match.group(1)), int(match.group(2))
        ratio = _aspect_label(w, h)
        return f"_Resolution: {w} × {h} · {ratio}_" if ratio else f"_Resolution: {w} × {h}_"

    def _image_size_blocks(self, settings: Dict, supports_custom_sizes: bool) -> List[Dict]:
        """Shape (+ size tier, where the model takes custom sizes) and the resolution readout.

        Storage is unchanged: `image_size` still holds a resolved WxH or "auto". These two
        controls are only how a person picks one without being shown pixel arithmetic.
        """
        shape_keys = [s for s, _ in _IMAGE_SHAPES] if supports_custom_sizes else list(_V1_SHAPES)
        tier_keys = [t for t, _ in _IMAGE_TIERS] if supports_custom_sizes else ["standard"]

        stored_size = str(settings.get('image_size') or 'auto')
        stored_tier = settings.get('image_tier')
        match = self.shape_tier_for(stored_size, shape_keys, tier_keys,
                                    str(stored_tier) if stored_tier else None)

        # SYNTHETIC OPTION. Slack rejects the whole view when `initial_option` is not also in
        # `options`, so the same dict OBJECT is used in both places rather than two equal copies.
        # The tier the select opens on when the stored size does not name one itself: the saved
        # `image_tier`, coerced to what these controls offer. It must not silently become
        # "standard", because the tier select is extracted on submit whatever the shape is, and
        # a person with a custom size would lose their saved tier just by opening the modal.
        _, fallback_tier = cast(Tuple[str, str],
                                self.shape_tier_for("auto", shape_keys, tier_keys,
                                                    str(stored_tier) if stored_tier else None))

        synthetic: Optional[Dict[str, Any]] = None
        if match is not None:
            shape, tier = match
        elif _WXH_RE.match(stored_size):
            w, h = stored_size.split('x')
            ratio = _aspect_label(int(w), int(h))
            label = f"Custom: {w} × {h}" + (f" ({ratio})" if ratio else "")
            synthetic = {"text": {"type": "plain_text", "text": label}, "value": stored_size}
            shape, tier = stored_size, fallback_tier
        else:
            # Not a size at all (a value from a schema that never existed). Fall back to auto.
            shape, tier = "auto", fallback_tier

        shape_options: List[Dict[str, Any]] = [
            {"text": {"type": "plain_text", "text": label}, "value": key}
            for key, label in _IMAGE_SHAPES if key in shape_keys
        ]
        if synthetic is not None:
            shape_options.insert(0, synthetic)
            initial_shape = synthetic
        else:
            initial_shape = next(o for o in shape_options if o["value"] == shape)

        blocks: List[Dict[str, Any]] = [{
            "type": "section",
            "block_id": "image_ratio_block",
            "text": {"type": "mrkdwn", "text": "Image shape:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_ratio",
                "placeholder": {"type": "plain_text", "text": "Select shape"},
                "initial_option": initial_shape,
                "options": shape_options
            }
        }]

        if supports_custom_sizes:
            tier_options: List[Dict[str, Any]] = [
                {"text": {"type": "plain_text", "text": label}, "value": key}
                for key, label in _IMAGE_TIERS
            ]
            initial_tier = next(o for o in tier_options if o["value"] == tier)
            blocks.append({
                "type": "section",
                "block_id": "image_tier_block",
                "text": {"type": "mrkdwn", "text": "Size:"},
                "accessory": {
                    "type": "static_select",
                    "action_id": "image_tier",
                    "placeholder": {"type": "plain_text", "text": "Select size"},
                    "initial_option": initial_tier,
                    "options": tier_options
                }
            })

        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": self._resolution_line(
                              self.image_size_for(shape, tier),
                              tier if supports_custom_sizes else None)}]
        })
        return blocks

    def _add_common_settings(self, settings: Dict,
                            user_memory: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Add settings common to all models"""
        blocks: List[Dict[str, Any]] = []
        
        # Custom Instructions section (available for all models)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Custom Instructions*"}
        })
        
        # Ensure initial_value is always a string
        custom_instructions_value = settings.get('custom_instructions', '')
        if custom_instructions_value is None:
            custom_instructions_value = ''
        
        blocks.append({
            "type": "input",
            "block_id": "custom_instructions_block",
            "element": {
                "type": "plain_text_input",
                "action_id": "custom_instructions",
                "multiline": True,
                "initial_value": custom_instructions_value,
                "placeholder": {
                    "type": "plain_text",
                    "text": "- Be concise and use bullet points\n- Explain technical topics simply\n- Include code examples\n- Use professional tone"
                },
                "max_length": 3000
            },
            "label": {
                "type": "plain_text",
                "text": "How would you like the AI to respond? (Custom GPT Instructions)"
            },
            "optional": True
        })

        blocks.append({"type": "divider"})

        # Personal memory sits directly under Custom Instructions on purpose: both are "what the
        # bot knows about me", one written by hand and one written by the bot, and separating them
        # would leave a person hunting for where the facts they were told about live.
        if user_memory is not None:
            blocks.extend(self._build_user_memory_blocks(
                user_memory.get("value") or "",
                int(user_memory.get("hidden") or 0),
                bool(user_memory.get("forget_all")),
            ))

        # Feature toggles
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Features*"}
        })

        # All supported models (gpt-5.5) support MCP; the old GPT-4-era
        # hide-MCP branch is gone with the pre-5.5 lineup.
        current_enable_mcp = settings.get('enable_mcp', True)

        # Build checkbox options for features
        feature_options = []
        initial_options = []

        # Web search
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🌐 *Web Search*\nAllow searching the web for current information"},
            "value": "web_search"
        })
        if settings.get('enable_web_search', True):
            initial_options.append(feature_options[-1])

        # Streaming
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🌊 *Streaming*\nShow responses as they're generated"},
            "value": "streaming"
        })
        if settings.get('enable_streaming', True):
            initial_options.append(feature_options[-1])

        # MCP Servers
        feature_options.append({
            "text": {"type": "mrkdwn", "text": "🔌 *MCP Servers*\nAccess specialized data sources"},
            "value": "mcp"
        })
        if current_enable_mcp:
            initial_options.append(feature_options[-1])

        # Build the features block (ids kept stable for the registered action handler)
        block_id = "features_block_gpt5"
        action_id = "features_with_mcp"

        # An `actions` block, not a `section` + accessory: the *Features* header above already
        # names this group, and a section must carry text, so the redundant "Enable features:"
        # label is what forced the section shape. Ids stay byte-identical either way.
        checkboxes: Dict[str, Any] = {
            "type": "checkboxes",
            "action_id": action_id,
            "options": feature_options
        }

        # Only add initial_options if we have some (Slack requires array or omitted entirely)
        if initial_options:
            checkboxes["initial_options"] = initial_options

        features_block: Dict[str, Any] = {
            "type": "actions",
            "block_id": block_id,
            "elements": [checkboxes]
        }

        blocks.append(features_block)

        blocks.append({"type": "divider"})

        # Image Generation Settings
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Image Generation*"}
        })

        # Image model — coerce a stale stored/config value (e.g. a retired gpt-image-1-mini)
        # to a live option so views.open doesn't reject the block. Also feeds is_image_v2 below.
        # The allowlist is the module constant the channel resolver validates against, so the
        # personal picker and the channel profile can never disagree about what exists.
        from config import SUPPORTED_IMAGE_MODELS
        image_model_choices = set(SUPPORTED_IMAGE_MODELS)
        selected_image_model = self._coerce_choice(
            settings.get('image_model', config.image_model), image_model_choices,
            config.image_model if config.image_model in image_model_choices else 'gpt-image-2')
        blocks.append({
            "type": "section",
            "block_id": "image_model_block",
            "text": {"type": "mrkdwn", "text": "Image model:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_model",
                "placeholder": {"type": "plain_text", "text": "Select image model"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_model_display_name(selected_image_model)},
                    "value": selected_image_model
                },
                # Built from the same constant the coercion above validates against. A hard-coded
                # list would let a newly supported model become a legal `initial_option` that is
                # absent from `options`, and Slack rejects the whole view for that.
                "options": [
                    {"text": {"type": "plain_text",
                              "text": self._get_image_model_display_name(m)}, "value": m}
                    for m in SUPPORTED_IMAGE_MODELS
                ]
            }
        })

        # Image shape and size tier. The pair resolves to the single stored `image_size` (a WxH
        # or "auto") on submit and is reverse-mapped from it here — no schema change.
        from message_processor.image_service import (backgrounds_for, qualities_for,
                                                     supports_custom_sizes,
                                                     supports_input_fidelity)
        custom_sizes = supports_custom_sizes(selected_image_model)
        blocks.extend(self._image_size_blocks(settings, custom_sizes))

        # Image quality — the legal set is per-model (the 2.5 family adds `xhigh` and `max`).
        legal_qualities = qualities_for(selected_image_model)
        selected_image_quality = self._coerce_choice(
            settings.get('image_quality', 'auto'), set(legal_qualities), 'auto')
        blocks.append({
            "type": "section",
            "block_id": "image_quality_block",
            "text": {"type": "mrkdwn", "text": "Image quality:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_quality",
                "placeholder": {"type": "plain_text", "text": "Select quality"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_quality_display(selected_image_quality)},
                    "value": selected_image_quality
                },
                "options": [
                    {"text": {"type": "plain_text", "text": self._get_image_quality_display(q)},
                     "value": q}
                    for q in legal_qualities
                ]
            }
        })

        # Image background — read from the capability table, which says every model takes all
        # three (transparent + png returns 200 on gpt-image-2; the belief that it did not was
        # wrong, and it cost users the option on what was then the default model).
        legal_backgrounds = backgrounds_for(selected_image_model)
        background_options = [
            {"text": {"type": "plain_text", "text": self._get_image_background_display(b)},
             "value": b}
            for b in legal_backgrounds
        ]

        # Coerce the saved value against the visible options, so initial_option always matches.
        saved_background = self._coerce_choice(
            settings.get('image_background', 'auto'),
            {opt['value'] for opt in background_options}, 'auto')

        blocks.append({
            "type": "section",
            "block_id": "image_background_block",
            "text": {"type": "mrkdwn", "text": "Image background:"},
            "accessory": {
                "type": "static_select",
                "action_id": "image_background",
                "placeholder": {"type": "plain_text", "text": "Select background"},
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_image_background_display(saved_background)},
                    "value": saved_background
                },
                "options": background_options
            }
        })

        # Input fidelity for edits — only gpt-image-1 takes the parameter; the 2.x models 400 on
        # it, so they get no control (they handle fidelity themselves).
        if supports_input_fidelity(selected_image_model):
            selected_input_fidelity = self._coerce_choice(
                settings.get('input_fidelity', 'high'), {'high', 'low'}, 'high')
            blocks.append({
                "type": "section",
                "block_id": "input_fidelity_block",
                "text": {"type": "mrkdwn", "text": "Image edit style:"},
                "accessory": {
                    "type": "radio_buttons",
                    "action_id": "input_fidelity",
                    "initial_option": {
                        "text": {"type": "plain_text", "text": self._get_fidelity_display(selected_input_fidelity)},
                        "value": selected_input_fidelity
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "🎨 Preserve Original Style"}, "value": "high"},
                        {"text": {"type": "plain_text", "text": "✨ Allow Reinterpretation"}, "value": "low"}
                    ]
                }
            })
        
        blocks.append({"type": "divider"})

        # Vision detail level
        selected_vision_detail = self._coerce_choice(
            settings.get('vision_detail', 'auto'), {'auto', 'low', 'high'}, 'auto')
        blocks.append({
            "type": "section",
            "block_id": "vision_detail_block",
            "text": {"type": "mrkdwn", "text": "Vision analysis detail:"},
            "accessory": {
                "type": "radio_buttons",
                "action_id": "vision_detail",
                "initial_option": {
                    "text": {"type": "plain_text", "text": self._get_vision_detail_display(selected_vision_detail)},
                    "value": selected_vision_detail
                },
                "options": [
                    {"text": {"type": "plain_text", "text": "🤖 Auto"}, "value": "auto"},
                    {"text": {"type": "plain_text", "text": "🔍 Low Detail"}, "value": "low"},
                    {"text": {"type": "plain_text", "text": "🔬 High Detail"}, "value": "high"}
                ]
            }
        })
        
        return blocks
    
    def extract_user_memory(self, view_state: Dict) -> Optional[Dict[str, Any]]:
        """The personal-memory box as submitted, or None when the section was not in the view.

        None is the "nothing to reconcile" answer and is deliberately distinct from an empty
        textarea: a modal opened before this section existed, or with ENABLE_USER_MEMORY off, must
        not be read as the user asking to forget everything. Returns
        ``{'raw': str, 'lines': [normalized], 'forget_all': bool}``; `lines` are normalized,
        blank-dropped and deduped in submitted order, exactly as the channel path does it.
        """
        from database import normalize_memory_line

        values = (view_state or {}).get('values', {})
        block = values.get(self.USER_MEMORY_BLOCK)
        forget_block = values.get(self.USER_MEMORY_FORGET_BLOCK)
        if block is None and forget_block is None:
            return None

        raw = ((block or {}).get(self.USER_MEMORY_ACTION, {}) or {}).get('value') or ""
        lines: List[str] = []
        seen: set = set()
        for ln in raw.split("\n"):
            norm = normalize_memory_line(ln)
            if norm and norm not in seen:
                seen.add(norm)
                lines.append(norm)

        selected = ((forget_block or {}).get(self.USER_MEMORY_FORGET_ACTION, {}) or {}
                    ).get('selected_options') or []
        forget_all = any(o.get('value') == self.USER_MEMORY_FORGET_VALUE for o in selected)
        return {"raw": raw, "lines": lines, "forget_all": forget_all}

    def extract_form_values(self, view_state: Dict) -> Dict:
        """Extract form values from modal submission"""
        values = view_state.get('values', {})
        extracted = {}
        
        # Model selection
        model_block = values.get('model_block', {})
        if 'model_select' in model_block:
            selected = model_block['model_select'].get('selected_option')
            if selected:
                extracted['model'] = selected['value']
        
        # Reasoning effort (block/action ids kept stable across model families)
        reasoning_block = values.get('reasoning_block_gpt54', {})
        reasoning_found = False
        if 'reasoning_level_gpt54' in reasoning_block:
            selected = reasoning_block['reasoning_level_gpt54'].get('selected_option')
            if selected:
                extracted['reasoning_effort'] = selected['value']
                reasoning_found = True
            else:
                # No selection - might happen during modal updates
                self.log_debug("No reasoning_level_gpt54 selected_option found")

        # Fallback if no reasoning selection due to Slack modal update bug. This forced 'none'
        # for years, which silently overwrote a stored effort with the weakest setting the
        # moment Slack failed to report the selection — and it is the ONLY seam that runs, so a
        # configured default further downstream never got a say. The workspace default is the
        # honest stand-in for "we could not read what they picked".
        if not reasoning_found:
            from config import clamp_effort
            model_for_clamp = extracted.get('model') or config.gpt_model
            extracted['reasoning_effort'] = clamp_effort(model_for_clamp,
                                                         config.default_reasoning_effort)
            self.log_debug("No reasoning selection found - using workspace default: "
                           f"{extracted['reasoning_effort']}")
        
        verbosity_block = values.get('verbosity_block', {})
        if 'verbosity' in verbosity_block:
            selected = verbosity_block['verbosity'].get('selected_option')
            if selected:
                extracted['verbosity'] = selected['value']
        
        # Temperature / Top P (only present in the form when reasoning=none)
        temp_block = values.get('temperature_block', {})
        if 'temperature' in temp_block:
            extracted['temperature'] = float(temp_block['temperature'].get('value', 0.8))
        
        top_p_block = values.get('top_p_block', {})
        if 'top_p' in top_p_block:
            extracted['top_p'] = float(top_p_block['top_p'].get('value', 1.0))
        
        # Features
        features_block = values.get('features_block_gpt5', {})
        if 'features_with_mcp' in features_block:
            selected_options = features_block['features_with_mcp'].get('selected_options', [])
            selected_values = [opt['value'] for opt in selected_options]
            extracted['enable_web_search'] = 'web_search' in selected_values
            extracted['enable_streaming'] = 'streaming' in selected_values
            extracted['enable_mcp'] = 'mcp' in selected_values
        
        # Custom Instructions
        custom_instructions_block = values.get('custom_instructions_block', {})
        if 'custom_instructions' in custom_instructions_block:
            custom_value = custom_instructions_block['custom_instructions'].get('value')
            # Handle None (cleared field) or empty string
            if custom_value:
                custom_text = custom_value.strip()
                extracted['custom_instructions'] = custom_text if custom_text else None
            else:
                extracted['custom_instructions'] = None
        
        # Image settings. Shape and size tier are two controls that resolve to the ONE stored
        # `image_size` key; the tier block is absent on models without custom sizes, and
        # `standard` is then the only column that exists.
        ratio_block = values.get('image_ratio_block', {})
        tier_block = values.get('image_tier_block', {})
        tier_selected = (tier_block.get('image_tier') or {}).get('selected_option') or {}
        if 'image_ratio' in ratio_block:
            selected = ratio_block['image_ratio'].get('selected_option')
            if selected:
                extracted['image_size'] = self.image_size_for(
                    selected['value'], tier_selected.get('value') or 'standard')
        # The tier is stored in its OWN key as well, independently of the shape: under Auto the
        # shape resolves to the literal "auto" and the tier is the only thing left carrying the
        # person's size choice through to render time.
        if tier_selected.get('value'):
            extracted['image_tier'] = tier_selected['value']

        image_quality_block = values.get('image_quality_block', {})
        if 'image_quality' in image_quality_block:
            selected = image_quality_block['image_quality'].get('selected_option')
            if selected:
                extracted['image_quality'] = selected['value']

        image_background_block = values.get('image_background_block', {})
        if 'image_background' in image_background_block:
            selected = image_background_block['image_background'].get('selected_option')
            if selected:
                extracted['image_background'] = selected['value']

        image_model_block = values.get('image_model_block', {})
        if 'image_model' in image_model_block:
            selected = image_model_block['image_model'].get('selected_option')
            if selected:
                extracted['image_model'] = selected['value']

        fidelity_block = values.get('input_fidelity_block', {})
        if 'input_fidelity' in fidelity_block:
            selected = fidelity_block['input_fidelity'].get('selected_option')
            if selected:
                extracted['input_fidelity'] = selected['value']
        
        vision_block = values.get('vision_detail_block', {})
        if 'vision_detail' in vision_block:
            selected = vision_block['vision_detail'].get('selected_option')
            if selected:
                extracted['vision_detail'] = selected['value']

        # Fast service tier. The block exists only in the global scope and only when the admin
        # gate is on and the model is eligible; anywhere else its absence leaves the stored value
        # alone (global preference updates are partial).
        service_tier_block = values.get('service_tier_block', {})
        if 'service_tier' in service_tier_block:
            selected_options = service_tier_block['service_tier'].get('selected_options') or []
            extracted['service_tier'] = (
                'fast' if any(o.get('value') == 'fast' for o in selected_options) else 'standard')

        return extracted
    
    def validate_settings(self, settings: Dict) -> Dict:
        """Validate and adjust settings for compatibility"""
        validated = settings.copy()

        model = validated.get('model', config.gpt_model)

        # Clamp legacy/incompatible efforts per model (5.6 rejects `minimal`; 5.5 has no `max`)
        from config import clamp_effort
        if validated.get('reasoning_effort'):
            clamped = clamp_effort(model, validated['reasoning_effort'])
            if clamped != validated['reasoning_effort']:
                self.log_info(f"Clamped reasoning_effort {validated['reasoning_effort']} -> {clamped} for {model}")
                validated['reasoning_effort'] = clamped

        # Check if both temperature and top_p are changed for models that support them
        # (gpt-5.5, the 5.6 family and GPT-6 Sol/Luna support temp/top_p only with
        # reasoning=none; Astra never does)
        from config import supports_sampling
        supports_temp = supports_sampling(model, validated.get('reasoning_effort') or '')

        if supports_temp:
            default_temp = config.default_temperature
            default_top_p = config.default_top_p

            temp_changed = validated.get('temperature', default_temp) != default_temp
            top_p_changed = validated.get('top_p', default_top_p) != default_top_p

            if temp_changed and top_p_changed:
                self.log_warning(f"Both temperature ({validated.get('temperature')}) and top_p ({validated.get('top_p')}) "
                               f"were changed from defaults. OpenAI recommends using only one.")

        # Remove invalid parameters: reasoning models only take temp/top_p with reasoning=none
        if not supports_temp:
            validated.pop('temperature', None)
            validated.pop('top_p', None)

        # Same treatment for the fast tier: only some models honour it, and a value saved against
        # an ineligible model would buy nothing while looking like it had been granted.
        from config import FAST_SERVICE_TIER_MODELS
        if model not in FAST_SERVICE_TIER_MODELS:
            validated.pop('service_tier', None)

        return validated
    
    # Helper methods for display names
    def _get_model_display_name(self, model: str) -> str:
        """Get user-friendly model name"""
        display_names = {
            # Taglines follow OpenAI's own model classification (developers.openai.com/api/docs/models
            # and learn.chatgpt.com/docs/models, read 2026-09-09), shortened to fit a picker row.
            'gpt-6-astra': 'GPT-6 Astra (Most capable)',
            'gpt-6-sol': 'GPT-6 Sol (Everyday professional work)',
            'gpt-6-luna': 'GPT-6 Luna (Fast and affordable)',
            'gpt-5.6-sol': 'GPT-5.6 Sol (Complex professional work)',
            'gpt-5.6-terra': 'GPT-5.6 Terra (Balanced, everyday work)',
            'gpt-5.6-luna': 'GPT-5.6 Luna (Fast and affordable)',
            'gpt-5.5': 'GPT-5.5 (Previous generation)',
        }
        return display_names.get(model, model)
    
    def _get_reasoning_display(self, level: str) -> str:
        """Get display name for reasoning level"""
        displays = {
            'none': '🌟 None (Adaptive)',
            'low': '🚀 Low (Fast)',
            'medium': '⚖️ Medium (Balanced)',
            'high': '🧠 High (Thorough)',
            'xhigh': '💎 Extra High (Maximum Quality)',
            'max': '🚀💎 Max (Deepest Reasoning, Slowest)'
        }
        return displays.get(level, '⚖️ Medium (Balanced)')
    
    def _get_verbosity_display(self, level: str) -> str:
        """Get display name for verbosity"""
        displays = {
            'low': '📝 Concise',
            'medium': '📄 Standard',
            'high': '📚 Detailed'
        }
        return displays.get(level, '📄 Standard')
    
    @staticmethod
    def _coerce_choice(value, valid, default):
        """Return `value` only if it is one of `valid`, else `default`.

        Slack rejects a whole static_select/radio block when its initial_option value is not
        also present in `options` (invalid_arguments on views.open). A stored value can drift
        out of range when an option is dropped (e.g. a retired gpt-image-1-mini default), so
        every stored select value is coerced against its live option list before it is rendered.
        """
        return value if value in valid else default

    def _get_image_size_display(self, size: str) -> str:
        """A stored `image_size` in the vocabulary the modal now speaks.

        A grid value reads as its shape and tier with the pixels in brackets
        ("Widescreen · Large (1920 × 1088)"); anything off-grid keeps its raw dimensions.
        """
        if not size or size == 'auto':
            return 'Auto'
        shape_labels = dict(_IMAGE_SHAPES)
        tier_labels = dict(_IMAGE_TIERS)
        match = self.shape_tier_for(size, [s for s, _ in _IMAGE_SHAPES],
                                    [t for t, _ in _IMAGE_TIERS])
        wxh = _WXH_RE.match(size)
        pixels = f"{wxh.group(1)} × {wxh.group(2)}" if wxh else size
        if match is None:
            return f"Custom ({pixels})"
        shape, tier = match
        # The emoji belongs on the select option, not in a one-line summary.
        label = shape_labels.get(shape, shape).split(' ', 1)[-1]
        return f"{label} · {tier_labels.get(tier, tier).split(' —')[0]} ({pixels})"
    
    def _get_fidelity_display(self, fidelity: str) -> str:
        """Get display name for input fidelity"""
        displays = {
            'high': '🎨 Preserve Original Style',
            'low': '✨ Allow Reinterpretation'
        }
        return displays.get(fidelity, '🎨 Preserve Original Style')
    
    def _get_vision_detail_display(self, detail: str) -> str:
        """Get display name for vision detail"""
        displays = {
            'auto': '🤖 Auto',
            'low': '🔍 Low Detail',
            'high': '🔬 High Detail'
        }
        return displays.get(detail, '🤖 Auto')

    def _get_image_quality_display(self, quality: str) -> str:
        """Get display name for image quality"""
        # Multipliers are MEASURED image-token counts relative to High (sunburst, three sizes,
        # 2026-09-09); `auto` bills like Low on the 2.5 family. Owner asked for them on the
        # picker so the cost of a click is visible where the click happens.
        displays = {
            'auto': 'Auto (≈ Low)',
            'low': 'Low (0.1× cost)',
            'medium': 'Medium (0.25× cost)',
            'high': 'High (1× · default)',
            'xhigh': 'Extra High (2× cost)',
            'max': 'Maximum (4× cost)'
        }
        return displays.get(quality, 'Auto')

    def _get_image_background_display(self, background: str) -> str:
        """Get display name for image background"""
        displays = {
            'auto': 'Auto',
            'transparent': 'Transparent',
            'opaque': 'Opaque'
        }
        return displays.get(background, 'Auto')

    def _get_image_model_display_name(self, model: str) -> str:
        """Get user-friendly image model name"""
        displays = {
            # Static-select rows truncate around 30 characters in Slack's dropdown, so these stay
            # short; the longer classification lives in the tool descriptions, not the picker.
            'gpt-image-2.5-flare': 'GPT-Image-2.5 Flare (Faster)',
            'gpt-image-2.5-sunburst': 'GPT-Image-2.5 Sunburst (Best)',
            'gpt-image-2': 'GPT-Image-2 (Legacy)',
            'gpt-image-1': 'GPT-Image-1 (Legacy)',
            'gpt-image-1-mini': 'GPT Image 1 Mini',
        }
        return displays.get(model, model)