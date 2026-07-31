"""What the rich gate left behind, and the proof it is gone.

Commit 6 replaced a multi-signal participation gate with a one-bit wake gate. That made a lot of
machinery unreachable rather than wrong — gate vision, the ambient image hold that fed it, the
prose tails rendered for its prompt, the capability inventory it weighed, the ranked emoji
shortlist it chose from. Unreachable code is not harmless: it reads as live to anyone changing
nearby code, it keeps its config keys documented as if they did something, and the hold in
particular kept a resolver contract alive that nothing was going to call.

The same file now also guards the SECOND retirement, ChannelPulse. The pulse was the responder's
answer to "what is happening in this channel" — an in-memory ring of recent message text, a
rendered "[Recent channel activity]" envelope, a people line, per-message reaction counts, a
one-page backfill per channel per process. The channel stream replaced all of it by rebuilding
from Slack and the database on every turn, which is both current and complete where the ring was
neither. The danger in a retirement like that is a half-return: one envelope injection, one
`getattr(client, "channel_pulse", None)`, and the stream is quietly competing with a stale ring
for the same job.

These tests are absence assertions, which are usually a smell. They earn their place here because
each of these things came back at least once during the rewrite as somebody "restored" an input
that no longer had a use. What the file does NOT assert is equally deliberate: everything that
survives — the actor tail, the people summary, the custom-emoji catalog and its search tool — is
covered by its own tests and must keep working.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SKIP_DIRS = {"tests", "venv", ".venv", "node_modules", "build", "dist", "__pycache__",
             "site-packages", "Docs", "logs", "data"}


def _sources():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS or any(p.startswith(".") for p in rel.parts):
            continue
        yield rel, path.read_text(encoding="utf-8")


def _defined_names(src: str) -> set:
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


# --------------------------------------------------------------------------- modules

def test_gate_vision_module_is_gone():
    """It downloaded and resized pictures so a classifier could judge them. The binary gate never
    looks at an image — it decides whether the RESPONDER runs, and the responder is what reads the
    picture.

    The NAME survives in two honest places and this test must not chase them: the schema comment
    listing `gate_vision` as a legal historical `derivation_source`, and prose explaining where a
    mimetype allowlist came from. Only the module and its callers are gone, so the check is
    structural (imports and attribute access) rather than a text search."""
    assert not (ROOT / "message_processor" / "gate_vision.py").exists()
    assert not (ROOT / "tests" / "unit" / "test_gate_vision.py").exists()

    offenders = []
    for rel, src in _sources():
        tree = ast.parse(src, filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[-1] == "gate_vision" for a in node.names):
                    offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == "gate_vision" for a in node.names) \
                        or (node.module or "").endswith("gate_vision"):
                    offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "gate_vision":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, "gate_vision still has callers at: " + ", ".join(offenders)


def test_the_rich_gates_eval_harness_is_gone():
    """It scored respond/react/ignore verdicts against a `must_be` set. Those labels no longer
    exist, and porting it would have graded the new gate against the old gate's standard — which
    is precisely what the binary design changes."""
    assert not (ROOT / "tests" / "integration" / "participation_eval.py").exists()


@pytest.mark.parametrize("symbol", [
    "render_capabilities_line",  # the inventory the gate weighed
    "render_thread_tail",        # prose tails rendered for the gate prompt
    "render_channel_addressee_tail",
    "top_custom_reactions",      # the ranked emoji shortlist
    "_reaction_vocab",
    "resolve_gate",              # the ambient hold's release contract
    "_defer_image",
    "_store_gate_observation",
    "_gate_will_see_images",
    "defer_images",
    "handle_response",           # the dead alternate delivery path
    "pulse_tail_text_truncate",
])
def test_no_source_file_still_names_it(symbol):
    # Word-boundary matched: `handle_response` is a substring of the live
    # `handle_response_feedback` Slack action handler, which has nothing to do with the deleted
    # dispatcher, and a substring sweep would demand its deletion too.
    import re
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    offenders = [f"{rel}" for rel, src in _sources() if pattern.search(src)]
    assert not offenders, f"{symbol} still appears in: {', '.join(offenders)}"


# The pulse's names, which must not be REACHED — a retirement note that says what was retired is
# good practice, so unlike the gate symbols above these are swept structurally (attribute access
# and calls in the AST) rather than textually. Prose may say "channel_pulse"; code may not.
_RETIRED_PULSE_NAMES = frozenset({
    "channel_pulse",                 # the attribute
    "_build_pulse_envelope",         # the responder's injection site
    "render_envelope", "render_envelope_with_meta",
    "recent_speakers",               # the people line (membership count moved to the suffix)
    "recent_taggable_speakers",      # superseded by the stream's taggable roster
    "count_since", "snapshot_pulse", "upsert_artifacts",
    "ensure_backfill",               # the one-page-per-channel-per-process seed
    "_feed_channel_pulse", "note_reaction_pulse",
    "_record_own_reply_pulse", "_record_own_reaction_pulse",
    "record_own_reaction", "remove_own_reaction",
    "pulse_supplementary_budget",
    # the narrative's retired refresh path (the stream is the room now)
    "render_for_channel", "maybe_refresh", "_ring_counts", "_decide_build",
})


def test_no_source_file_still_reaches_the_pulse():
    offenders = []
    for rel, src in _sources():
        tree = ast.parse(src, filename=str(rel))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
            if name in _RETIRED_PULSE_NAMES:
                offenders.append(f"{rel}:{node.lineno} ({name})")
    assert not offenders, "the pulse is still reached at: " + ", ".join(offenders)


def test_the_pulse_config_keys_are_gone():
    """A key nobody reads still documents itself in .env.example and still looks tunable."""
    from config import BotConfig

    fields = set(BotConfig.__dataclass_fields__)
    for gone in ("enable_channel_pulse", "channel_pulse_size", "pulse_text_truncate",
                 "channel_pulse_envelope_max", "pulse_thread_tails_max",
                 "pulse_thread_tail_channels_max",
                 # the narrative's refresh cadence went with the refresh path
                 "channel_summary_refresh_msgs", "channel_summary_ttl_hours",
                 "channel_summary_failure_cooldown_hours"):
        assert gone not in fields, gone
    for renamed in ("actor_tail_threads_max", "actor_tail_channels_max",
                    "participation_thread_tail", "index_drain_timeout_seconds"):
        assert renamed in fields, renamed
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    for gone_env in ("ENABLE_CHANNEL_PULSE", "CHANNEL_PULSE_SIZE", "PULSE_TEXT_TRUNCATE",
                     "CHANNEL_PULSE_ENVELOPE_MAX", "PULSE_THREAD_TAILS_MAX",
                     "PULSE_THREAD_TAIL_CHANNELS_MAX"):
        assert gone_env not in env, gone_env
    for kept_env in ("ACTOR_TAIL_THREADS_MAX", "ACTOR_TAIL_CHANNELS_MAX",
                     "INDEX_DRAIN_TIMEOUT_SECONDS"):
        assert kept_env in env, kept_env


def test_thread_state_no_longer_carries_a_dead_system_prompt():
    """It was written and never read — for months the channel-memory block went into it and
    reached no model call at all. The field outliving that bug is how the next one starts."""
    from thread_manager import ThreadState

    assert "system_prompt" not in ThreadState.__dataclass_fields__


# --------------------------------------------------------------------------- what must remain

def test_the_context_helpers_that_survive_the_pulse_survive():
    """The deletions above are about inputs nothing reads any more. The people summary still
    renders a roster for the responder, and the actor tail still answers the one structural
    question the ring was kept for."""
    from message_processor.people_tools import format_people_summary
    from slack_client import actor_tail

    assert callable(format_people_summary)
    for kept in ("record", "remove", "thread_has_other_bot", "reconcile_window", "generation"):
        assert hasattr(actor_tail.actor_tail, kept), kept


def test_the_channel_pulse_module_is_gone():
    """Structural, not textual: the module file and every import of it. The NAME survives in
    honest prose — the actor tail's own docstring says where it was extracted from — and this test
    must not chase that."""
    assert not (ROOT / "slack_client" / "channel_pulse.py").exists()
    assert not (ROOT / "tests" / "unit" / "test_channel_pulse.py").exists()
    assert not (ROOT / "tests" / "unit" / "test_thread_tail_context.py").exists()

    offenders = []
    for rel, src in _sources():
        tree = ast.parse(src, filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[-1] == "channel_pulse" for a in node.names):
                    offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").endswith("channel_pulse"):
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, "channel_pulse is still imported at: " + ", ".join(offenders)


def test_the_narrative_no_longer_reaches_a_turn():
    """The channel narrative was injected as a role:user block on every channel turn. The stream
    carries the room now, so the ONE surviving consumer is the join intro, which has no turn to
    read a stream from. What must stay: the neutralizing frame (identical bytes wherever it is
    injected), the mutation invalidation, and the build itself."""
    import inspect

    from message_processor.channel_summary import ChannelSummaryService

    for gone in ("render_for_channel", "maybe_refresh", "_decide_and_build", "_decide_build",
                 "_ring_counts", "_in_cooldown", "_age_hours"):
        assert not hasattr(ChannelSummaryService, gone), gone
    for kept in ("render_block", "note_message_mutation", "build_for_intro", "shutdown"):
        assert hasattr(ChannelSummaryService, kept), kept
    assert "pulse" not in inspect.signature(ChannelSummaryService.build_for_intro).parameters


def test_the_reaction_lease_carries_no_pulse_receipt():
    """The receipt existed so a taken-back reaction could take its synthetic ring entry back with
    it. There is no ring entry, and the stream re-reads reactions from Slack — a receipt now would
    be a key nothing consumes."""
    import inspect

    from slack_client.messaging import SlackMessagingMixin

    assert "pulse" not in inspect.getsource(SlackMessagingMixin._reserve_and_react)
    assert "pulse" not in inspect.getsource(SlackMessagingMixin._settle_removal_slot)


def test_the_custom_emoji_catalog_and_search_survive():
    """Only the RANKED SHORTLIST died — the thing that picked a handful of names to paste into the
    old gate prompt. The responder still looks a name up when it wants one."""
    from slack_client.messaging import WorkspaceEmojiCache

    assert hasattr(WorkspaceEmojiCache, "get_custom_emoji_names")
    assert hasattr(WorkspaceEmojiCache, "refresh")


def test_no_destructive_migration_was_introduced():
    """The emoji tally table stops being created, but an existing installation keeps its rows. A
    DROP here would destroy data on upgrade to buy nothing: an orphaned table costs a few
    kilobytes, and a migration that deletes user data to tidy up is not a tidy-up."""
    # Pre-existing drops (the v3 `messages` mirror, the retired mute table) are settled history
    # and are not what this guards. The new one that must NOT appear is the emoji tally.
    src = (ROOT / "database.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if "DROP TABLE" in line.upper():
            assert "emoji_usage" not in line, f"destructive migration introduced: {line.strip()}"


def test_the_upgrade_path_from_old_installations_is_intact():
    """All three startup migrations stay, and so does the legacy column one of them reads. A
    database from before any of this must still be able to upgrade directly."""
    from database import DatabaseManager

    for migration in ("migrate_channel_directives_to_policy_async",
                      "migrate_participation_levels_to_binary_async",
                      "migrate_participation_prefs_to_policy_async"):
        assert hasattr(DatabaseManager, migration), migration
    schema = (ROOT / "database.py").read_text(encoding="utf-8")
    assert "directives TEXT" in schema


def test_historical_gate_vision_rows_stay_readable():
    """Only the WRITER died. Rows already in the database carry
    `derivation_source='gate_vision'`, and nothing may treat that value as invalid — the artifact
    it labels is still a real analysis of a real image."""
    schema = (ROOT / "database.py").read_text(encoding="utf-8")
    assert "derivation_source" in schema
    offenders = []
    for rel, src in _sources():
        for lineno, line in enumerate(src.splitlines(), 1):
            if "derivation_source" in line and "!=" in line and "gate_vision" in line:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, "a reader now rejects historical rows at: " + ", ".join(offenders)


def test_the_unleased_ok_markers_are_still_inline():
    """They are executable review evidence, consumed by the stale-guard AST sweep. A central
    registry would separate each justification from the mutation it certifies, and the two would
    drift — which is exactly the failure the markers exist to prevent."""
    marked = [rel for rel, src in _sources() if "# unleased-ok:" in src]
    assert marked, "the unleased-ok markers vanished; the stale-guard sweep has nothing to read"


# --------------------------------------------------------------------------- the streaming fix

def test_the_streaming_background_return_carries_the_same_facts_as_the_terminal_one():
    """A turn can start a background job AND build something in the same round. The streaming
    branch returned four fewer facts than its terminal twin, so the job ate the chart: delivery
    never learned the artifact existed, and a suppressed ack reply looks identical either way."""
    import inspect

    from message_processor.handlers.text import TextHandlerMixin

    src = inspect.getsource(TextHandlerMixin._handle_streaming_text_response)
    marker = 'metadata={"streamed": True, "background_job_started": True'
    assert marker in src
    branch = src[src.index(marker):src.index(marker) + 800]
    for fact in ("artifact_containers", "sandbox_image_assets", "mounted_digests",
                 "response_reaction_committed"):
        assert fact in branch, f"the streaming background return still drops {fact}"
