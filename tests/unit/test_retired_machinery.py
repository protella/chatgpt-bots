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
from typing import Optional

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


# ======================================================================= P4a compaction (W1)
#
# The THIRD retirement, and the largest. P4a built a background compaction machine — eight
# tables, a crawl with checkpoints, a summary generation pipeline, a telemetry outbox, a
# dormancy state machine, a snapshot store with namespaces and a publication CAS — to answer
# "what if the channel does not fit". The shallow window answers it instead, by not asking for
# the whole channel in the first place. That makes every one of those parts unreachable at once,
# which is exactly the condition under which pieces come back one plausible import at a time.

_COMPACTION_TABLES = frozenset({
    "snapshot_mutation_observations", "snapshot_capture_manifest", "snapshot_anchor_provenance",
    "compaction_crawl_checkpoints", "compaction_event_skeleton", "compaction_telemetry_outbox",
    "pending_recompaction", "compaction_cancellation_intent",
    "channel_snapshots", "channel_snapshot_pointer",
})

# The ONE function allowed to name the dropped tables: the migration that drops them. Scoping the
# allowance to `database.py` as a whole was too loose — a newly added accessor querying a dropped
# table under any other name in that file would have passed.
_TABLE_NAME_EXEMPT = frozenset({pathlib.Path("database.py")})
_TABLE_NAME_EXEMPT_FUNCTION = "_migrate_drop_compaction_schema"

_RETIRED_COMPACTION_NAMES = frozenset({
    # the modules' own entry points
    "ChannelSnapshotCoordinator", "snapshot_coordinator", "select_and_pin",
    "resolve_pending_invalidation", "drain_outbox", "revalidate", "unpin",
    # the turn's carriers
    "snapshot_lease", "compaction_evidence", "snapshot_selection", "CompactionEvidence",
    "_trigger_compaction", "_post_turn_compaction", "_compaction_evidence",
    "_release_snapshot_lease",
    # the database accessors
    "insert_channel_snapshot_async", "publish_channel_snapshot_async", "get_active_snapshot_async",
    "get_snapshot_async", "get_snapshot_row_async", "invalidate_snapshot_async",
    "delete_snapshot_async", "select_snapshot_for_pin_async", "snapshot_manifest_async",
    "snapshot_anchor_provenance_async", "insert_compaction_candidate_async",
    "publish_compaction_candidate_async", "retire_snapshot_lineage_async",
    "rollback_published_generation_async", "mutation_observations_after_async",
    "affected_snapshot_ids_async", "sweep_mutation_observations_async",
    "load_crawl_checkpoint_async", "upsert_crawl_checkpoint_async", "delete_crawl_state_async",
    "commit_crawl_page_async", "seal_event_skeleton_async", "skeleton_slice_async",
    "insert_outbox_rows_async", "read_outbox_batch_async", "delete_outbox_row_async",
    "load_pending_recompaction_async", "merge_pending_recompaction_async",
    "cas_pending_recompaction_async", "write_cancellation_intent_async",
    "terminal_publish_nothing_async", "sweep_snapshots_async", "late_artifact_evidence_async",
    "record_activity_and_mutation_async", "max_mutation_observation_id_async",
    "_init_compaction_schema", "_migrate_snapshot_namespace", "_migrate_retire_v1_pointers",
    # the serializer's v2 grammar
    "render_summary_block", "_summary_item", "_snapshot_text", "stale_marked_payload",
    "render_anchor_block", "anchor_roots_in", "anchor_is_eligible", "escape_anchor_text",
    "render_late_artifact", "render_rehydration", "render_rehydration_omission",
    "build_late_artifact_items", "build_rehydration_item", "rehydration_variant_headroom",
    "SnapshotUnsupportedError", "CoverageNotReady", "CoveragePin", "StreamFloorUnknown",
    # the telemetry vocabulary
    "compaction_snapshot", "validate_outbox_body", "emit_outbox_body", "extract_canonical_body",
    "canonical_body_bytes", "canonical_json",
    # the mutation feed, and the normalizer machinery that fed only it
    "mutation_from_event", "feed_own_mutation", "mutation_kind",
    "mutation_observation_identity", "MUTATION_KIND_NAMES",
    # the capture-manifest renderer and the coordinator's dormant field
    "artifact_render_bytes", "AMBIENT_ARTIFACT_CHARS", "_malformed_pending_seen",
    # the barrier seam
    "pre_resume_after_compaction",
})

_RETIRED_CONFIG_KEYS = (
    "snapshot_retain_generations", "snapshot_retain_days", "compaction_trigger_ratio",
    "compaction_target_ratio", "root_anchor_text_max", "compaction_min_tail",
    "COMPACTION_MIN_TAIL_MAX", "snapshot_anchor_map_bound", "rehydration_max_messages",
    "rehydration_max_bytes", "rehydration_page_budget", "rehydration_time_budget",
    "crawl_page_budget", "crawl_time_budget", "crawl_fixed_headroom_tokens",
    "summary_byte_cap", "revalidation_claim_ttl",
)

# Appendix A7: the serializer-v2 template constants. Grep-driven, so a constant cannot survive by
# merely being unreferenced — an unused template is exactly what gets "restored" later.
_RETIRED_TEMPLATES = (
    "SUMMARY_HEADER_TEMPLATE", "SUMMARY_PREAMBLE", "SUMMARY_END_TEXT", "ANCHOR_HEADER_TEXT",
    "ANCHOR_NONE_LINE", "ANCHOR_OMITTED_TEMPLATE", "ANCHOR_UNAVAILABLE_TEXT",
    "ANCHOR_TOMBSTONE_MARKER", "STALE_MARKER_TEMPLATE", "STATUS_PUBLISHED_STALE",
    "LATE_ARTIFACT_TEMPLATE", "LATE_ARTIFACT_FAILURE_TEMPLATE", "LATE_ARTIFACT_KIND_LINES",
    "REHYDRATION_HEADER", "REHYDRATION_BOUND_CLAUSE", "REHYDRATION_END_TEXT",
    "REHYDRATION_OMISSION_TEMPLATE", "REHYDRATION_REASONS", "SUMMARY_CLAUSE_NONE",
    "SUMMARY_CLAUSE_TEMPLATE", "REASON_CLAUSE_GENESIS", "REASON_CLAUSE_RETENTION",
    "REASON_CLAUSE_DEPTH_TEMPLATE", "REASON_CLAUSE_UNKNOWN",
)


def test_no_module_imports_channel_compaction():
    """T1. REACHES, not mentions: an `import` statement, not the word in a comment. A module
    that still imports either one would fail at boot, but the point is to catch the import being
    ADDED BACK — a plausible `from message_processor.channel_snapshots import ...` in a future
    edit is how a deleted module gets resurrected as a stub."""
    dead_leaves = ("channel_compaction", "channel_snapshots")

    def _is_dead(dotted: str) -> bool:
        return bool(dotted) and dotted.split(".")[-1] in dead_leaves

    offenders = []
    for rel, src in _sources():
        for node in ast.walk(ast.parse(src, filename=str(rel))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_dead(alias.name):
                        offenders.append(f"{rel}:{node.lineno} (import {alias.name})")
            elif isinstance(node, ast.ImportFrom):
                # `from message_processor.channel_compaction import X` names the module in
                # `.module`; `from message_processor import channel_compaction` and
                # `from . import channel_compaction` name it in the ALIASES, with `.module`
                # holding the parent package (or None for a bare relative import). Both forms
                # are how a deleted module comes back, so both are checked.
                if _is_dead(node.module or ""):
                    offenders.append(f"{rel}:{node.lineno} (from {node.module})")
                for alias in node.names:
                    if _is_dead(alias.name):
                        offenders.append(
                            f"{rel}:{node.lineno} (from {node.module or '.'} import "
                            f"{alias.name})")
    assert not offenders, "the retired modules are imported at: " + ", ".join(offenders)


def test_the_compaction_modules_are_gone():
    for name in ("channel_compaction.py", "channel_snapshots.py"):
        assert not (ROOT / "message_processor" / name).exists()


def test_no_compaction_symbol_survives_anywhere():
    """T15. A SYMBOL inventory rather than an import check: an accessor, a turn field or a
    coordinator attribute can outlive the module that used it and read as live to the next
    person. Structural (AST attribute/name/def), so prose in a retirement note stays legal."""
    offenders = []
    for rel, src in _sources():
        for node in ast.walk(ast.parse(src, filename=str(rel))):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            if name in _RETIRED_COMPACTION_NAMES:
                offenders.append(f"{rel}:{node.lineno} ({name})")
    assert not offenders, "retired compaction machinery survives at: " + ", ".join(offenders)


_IMPORT_CALLS = frozenset({"import_module", "__import__", "find_spec", "load_module",
                           "module_from_spec"})
_DEAD_MODULES = ("channel_compaction", "channel_snapshots")


def _dead_leaf(text: str) -> bool:
    return text.strip().split(".")[-1] in _DEAD_MODULES


def _folded(node: ast.AST) -> Optional[str]:
    """A string this expression provably IS, or None. Constants and `+` chains of them.

    Folding matters because `"message_processor." + "channel_compaction"` is a complete module
    path that no single Constant node contains — statically evaluable, so still fair game.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _folded(node.left), _folded(node.right)
        return None if left is None or right is None else left + right
    return None


def _import_aliases(tree: ast.AST) -> set:
    """Names in this module that REFER to an import callable, so a bare-name call through one is
    scanned like the real thing: `im = importlib.import_module`, `from importlib import
    import_module as x`."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _IMPORT_CALLS:
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, (ast.Attribute, ast.Name)):
            referent = (node.value.attr if isinstance(node.value, ast.Attribute)
                        else node.value.id)
            if referent in _IMPORT_CALLS:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.add(target.id)
    return aliases


def _dead_module_strings(tree: ast.AST) -> list:
    """Every place this tree names a removed module as a string. THE ONE SCANNER — the repo-wide
    test and its own regression suite below both run this, so the thing under test and the thing
    proven cannot drift apart."""
    callables = _IMPORT_CALLS | _import_aliases(tree)
    found = []
    flagged = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = (node.func.attr if isinstance(node.func, ast.Attribute)
                  else getattr(node.func, "id", ""))
        if callee not in callables:
            continue
        for arg in ast.walk(node):
            text = _folded(arg)
            if text is not None and _dead_leaf(text):
                flagged.add(id(arg))
                found.append((getattr(arg, "lineno", 0), f"{callee}({text!r})"))
    for node in ast.walk(tree):
        if id(node) in flagged:
            continue
        text = _folded(node)
        if text is not None and "." in text and _dead_leaf(text):
            found.append((getattr(node, "lineno", 0), repr(text)))
    return found


def test_no_dynamic_import_reaches_a_removed_module():
    """T15's second half. A `getattr`/`importlib` route around the AST check above is the one way
    a deleted module comes back without any import statement naming it.

    WHERE THIS STOPS, AND WHY IT STOPS THERE. A runtime-computed module name — one assembled from
    a variable, a config value or a loop — cannot be recognized statically, and chasing it would
    be an arms race this test cannot win. It does not need to win it: **the module files no longer
    exist, so ANY route that actually reaches one raises `ModuleNotFoundError` loudly at the
    call**, and `test_no_compaction_table_survives_a_boot` proves the schema those modules wrote
    is gone too. This scanner exists to catch the PLAUSIBLE reintroduction — an import someone
    adds back while reading nearby code — not to be a sandbox. Do not escalate it further.

    NO FILE EXEMPTIONS — the check is made PRECISE instead. `database.py` legitimately names the
    string "channel_snapshots" in its drop list, because a dropped TABLE happens to share a name
    with a deleted MODULE, and exempting the whole file to cope with that would wave through a
    dynamic import in exactly the file most likely to want one.
    """
    offenders = []
    for rel, src in _sources():
        for lineno, what in _dead_module_strings(ast.parse(src, filename=str(rel))):
            offenders.append(f"{rel}:{lineno} ({what})")
    assert not offenders, "a removed module is named as a string at: " + ", ".join(offenders)


@pytest.mark.parametrize("source,why", [
    ('import importlib\n'
     'name = "message_processor." + "channel_compaction"\n'
     'importlib.import_module(name)\n',
     "a name assembled from adjacent literals"),
    ('import importlib\n'
     'importlib.import_module("message_processor." + "channel_snapshots")\n',
     "the same concatenation, inline"),
    ('from importlib import import_module as _load\n'
     '_load("channel_compaction")\n',
     "an ALIASED import callable called with a bare module name"),
    ('import importlib\n'
     'loader = importlib.import_module\n'
     'loader("channel_snapshots")\n',
     "a module-level alias assignment"),
    ('X = "message_processor.channel_compaction"\n',
     "a plain dotted literal with no call at all"),
    ('import importlib\n'
     'importlib.import_module("message_processor.channel_compaction")\n',
     "the ordinary qualified form"),
])
def test_the_dynamic_import_scanner_catches_the_evasions_it_claims_to(source, why):
    """The scanner's OWN regression suite. Every form here was a real hole at some point in this
    wave, and each is cheap to keep because the subject is the scanner rather than the repo.

    Deliberately absent: a name computed at RUNTIME (from config, from a loop). The docstring
    above says why that one does not need catching."""
    assert _dead_module_strings(ast.parse(source)), f"the scanner MISSES {why}: {source!r}"


def test_the_scanner_does_not_fire_on_a_dropped_table_name():
    """The other direction, and the reason the file exemption could be removed at all: the drop
    list names `channel_snapshots` as a TABLE, which is not a module reference."""
    drop_list = ('TABLES = ("compaction_event_skeleton", "channel_snapshot_pointer",\n'
                 '          "channel_snapshots")\n')
    assert _dead_module_strings(ast.parse(drop_list)) == []


def test_no_compaction_table_name_survives_outside_the_cleanup_migration():
    """T15's third half. The table names are the durable trace: an accessor written against one
    would fail at runtime rather than at import, and only in the channel that hit it.

    The allowance is ONE FUNCTION, not one file. `database.py` legitimately names all ten in the
    migration that drops them; anywhere else in that same file — a new accessor, a stray helper —
    is exactly the regression this exists to catch, and a file-wide exemption would wave it
    through.
    """
    offenders = []
    for rel, src in _sources():
        tree = ast.parse(src, filename=str(rel))
        allowed_lines = set()
        if rel in _TABLE_NAME_EXEMPT:
            for node in ast.walk(tree):
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == _TABLE_NAME_EXEMPT_FUNCTION):
                    allowed_lines = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if node.lineno in allowed_lines:
                continue
            for table in _COMPACTION_TABLES:
                if table in node.value:
                    offenders.append(f"{rel}:{node.lineno} ({table})")
    assert not offenders, "a dropped table is still named at: " + ", ".join(offenders)


def test_the_normalized_event_carries_no_mutation_identity_field():
    """Finding 5's regression rail. `event_id` existed ONLY to key a durable mutation
    observation, and a dataclass FIELD is invisible to the symbol sweep above — it would come
    back as a one-line addition that reads like ordinary Slack metadata."""
    from slack_client.normalizer import NormalizedEvent

    fields = set(NormalizedEvent.__dataclass_fields__)
    assert "event_id" not in fields
    # The live shape, stated so a removal here is as loud as an addition.
    assert fields == {"kind", "team_id", "channel_id", "subject_ts", "activity_ts",
                      "root_if_indexed", "owner_probe_ts", "deleted_ts", "message"}


def test_the_database_module_defines_no_stray_ts_helper():
    """Finding 7, SCOPED TO `database.py` DELIBERATELY. Its `_ts_key` existed only for the
    snapshot accessors and went with them, but the name is generic and FOUR unrelated live ones
    remain — `activity_index.py`, `channel_summary.py`, `thread_management.py`, and an alias in
    `participation.py`. A repo-wide symbol tripwire on it would demand all four be deleted, so
    the check names the one module the removal applies to."""
    import database

    assert not hasattr(database, "_ts_key")
    assert not hasattr(database, "canonical_json")


def test_no_retired_constant_survives():
    """T7. Two name lists, both grep-driven so a constant cannot survive by being unreferenced:
    the config keys (which would still read as tunable in `.env.example`) and Appendix A7's
    serializer-v2 templates (which would still read as part of the grammar)."""
    from config import BotConfig, config

    loaded = BotConfig()
    for key in _RETIRED_CONFIG_KEYS:
        assert not hasattr(loaded, key), f"BotConfig still exposes {key}"
        assert not hasattr(config, key), f"the live config still exposes {key}"

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in _RETIRED_CONFIG_KEYS:
        assert key.upper() not in env_example, f".env.example still documents {key.upper()}"

    serializer = (ROOT / "message_processor" / "channel_stream.py").read_text(encoding="utf-8")
    stream_module = __import__("message_processor.channel_stream", fromlist=["x"])
    for template in _RETIRED_TEMPLATES:
        assert template not in serializer, f"channel_stream.py still carries {template}"
        assert not hasattr(stream_module, template)


def test_nothing_depends_on_a_snapshot_or_crawl(tmp_path, monkeypatch):
    """T58. W1's inventory tripwires, re-run as a W2 GATE.

    W2 is the wave that rebuilds the window, and every part of it — the selector, the layout, the
    re-anchor — sits exactly where the snapshot and the crawl used to sit. That is the shape of
    edit under which a removed accessor, a dropped table name or a retired template comes back:
    not as a deliberate restoration, but as a plausible line written while reading the code the
    removal left behind.

    THIS IS THE SAME INVENTORY, RE-RUN — not a second implementation of it. The name lists, the
    scanners and the boot check all live above and are shared, deliberately: a parallel W2 copy
    would be a second thing to keep in step, and the first time the two disagreed the gate would
    be grading a list nobody had updated. What W2 adds is the OCCASION, not the content — the
    ten table names, the symbol inventory (`StreamFloorUnknown`, `CoverageNotReady` and
    `canonical_json` among them), `database.py`'s scoped `_ts_key`, the retired config keys and
    Appendix A7's serializer-v2 templates are all asserted here through the W1 tests that own
    them.
    """
    test_no_module_imports_channel_compaction()
    test_the_compaction_modules_are_gone()
    test_no_compaction_symbol_survives_anywhere()
    test_no_dynamic_import_reaches_a_removed_module()
    test_no_compaction_table_name_survives_outside_the_cleanup_migration()
    test_the_database_module_defines_no_stray_ts_helper()
    test_no_retired_constant_survives()
    test_no_compaction_table_survives_a_boot(tmp_path, monkeypatch)


def test_no_compaction_table_survives_a_boot(tmp_path, monkeypatch):
    """T2. A REAL DatabaseManager brought all the way up on a fresh file — init_schema AND the
    migrations — and none of the ten tables exists afterwards. Asserted against `sqlite_master`
    rather than against the DDL source, because a CREATE TABLE reachable from any code path at
    all is what matters, including one a migration adds back."""
    import os

    from database import DatabaseManager

    monkeypatch.setitem(os.environ, "DATABASE_DIR", str(tmp_path))
    db = DatabaseManager("boot")
    try:
        assert db.db_path.startswith(str(tmp_path)), "the test must not touch the real database"
        db.init_schema()
        tables = {row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        db.close()
    assert not (tables & _COMPACTION_TABLES), sorted(tables & _COMPACTION_TABLES)
    # And the ones that must still be there, so this cannot pass by creating no schema at all.
    assert {"channel_thread_activity", "channel_coverage", "outbound_receipts"} <= tables


# --------------------------------------------------------------------------- the live-test skill

_SKILL = ROOT / ".claude" / "skills" / "live-bot-test" / "SKILL.md"

# What the skill must NOT still describe. Both were retired: the pulse in P2 (the module is gone —
# see `test_the_channel_pulse_module_is_gone`), and the rich gate's verdict call with it. A skill
# is not documentation housekeeping — harness accuracy is first-class (owner rule, P1), and a
# skill describing retired machinery sends the next live pass down a path that no longer exists.
_RETIRED_IN_THE_SKILL = (
    "ChannelPulse", "channel_pulse", "classify_participation", "classifier",
    # Retired MECHANISM VOCABULARY, not just retired symbols. §7.1b's "keep the contamination
    # traps verbatim" protects the LESSONS — residue accumulates, a restart brings it back, vary
    # the wording — not a description of a ring buffer that no longer exists or of a rich gate
    # that now returns one bit. A run book that describes machinery the reader cannot find sends
    # the next live pass looking for it.
    "channel ring", "verdict reason",
)

# What it must name instead, and BOTH paths are the operative requirement (§7.1b): the durable
# harness AND the capped unit path. A version asserting only the second would pass on a skill that
# never pointed at the harness at all.
_REQUIRED_IN_THE_SKILL = ("stream_render", "tests/live/", "tests/unit/")


def test_the_live_test_skill_matches_the_shipped_architecture():
    """T115. §7.1b — the skill is updated in W5, and this is what "updated" means."""
    text = _SKILL.read_text(encoding="utf-8")
    lowered = text.lower()

    still_there = [name for name in _RETIRED_IN_THE_SKILL if name.lower() in lowered]
    assert not still_there, f"{_SKILL.name} still describes retired machinery: {still_there}"

    missing = [name for name in _REQUIRED_IN_THE_SKILL if name not in text]
    assert not missing, f"{_SKILL.name} never names: {missing}"

    # The standing rules §7.1b keeps: the bot token cannot trigger the bot, so seeds post as the
    # user; and the authorized channel with prod hands-off.
    assert "SLACK_TEST_USER_TOKEN" in text
    assert "DEV_TREAT_BOT_IDS_AS_HUMAN" in text
    assert "C0BKX77NU66" in text
    assert "Contamination" in text
