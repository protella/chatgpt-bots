"""Status variance — file-based loading-message pools and stage-keyed pipeline variants.

Covers: message-file parsing (comments/blanks), loading-pool precedence
(inline env > file > defaults), [stage]-sectioned pipeline files,
pipeline_status() selection/formatting/fallbacks, and the history-rebuild
markers derived from the variants.
"""
import pytest

import config as config_module
from config import (
    BotConfig,
    config,
    pipeline_status,
    pipeline_status_markers,
    _load_message_file,
    _load_stage_map,
)


@pytest.fixture(autouse=True)
def _clear_file_caches():
    _load_message_file.cache_clear()
    _load_stage_map.cache_clear()
    yield
    _load_message_file.cache_clear()
    _load_stage_map.cache_clear()


# ---------------- message file parsing ----------------

def test_message_file_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "msgs.txt"
    f.write_text("# a comment\n\none…\n  two…  \n\n# more\nthree…\n", encoding="utf-8")
    assert list(_load_message_file(str(f))) == ["one…", "two…", "three…"]


def test_missing_message_file_is_empty_not_fatal():
    assert _load_message_file("/nope/does-not-exist.txt") == ()


def test_bundled_pools_load_and_are_plain_text():
    # The committed generic pool: 100 messages, no shortcodes/emoji.
    msgs = _load_message_file("status_messages/loading_messages.generic.txt")
    assert len(msgs) == 100
    assert all(":" not in m for m in msgs)


# ---------------- loading pool precedence ----------------

def test_pool_prefers_file_by_default(tmp_path, monkeypatch):
    f = tmp_path / "brand.txt"
    f.write_text("branded one…\nbranded two…\n", encoding="utf-8")
    monkeypatch.setattr(config, "status_loading_messages_inline", False)
    monkeypatch.setattr(config, "status_loading_messages_file", str(f))
    assert config.get_loading_messages() == ["branded one…", "branded two…"]


def test_inline_env_beats_file(monkeypatch):
    monkeypatch.setattr(config, "status_loading_messages_inline", True)
    monkeypatch.setattr(config, "status_loading_messages", ["inline…"])
    assert config.get_loading_messages() == ["inline…"]


def test_unreadable_file_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(config, "status_loading_messages_inline", False)
    monkeypatch.setattr(config, "status_loading_messages_file", "/nope/missing.txt")
    assert config.get_loading_messages() == config.status_loading_messages


def test_default_config_points_at_generic_pool(monkeypatch):
    monkeypatch.delenv("STATUS_LOADING_MESSAGES", raising=False)
    monkeypatch.delenv("STATUS_LOADING_MESSAGES_FILE", raising=False)
    fresh = BotConfig()
    assert fresh.status_loading_messages_file.endswith("loading_messages.generic.txt")
    assert len(fresh.get_loading_messages()) == 100


# ---------------- stage map parsing ----------------

def test_stage_map_sections(tmp_path):
    f = tmp_path / "stages.txt"
    f.write_text(
        "# header comment\n[alpha]\na1…\na2…\n\n[beta]\nb1 {x}…\n[empty]\n",
        encoding="utf-8",
    )
    m = _load_stage_map(str(f))
    assert m["alpha"] == ("a1…", "a2…")
    assert m["beta"] == ("b1 {x}…",)
    assert "empty" not in m  # sections with no variants are dropped


def test_bundled_pipeline_file_has_expected_stages():
    m = _load_stage_map(config.pipeline_messages_file)
    for stage in [
        "understanding_request", "generating_response", "generating_image",
        "editing_image", "enhancing_prompt", "finding_image", "downloading_image",
        "analyzing_image", "analyzing_images", "processing_document",
        "extracting_document", "summarizing_document", "combining_documents",
        "optimizing_history", "rebuilding_history",
    ]:
        assert m.get(stage), f"missing stage: {stage}"
        assert all(":" not in v for v in m[stage]), f"emoji/shortcode in {stage}"


# ---------------- pipeline_status ----------------

def test_pipeline_status_picks_a_known_variant():
    variants = _load_stage_map(config.pipeline_messages_file)["editing_image"]
    assert pipeline_status("editing_image", "fallback") in variants


def test_pipeline_status_formats_placeholders():
    text = pipeline_status("extracting_document", "fb", file_name="menu.pdf")
    assert "menu.pdf" in text and "{file_name}" not in text


def test_pipeline_status_unknown_stage_falls_back():
    assert pipeline_status("no_such_stage", "the default") == "the default"


def test_pipeline_status_bad_placeholder_falls_back(tmp_path, monkeypatch):
    f = tmp_path / "stages.txt"
    f.write_text("[weird]\nneeds {missing_key}…\n", encoding="utf-8")
    monkeypatch.setattr(config, "pipeline_messages_file", str(f))
    assert pipeline_status("weird", "safe default", other="x") == "safe default"


def test_pipeline_status_unreadable_file_falls_back(monkeypatch):
    monkeypatch.setattr(config, "pipeline_messages_file", "/nope/missing.txt")
    assert pipeline_status("editing_image", "the default") == "the default"


# ---------------- history-rebuild markers ----------------

def test_markers_cover_variants_and_truncate_templates():
    markers = pipeline_status_markers()
    assert "Rebuilding thread history from Slack…" in markers
    # Templated variants contribute their pre-placeholder prefix…
    assert "Extracting content from" in markers
    # …and no marker retains a raw placeholder.
    assert all("{" not in m for m in markers)
    # Too-short/generic prefixes (e.g. "Reading ", "Opening ") are dropped.
    assert all(len(m) >= 8 for m in markers)
