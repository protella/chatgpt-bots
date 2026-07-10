"""Unit tests for prompts.py (modernized prompt contracts)"""

import pytest
from prompts import (
    SLACK_SYSTEM_PROMPT,
    CLI_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    IMAGE_INTENT_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_PROMPT,
    VISION_DEFAULT_QUESTION,
    VISION_ENHANCEMENT_PROMPT,
    IMAGE_EDIT_SYSTEM_PROMPT,
    IMAGE_GEN_SYSTEM_PROMPT,
)


class TestPrompts:
    """Test that all prompts are properly defined"""

    def test_slack_system_prompt_defined(self):
        """SLACK_SYSTEM_PROMPT carries the teammate identity + Slack formatting essentials"""
        assert SLACK_SYSTEM_PROMPT is not None
        assert isinstance(SLACK_SYSTEM_PROMPT, str)
        assert len(SLACK_SYSTEM_PROMPT) > 0
        assert "Slack" in SLACK_SYSTEM_PROMPT
        # Teammate identity + channel etiquette (modernization contract)
        assert "teammate" in SLACK_SYSTEM_PROMPT
        assert "thread" in SLACK_SYSTEM_PROMPT.lower()

    def test_slack_prompt_channel_brevity(self):
        """Channel-brevity etiquette: brief at top level, long-form in threads, offer to expand"""
        assert "brief" in SLACK_SYSTEM_PROMPT.lower()
        assert "offer to expand" in SLACK_SYSTEM_PROMPT.lower()

    def test_slack_prompt_reaction_as_response(self):
        """A reaction may be the entire response"""
        assert "emoji reaction is your entire response" in SLACK_SYSTEM_PROMPT

    def test_slack_prompt_batch_answer_rule(self):
        """Phase Q: queued multi-sender batches answered in one coherent reply"""
        assert "several queued messages" in SLACK_SYSTEM_PROMPT
        assert "one coherent reply" in SLACK_SYSTEM_PROMPT

    def test_slack_prompt_no_mrkdwn_coaching(self):
        """The converter handles markdown->mrkdwn mechanically; the prompt must not
        teach Slack mrkdwn syntax (old '*bold*' style coaching)."""
        assert "*bold*" not in SLACK_SYSTEM_PROMPT
        assert "normal markdown" in SLACK_SYSTEM_PROMPT

    def test_cli_system_prompt_defined(self):
        assert CLI_SYSTEM_PROMPT is not None
        assert isinstance(CLI_SYSTEM_PROMPT, str)
        assert len(CLI_SYSTEM_PROMPT) > 0
        assert "helpful assistant" in CLI_SYSTEM_PROMPT.lower()

    def test_intent_classifier_prompt_defined(self):
        """INTENT_CLASSIFIER_PROMPT keeps all five labels + the learned rules"""
        assert INTENT_CLASSIFIER_PROMPT is not None
        assert isinstance(INTENT_CLASSIFIER_PROMPT, str)
        for label in ("new", "edit", "vision", "ambiguous", "none"):
            assert label in INTENT_CLASSIFIER_PROMPT
        # Learned production rules survive the trim
        assert "PREVIOUS response type" in INTENT_CLASSIFIER_PROMPT  # continuation rule
        assert "attachments" in INTENT_CLASSIFIER_PROMPT             # vision requires files
        assert "URLs" in INTENT_CLASSIFIER_PROMPT                    # links are not images
        assert "logo" in INTENT_CLASSIFIER_PROMPT.lower()            # logo/icon -> new
        # One-line output instruction
        assert "Output exactly one word" in INTENT_CLASSIFIER_PROMPT

    def test_intent_classifier_backcompat_alias(self):
        """Old constant name still resolves (pre-modernization imports/tests)"""
        assert IMAGE_INTENT_SYSTEM_PROMPT is INTENT_CLASSIFIER_PROMPT

    def test_intent_classifier_is_trimmed(self):
        """The classifier fires on every responded message and sits below OpenAI's
        1024-token cache threshold — it must stay small. chars/4 proxy < 350 tokens."""
        assert len(INTENT_CLASSIFIER_PROMPT) / 4 < 350

    def test_image_analysis_prompt_defined(self):
        assert IMAGE_ANALYSIS_PROMPT is not None
        assert "image" in IMAGE_ANALYSIS_PROMPT.lower()
        assert "concise" in IMAGE_ANALYSIS_PROMPT.lower()
        # Stored as hidden context in every rebuild with images — bounded length
        assert "Maximum 120 words" in IMAGE_ANALYSIS_PROMPT

    def test_vision_default_question_defined(self):
        """Standard question used when the user attaches an image with no real ask"""
        assert isinstance(VISION_DEFAULT_QUESTION, str)
        assert "conversationally" in VISION_DEFAULT_QUESTION

    def test_vision_enhancement_prompt_defined(self):
        assert VISION_ENHANCEMENT_PROMPT is not None
        assert isinstance(VISION_ENHANCEMENT_PROMPT, str)
        assert "conversational" in VISION_ENHANCEMENT_PROMPT.lower()
        assert "troubleshooting" in VISION_ENHANCEMENT_PROMPT.lower()

    def test_image_edit_system_prompt_defined(self):
        """Edit prompt: literal instructions, bounded length, no unasked embellishment"""
        assert IMAGE_EDIT_SYSTEM_PROMPT is not None
        assert "edit" in IMAGE_EDIT_SYSTEM_PROMPT.lower()
        assert "10-80 words" in IMAGE_EDIT_SYSTEM_PROMPT
        assert "Never add elements" in IMAGE_EDIT_SYSTEM_PROMPT
        # The photo-edit-only convention survives (touch-ups must not restyle)
        assert "photo edit only" in IMAGE_EDIT_SYSTEM_PROMPT
        # Style transformations still supported
        assert "Style transformation" in IMAGE_EDIT_SYSTEM_PROMPT

    def test_image_gen_system_prompt_defined(self):
        assert IMAGE_GEN_SYSTEM_PROMPT is not None
        assert "prompt" in IMAGE_GEN_SYSTEM_PROMPT.lower()
        # Kept: length bound + style/camera nudges (still help image models)
        assert "50 and 150 words" in IMAGE_GEN_SYSTEM_PROMPT
        assert "camera" in IMAGE_GEN_SYSTEM_PROMPT.lower()
        # New: literal preservation of explicit user specs
        assert "verbatim" in IMAGE_GEN_SYSTEM_PROMPT

    def test_all_prompts_are_strings(self):
        prompts = [
            SLACK_SYSTEM_PROMPT,
            CLI_SYSTEM_PROMPT,
            INTENT_CLASSIFIER_PROMPT,
            IMAGE_ANALYSIS_PROMPT,
            VISION_DEFAULT_QUESTION,
            VISION_ENHANCEMENT_PROMPT,
            IMAGE_EDIT_SYSTEM_PROMPT,
            IMAGE_GEN_SYSTEM_PROMPT,
        ]
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_prompts_contain_no_template_variables(self):
        prompts = [
            SLACK_SYSTEM_PROMPT,
            CLI_SYSTEM_PROMPT,
            INTENT_CLASSIFIER_PROMPT,
            IMAGE_ANALYSIS_PROMPT,
            VISION_DEFAULT_QUESTION,
            VISION_ENHANCEMENT_PROMPT,
            IMAGE_EDIT_SYSTEM_PROMPT,
            IMAGE_GEN_SYSTEM_PROMPT,
        ]
        for prompt in prompts:
            assert "{" not in prompt or "}" not in prompt  # No f-string style
            assert "{{" not in prompt  # No jinja2 style
            assert "${" not in prompt  # No bash/JS style

    @pytest.mark.critical
    def test_critical_prompts_structure(self):
        """Critical: the pieces production behavior depends on"""
        # Slack formatting essentials
        assert "code blocks" in SLACK_SYSTEM_PROMPT.lower()
        # Username-prefix convention + never-echo rule
        assert 'prefixed "Username: "' in SLACK_SYSTEM_PROMPT
        assert "never copy the format" in SLACK_SYSTEM_PROMPT.lower()
        # Intent classifier has all categories
        for category in ("new", "edit", "vision", "ambiguous", "none"):
            assert category in INTENT_CLASSIFIER_PROMPT
        # Edit prompt distinguishes both edit types
        assert "photo edit only" in IMAGE_EDIT_SYSTEM_PROMPT
        assert "Style transformation" in IMAGE_EDIT_SYSTEM_PROMPT
