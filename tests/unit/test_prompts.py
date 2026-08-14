"""Unit tests for prompts.py (modernized prompt contracts)"""

import pytest
from message_processor.prompts import (
    CODE_INTERPRETER_GUIDANCE,
    SLACK_SYSTEM_PROMPT,
    CLI_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_PROMPT,
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
        """Channel-brevity etiquette: brief at top level, long-form detail moves to a thread."""
        assert "brief" in SLACK_SYSTEM_PROMPT.lower()
        assert "use a thread when the request calls for the detail" in SLACK_SYSTEM_PROMPT

    def test_slack_prompt_reaction_as_response(self):
        """A reaction may be the entire response"""
        assert "emoji reaction is your entire response" in SLACK_SYSTEM_PROMPT

    def test_slack_prompt_followups_allowed_only_for_real_next_step(self):
        """Follow-up offers are permitted ONLY for a concrete emerging next step — never generic
        filler. Replaces the old blanket 'DO NOT offer follow-up' ban."""
        assert "Follow-up offers are fine only when" in SLACK_SYSTEM_PROMPT
        assert "concrete next step" in SLACK_SYSTEM_PROMPT
        # the filler examples that are still banned
        assert "Anything else?" in SLACK_SYSTEM_PROMPT
        # the old blanket prohibition is gone
        assert "DO NOT offer follow-up questions or actions" not in SLACK_SYSTEM_PROMPT

    def test_slack_prompt_batch_answer_rule(self):
        """Phase Q: queued multi-sender batches answered in one coherent reply"""
        assert "several queued messages" in SLACK_SYSTEM_PROMPT
        assert "one coherent reply" in SLACK_SYSTEM_PROMPT

    def test_slack_prompt_makes_the_reply_deliver_a_failed_file(self):
        """Model-first failure delivery: nothing is posted ahead of the answer for a file that
        couldn't be read, so the reply itself owes the user the news."""
        assert "couldn't be read" in SLACK_SYSTEM_PROMPT
        assert "which file, why in a word, and what they can do about it" in SLACK_SYSTEM_PROMPT
        assert "never guess at what was in it" in SLACK_SYSTEM_PROMPT

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

    def test_image_analysis_prompt_defined(self):
        assert IMAGE_ANALYSIS_PROMPT is not None
        assert "image" in IMAGE_ANALYSIS_PROMPT.lower()
        assert "concise" in IMAGE_ANALYSIS_PROMPT.lower()
        # Stored as hidden context in every rebuild with images — bounded length
        assert "Maximum 120 words" in IMAGE_ANALYSIS_PROMPT

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
            IMAGE_ANALYSIS_PROMPT,
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
            IMAGE_ANALYSIS_PROMPT,
            IMAGE_EDIT_SYSTEM_PROMPT,
            IMAGE_GEN_SYSTEM_PROMPT,
        ]
        for prompt in prompts:
            assert "{" not in prompt or "}" not in prompt  # No f-string style
            assert "{{" not in prompt  # No jinja2 style
            assert "${" not in prompt  # No bash/JS style

    def test_ci_guidance_forbids_sandbox_image_inspection(self):
        """Regression (live): asked whether a posted screenshot was forged, the model pushed the
        thread's auto-mounted screenshots through matplotlib and published the composite back
        into the channel as `output_1.png` — titled with their /mnt/data hash names. It already
        had vision; the sandbox round taught it nothing. The guidance must say so outright."""
        assert "NEVER push one through the sandbox" in CODE_INTERPRETER_GUIDANCE
        # ...and must forbid handing an existing thread image back, composites included.
        assert "NEVER re-post an image that is already in this thread" in CODE_INTERPRETER_GUIDANCE
        assert "several stitched into one figure" in CODE_INTERPRETER_GUIDANCE
        # The escape hatch stays open: building something genuinely new is still fair game.
        assert "Build a NEW image only when the user asked for" in CODE_INTERPRETER_GUIDANCE

    def test_ci_guidance_is_honest_about_which_images_are_visible(self):
        """The real capability gap behind the bug: only the CURRENT message's attachments ride as
        pixels (utilities.py sends `input_image`); earlier images enter context as TEXT — either
        `[Visual context …]` analysis or a bare URL. Telling the model it can see every image
        would trade a redundant screenshot dump for confident bluffing about unseen pixels, so the
        guidance must state the split AND give it an honest out."""
        assert "answering right now are in front of you" in CODE_INTERPRETER_GUIDANCE
        assert "only a written description, not the" in CODE_INTERPRETER_GUIDANCE
        assert "Do not bluff from the" in CODE_INTERPRETER_GUIDANCE
        # The honest out is now a real capability: re-attach the pixels via the tool...
        assert "call `view_image` and actually look" in CODE_INTERPRETER_GUIDANCE
        # ...and NOT a trip through the sandbox, which publishes the render as a side effect.
        assert "do not go hunting through" in CODE_INTERPRETER_GUIDANCE

    def test_ci_guidance_states_attachments_automount(self):
        """The old text claimed the sandbox 'starts EMPTY'. It doesn't — the turn's attachments
        auto-mount (handlers/text.py), which is how loose screenshots were sitting in /mnt/data
        waiting to be re-rendered. A prompt that misdescribes the sandbox invites that bug."""
        assert "starts EMPTY" not in CODE_INTERPRETER_GUIDANCE
        assert "land in\n/mnt/data on their own" in CODE_INTERPRETER_GUIDANCE
        assert "not a to-do list" in CODE_INTERPRETER_GUIDANCE

    def test_ci_guidance_scopes_compute_rule_to_data(self):
        """'COMPUTE, don't eyeball' must read as being about tabular/numeric data. Left generic
        ('attached data'), it reads as a standing order to inspect images in the sandbox too."""
        assert "attached DATA — a spreadsheet, CSV, table" in CODE_INTERPRETER_GUIDANCE

    def test_ci_guidance_routes_long_builds_to_the_background_job(self):
        """Live 2026-07-24: asked for a slide, the model built it in the INLINE sandbox — one call
        ran 10 minutes, during which the reply sat frozen at "Yep" with no progress surface (the
        status placeholder is deleted once the stream owns the message, so there is nothing left
        to update). The routing decision happens before the call and all of it runs inside one API
        call, so there is no point at which we can intervene — the prompt is the only lever."""
        assert "KEEP INLINE SANDBOX WORK SHORT" in CODE_INTERPRETER_GUIDANCE
        # It must say WHY, or the rule reads as arbitrary and loses to "but I can do it here".
        assert "nothing you have written reaches the user until the whole turn ends" in \
            CODE_INTERPRETER_GUIDANCE
        # ...name the destination, with the mode...
        assert "`start_background_job`" in CODE_INTERPRETER_GUIDANCE
        assert "mode\n`build`" in CODE_INTERPRETER_GUIDANCE
        # ...and cover the retry case, which is how a 30-second attempt becomes ten minutes.
        assert "if your first approach in here fails" in CODE_INTERPRETER_GUIDANCE

    def test_background_job_tool_advertises_itself_for_slow_sandbox_work(self):
        """The other half of the same routing fix: the model reading `start_background_job` must
        recognise the case. Before, `build` read as being only about decks and spreadsheets from
        material that already exists — nothing said a slow inline build belonged here."""
        from message_processor.research_tools import get_start_background_job_schema
        desc = get_start_background_job_schema()["description"]
        assert "minutes rather than seconds" in desc
        assert "frozen half-sentence" in desc

    def test_background_job_tool_excludes_slack_history_work(self):
        """Live: asked to summarize a channel's last month, the model dispatched a research job.
        Research jobs have no Slack access, so the job could only work from the snapshot it was
        handed and honestly declined. The exclusion has to be at the point of choice."""
        from message_processor.research_tools import get_start_background_job_schema
        desc = get_start_background_job_schema()["description"]
        assert "cannot fetch or search Slack history or the workspace" in desc
        assert "conversation snapshot captured at dispatch" in desc
        assert "use `export_conversation` and compute over the export instead" in desc
        # An export made in THIS turn lives in a different container than the job's sandbox.
        assert "do not pass it an export path from this turn" in desc

    def test_guidance_tells_the_model_a_job_can_be_stopped(self):
        """Live 2026-08-09: the bot agreed to stand down on a doc job and the job ran on for
        eight more minutes, then delivered the doc. The tool now exists; the guidance has to
        say that agreeing is not stopping, or the model keeps agreeing and nothing happens."""
        from message_processor.prompts import LOCAL_TOOLS_GUIDANCE
        assert "cancel_background_job" in LOCAL_TOOLS_GUIDANCE
        assert "agreeing to a change in words changes nothing" in LOCAL_TOOLS_GUIDANCE
        assert "posts whatever it built" in LOCAL_TOOLS_GUIDANCE
        # ...without turning every follow-up question into a cancel.
        assert "A refinement is not a withdrawal" in LOCAL_TOOLS_GUIDANCE

    def test_guidance_separates_steering_a_job_from_killing_it(self):
        """The same incident, other half: five corrections arrived for a job that was WANTED,
        and the only tool that existed would have thrown the work away. Cancel and update have
        to be told apart by what is being asked, or the model reaches for the brake when it
        needs the steering wheel — and, worse, replies to a correction without sending it."""
        from message_processor.prompts import LOCAL_TOOLS_GUIDANCE
        assert "update_background_job(job_id, note)" in LOCAL_TOOLS_GUIDANCE
        # Abandon vs. modify, named as the thing that picks between them.
        assert "ABANDONING the deliverable entirely" in LOCAL_TOOLS_GUIDANCE
        assert "MODIFYING work that should still continue" in LOCAL_TOOLS_GUIDANCE
        assert "never leave a correction unsent because you replied to it" in \
            LOCAL_TOOLS_GUIDANCE
        # The honesty rule: accepted is not applied, and a later cancel wins.
        assert '"passed along", not applied' in LOCAL_TOOLS_GUIDANCE
        assert "a later cancel supersedes it" in LOCAL_TOOLS_GUIDANCE

    @pytest.mark.critical
    def test_critical_prompts_structure(self):
        """Critical: the pieces production behavior depends on"""
        # Slack formatting essentials
        assert "code blocks" in SLACK_SYSTEM_PROMPT.lower()
        # Username-prefix convention + never-echo rule
        assert 'prefixed "Username: "' in SLACK_SYSTEM_PROMPT
        assert "never copy the format" in SLACK_SYSTEM_PROMPT.lower()
        # Edit prompt distinguishes both edit types
        assert "photo edit only" in IMAGE_EDIT_SYSTEM_PROMPT
        assert "Style transformation" in IMAGE_EDIT_SYSTEM_PROMPT
