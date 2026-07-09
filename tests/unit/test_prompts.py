"""Unit tests for prompts.py"""

import pytest
from prompts import (
    SLACK_SYSTEM_PROMPT,
    CLI_SYSTEM_PROMPT,
    IMAGE_INTENT_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_PROMPT,
    VISION_ENHANCEMENT_PROMPT,
    IMAGE_EDIT_SYSTEM_PROMPT,
    IMAGE_GEN_SYSTEM_PROMPT
)


class TestPrompts:
    """Test that all prompts are properly defined"""
    
    def test_slack_system_prompt_defined(self):
        """Test that SLACK_SYSTEM_PROMPT is defined and non-empty"""
        assert SLACK_SYSTEM_PROMPT is not None
        assert isinstance(SLACK_SYSTEM_PROMPT, str)
        assert len(SLACK_SYSTEM_PROMPT) > 0
        # Check for key content
        assert "Slack" in SLACK_SYSTEM_PROMPT
        # Check for Slack-specific formatting
        assert "*bold*" in SLACK_SYSTEM_PROMPT
    
    def test_cli_system_prompt_defined(self):
        """Test that CLI_SYSTEM_PROMPT is defined and non-empty"""
        assert CLI_SYSTEM_PROMPT is not None
        assert isinstance(CLI_SYSTEM_PROMPT, str)
        assert len(CLI_SYSTEM_PROMPT) > 0
        # Check for key content
        assert "helpful assistant" in CLI_SYSTEM_PROMPT.lower()
    
    def test_image_intent_system_prompt_defined(self):
        """Test that IMAGE_INTENT_SYSTEM_PROMPT is defined and non-empty"""
        assert IMAGE_INTENT_SYSTEM_PROMPT is not None
        assert isinstance(IMAGE_INTENT_SYSTEM_PROMPT, str)
        assert len(IMAGE_INTENT_SYSTEM_PROMPT) > 0
        # Check for key classification categories
        assert '"new"' in IMAGE_INTENT_SYSTEM_PROMPT
        assert '"edit"' in IMAGE_INTENT_SYSTEM_PROMPT
        assert '"vision"' in IMAGE_INTENT_SYSTEM_PROMPT
        assert '"ambiguous"' in IMAGE_INTENT_SYSTEM_PROMPT
        assert '"none"' in IMAGE_INTENT_SYSTEM_PROMPT
    
    def test_image_analysis_prompt_defined(self):
        """Test that IMAGE_ANALYSIS_PROMPT is defined and non-empty"""
        assert IMAGE_ANALYSIS_PROMPT is not None
        assert isinstance(IMAGE_ANALYSIS_PROMPT, str)
        assert len(IMAGE_ANALYSIS_PROMPT) > 0
        # Check for key content
        assert "image" in IMAGE_ANALYSIS_PROMPT.lower()
        assert "concise" in IMAGE_ANALYSIS_PROMPT.lower()
    
    def test_vision_enhancement_prompt_defined(self):
        """Test that VISION_ENHANCEMENT_PROMPT is defined and non-empty"""
        assert VISION_ENHANCEMENT_PROMPT is not None
        assert isinstance(VISION_ENHANCEMENT_PROMPT, str)
        assert len(VISION_ENHANCEMENT_PROMPT) > 0
        # Check for key content
        assert "enhance" in VISION_ENHANCEMENT_PROMPT.lower()
        assert "conversational" in VISION_ENHANCEMENT_PROMPT.lower()
    
    def test_image_edit_system_prompt_defined(self):
        """Test that IMAGE_EDIT_SYSTEM_PROMPT is defined and non-empty"""
        assert IMAGE_EDIT_SYSTEM_PROMPT is not None
        assert isinstance(IMAGE_EDIT_SYSTEM_PROMPT, str)
        assert len(IMAGE_EDIT_SYSTEM_PROMPT) > 0
        # Check for key content
        assert "edit" in IMAGE_EDIT_SYSTEM_PROMPT.lower()
        assert "STYLE TRANSFORMATION" in IMAGE_EDIT_SYSTEM_PROMPT
        assert "MINOR EDIT" in IMAGE_EDIT_SYSTEM_PROMPT
    
    def test_image_gen_system_prompt_defined(self):
        """Test that IMAGE_GEN_SYSTEM_PROMPT is defined and non-empty"""
        assert IMAGE_GEN_SYSTEM_PROMPT is not None
        assert isinstance(IMAGE_GEN_SYSTEM_PROMPT, str)
        assert len(IMAGE_GEN_SYSTEM_PROMPT) > 0
        # Check for key content
        assert "image generation" in IMAGE_GEN_SYSTEM_PROMPT.lower()
        assert "prompt" in IMAGE_GEN_SYSTEM_PROMPT.lower()
    
    def test_all_prompts_are_strings(self):
        """Test that all prompts are string type"""
        prompts = [
            SLACK_SYSTEM_PROMPT,
                    CLI_SYSTEM_PROMPT,
            IMAGE_INTENT_SYSTEM_PROMPT,
            IMAGE_ANALYSIS_PROMPT,
            VISION_ENHANCEMENT_PROMPT,
            IMAGE_EDIT_SYSTEM_PROMPT,
            IMAGE_GEN_SYSTEM_PROMPT
        ]
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0
    
    def test_prompts_contain_no_template_variables(self):
        """Test that prompts don't contain unresolved template variables"""
        prompts = [
            SLACK_SYSTEM_PROMPT,
                    CLI_SYSTEM_PROMPT,
            IMAGE_INTENT_SYSTEM_PROMPT,
            IMAGE_ANALYSIS_PROMPT,
            VISION_ENHANCEMENT_PROMPT,
            IMAGE_EDIT_SYSTEM_PROMPT,
            IMAGE_GEN_SYSTEM_PROMPT
        ]
        
        for prompt in prompts:
            # Check for common template variable patterns
            assert "{" not in prompt or "}" not in prompt  # No f-string style
            assert "{{" not in prompt  # No jinja2 style
            assert "${" not in prompt  # No bash/JS style
    
    @pytest.mark.critical
    def test_critical_prompts_structure(self):
        """Critical test for prompt structure and key instructions"""
        # Test Slack prompt has formatting instructions
        assert "*bold*" in SLACK_SYSTEM_PROMPT
        assert "```" in SLACK_SYSTEM_PROMPT  # Code blocks
        
        # Test image intent has all categories
        intent_categories = ["new", "edit", "vision", "ambiguous", "none"]
        for category in intent_categories:
            assert category in IMAGE_INTENT_SYSTEM_PROMPT
        
        # Test image edit has both transformation types
        assert "STYLE TRANSFORMATION" in IMAGE_EDIT_SYSTEM_PROMPT
        assert "MINOR EDIT" in IMAGE_EDIT_SYSTEM_PROMPT
