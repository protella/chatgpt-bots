"""Unit tests for prompts.py"""

import pytest
from prompts import (
    SLACK_SYSTEM_PROMPT,
    DISCORD_SYSTEM_PROMPT,
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
        # Company name can be customized via environment variables
        assert "mrkdwn" in SLACK_SYSTEM_PROMPT.lower()
    
    def test_discord_system_prompt_defined(self):
        """Test that DISCORD_SYSTEM_PROMPT is defined and non-empty"""
        assert DISCORD_SYSTEM_PROMPT is not None
        assert isinstance(DISCORD_SYSTEM_PROMPT, str)
        assert len(DISCORD_SYSTEM_PROMPT) > 0
        # Check for key content
        assert "Discord" in DISCORD_SYSTEM_PROMPT
        assert "gaming" in DISCORD_SYSTEM_PROMPT.lower()
        assert "sarcastic" in DISCORD_SYSTEM_PROMPT.lower()
    
    def test_cli_system_prompt_defined(self):
        """Test that CLI_SYSTEM_PROMPT is defined and non-empty"""
        assert CLI_SYSTEM_PROMPT is not None
        assert isinstance(CLI_SYSTEM_PROMPT, str)
        assert len(CLI_SYSTEM_PROMPT) > 0
        # Check for key content
        assert "helpful assistant" in CLI_SYSTEM_PROMPT.lower()
        assert "GPT-5" in CLI_SYSTEM_PROMPT
    
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
            DISCORD_SYSTEM_PROMPT,
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
            DISCORD_SYSTEM_PROMPT,
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
        assert "_italic_" in SLACK_SYSTEM_PROMPT
        
        # Test image intent has all categories
        intent_categories = ["new", "edit", "vision", "ambiguous", "none"]
        for category in intent_categories:
            assert category in IMAGE_INTENT_SYSTEM_PROMPT
        
        # Test image edit has both transformation types
        assert "STYLE TRANSFORMATION" in IMAGE_EDIT_SYSTEM_PROMPT
        assert "MINOR EDIT" in IMAGE_EDIT_SYSTEM_PROMPT
    
    @pytest.mark.smoke
    def test_smoke_prompts_loaded(self):
        """Smoke test that all prompts can be imported"""
        assert SLACK_SYSTEM_PROMPT
        assert DISCORD_SYSTEM_PROMPT
        assert CLI_SYSTEM_PROMPT
        assert IMAGE_INTENT_SYSTEM_PROMPT
        assert IMAGE_ANALYSIS_PROMPT
        assert VISION_ENHANCEMENT_PROMPT
        assert IMAGE_EDIT_SYSTEM_PROMPT
        assert IMAGE_GEN_SYSTEM_PROMPT
    
    def test_prompts_consistency(self):
        """Test that prompts have consistent information"""
        # All system prompts should mention GPT-5
        system_prompts = [SLACK_SYSTEM_PROMPT, DISCORD_SYSTEM_PROMPT, CLI_SYSTEM_PROMPT]
        for prompt in system_prompts:
            assert "GPT-5" in prompt or "GPT-4" in prompt  # Some might still use GPT-4
        
        # Image prompts should mention "image"
        image_prompts = [
            IMAGE_INTENT_SYSTEM_PROMPT,
            IMAGE_ANALYSIS_PROMPT,
            VISION_ENHANCEMENT_PROMPT,
            IMAGE_EDIT_SYSTEM_PROMPT,
            IMAGE_GEN_SYSTEM_PROMPT
        ]
        for prompt in image_prompts:
            assert "image" in prompt.lower()
    
    def test_prompts_no_trailing_whitespace(self):
        """Test that prompts don't have trailing whitespace"""
        prompts = {
            "SLACK_SYSTEM_PROMPT": SLACK_SYSTEM_PROMPT,
            "DISCORD_SYSTEM_PROMPT": DISCORD_SYSTEM_PROMPT,
            "CLI_SYSTEM_PROMPT": CLI_SYSTEM_PROMPT,
            "IMAGE_INTENT_SYSTEM_PROMPT": IMAGE_INTENT_SYSTEM_PROMPT,
            "IMAGE_ANALYSIS_PROMPT": IMAGE_ANALYSIS_PROMPT,
            "VISION_ENHANCEMENT_PROMPT": VISION_ENHANCEMENT_PROMPT,
            "IMAGE_EDIT_SYSTEM_PROMPT": IMAGE_EDIT_SYSTEM_PROMPT,
            "IMAGE_GEN_SYSTEM_PROMPT": IMAGE_GEN_SYSTEM_PROMPT
        }
        
        for name, prompt in prompts.items():
            assert prompt == prompt.rstrip(), f"{name} has trailing whitespace"
    
    def test_contract_prompts_interface(self):
        """Test that prompts module maintains expected interface"""
        import prompts
        
        # Required constants must exist
        required_prompts = [
            'SLACK_SYSTEM_PROMPT',
            'DISCORD_SYSTEM_PROMPT',
            'CLI_SYSTEM_PROMPT',
            'IMAGE_INTENT_SYSTEM_PROMPT',
            'IMAGE_ANALYSIS_PROMPT',
            'VISION_ENHANCEMENT_PROMPT',
            'IMAGE_EDIT_SYSTEM_PROMPT',
            'IMAGE_GEN_SYSTEM_PROMPT'
        ]
        
        for prompt_name in required_prompts:
            assert hasattr(prompts, prompt_name), f"Missing required prompt: {prompt_name}"
            prompt_value = getattr(prompts, prompt_name)
            assert isinstance(prompt_value, str), f"{prompt_name} must be a string"
            assert len(prompt_value) > 0, f"{prompt_name} must not be empty"