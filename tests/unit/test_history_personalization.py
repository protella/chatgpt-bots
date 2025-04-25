import os
import sys
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.history import remove_personalization_tags


class TestPersonalizationTagRemoval:
    """Tests for removing personalization tags from message text."""

    def test_remove_personalization_tags(self):
        """Test that personalization tags are correctly removed from text."""
        # Test basic tag removal
        input_text = "[username=John] Hello, how are you?"
        expected = "Hello, how are you?"
        assert remove_personalization_tags(input_text) == expected

        # Test with multiple spaces
        input_text = "[username=Jane]   What's the weather like today?"
        expected = "What's the weather like today?"
        assert remove_personalization_tags(input_text) == expected

        # Test with no personalization tag
        input_text = "Just a normal message without a tag"
        expected = "Just a normal message without a tag"
        assert remove_personalization_tags(input_text) == expected

        # Test with tag in the middle (shouldn't happen, but testing edge case)
        input_text = "This is a [username=Bob] strange message"
        expected = "This is a strange message"
        assert remove_personalization_tags(input_text) == expected

        # Test with name containing special characters
        input_text = "[username=O'Reilly-Smith] How does this work?"
        expected = "How does this work?"
        assert remove_personalization_tags(input_text) == expected

    def test_remove_personalization_tags_multiple(self):
        """Test removing multiple personalization tags (edge case)."""
        input_text = "[username=Multiple] [username=Tags] This shouldn't happen"
        expected = "This shouldn't happen"
        assert remove_personalization_tags(input_text) == expected 