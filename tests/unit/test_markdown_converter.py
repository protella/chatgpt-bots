"""
Unit tests for markdown_converter.py module
"""
import pytest
from markdown_converter import MarkdownConverter


class TestMarkdownConverter:
    """Test MarkdownConverter class"""
    
    def test_initialization_slack(self):
        """Test initialization with Slack platform"""
        converter = MarkdownConverter("slack")
        assert converter.platform == "slack"
    
    def test_initialization_discord(self):
        """Test initialization with Discord platform"""
        converter = MarkdownConverter("discord")
        assert converter.platform == "discord"
    
    def test_initialization_case_insensitive(self):
        """Test platform name is case insensitive"""
        converter = MarkdownConverter("SLACK")
        assert converter.platform == "slack"
    
    def test_convert_empty_text(self):
        """Test converting empty text"""
        converter = MarkdownConverter("slack")
        assert converter.convert("") == ""
        assert converter.convert(None) == ""
    
    def test_convert_unknown_platform(self):
        """Test conversion for unknown platform returns original"""
        converter = MarkdownConverter("unknown")
        text = "**bold** *italic*"
        assert converter.convert(text) == text


class TestSlackConversion:
    """Test Slack-specific markdown conversion"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")
    
    def test_convert_headers(self):
        """Test header conversion to bold"""
        text = "# Header 1\n## Header 2\n### Header 3"
        result = self.converter.convert(text)
        assert "*Header 1*" in result
        assert "*Header 2*" in result
        assert "*Header 3*" in result
    
    def test_convert_bold(self):
        """Test bold text conversion"""
        text = "This is **bold** and __also bold__"
        result = self.converter.convert(text)
        assert "This is *bold* and *also bold*" in result
    
    def test_convert_italic(self):
        """Test italic text conversion"""
        text = "This is *italic* text"
        result = self.converter.convert(text)
        assert "This is _italic_ text" in result
    
    def test_convert_strikethrough(self):
        """Test strikethrough conversion"""
        text = "This is ~~strikethrough~~ text"
        result = self.converter.convert(text)
        assert "This is ~strikethrough~ text" in result
    
    def test_convert_links(self):
        """Test link conversion"""
        text = "[Click here](https://example.com)"
        result = self.converter.convert(text)
        assert "<https://example.com|Click here>" in result
    
    def test_convert_bare_urls(self):
        """Test bare URL wrapping"""
        text = "Visit https://example.com for more"
        result = self.converter.convert(text)
        assert "Visit <https://example.com> for more" in result
    
    def test_preserve_slack_formatted_urls(self):
        """Test that already formatted Slack URLs are preserved"""
        text = "Visit <https://example.com> for more"
        result = self.converter.convert(text)
        assert "Visit <https://example.com> for more" in result
    
    def test_convert_unordered_lists(self):
        """Test unordered list conversion"""
        text = "- Item 1\n- Item 2\n* Item 3\n+ Item 4"
        result = self.converter.convert(text)
        assert "• Item 1" in result
        assert "• Item 2" in result
        assert "• Item 3" in result
        assert "• Item 4" in result
    
    def test_convert_ordered_lists(self):
        """Test ordered lists remain unchanged"""
        text = "1. First\n2. Second\n3. Third"
        result = self.converter.convert(text)
        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result
    
    def test_convert_blockquotes(self):
        """Test blockquote conversion"""
        text = "> This is a quote\n> Multi-line quote"
        result = self.converter.convert(text)
        assert "> This is a quote" in result
        assert "> Multi-line quote" in result
    
    def test_convert_horizontal_rules(self):
        """Test horizontal rule conversion"""
        text = "---\nContent\n***\nMore content\n___"
        result = self.converter.convert(text)
        assert "———————————" in result
    
    def test_preserve_code_blocks(self):
        """Test that code blocks are preserved"""
        text = "```python\ndef hello():\n    print('world')\n```"
        result = self.converter.convert(text)
        assert "def hello():" in result
        assert "print('world')" in result
    
    def test_preserve_inline_code(self):
        """Test that inline code is preserved"""
        text = "Use `print()` function"
        result = self.converter.convert(text)
        assert "`print()`" in result
    
    def test_mixed_formatting(self):
        """Test mixed formatting elements"""
        text = "# Title\n\nThis is **bold** and *italic* with `code`"
        result = self.converter.convert(text)
        assert "*Title*" in result
        assert "*bold*" in result
        assert "_italic_" in result
        assert "`code`" in result
    
    def test_clean_whitespace(self):
        """Test whitespace cleaning"""
        text = "Line 1\n\n\n\nLine 2   \nLine 3"
        result = self.converter.convert(text)
        # Should reduce multiple newlines to max 2
        assert "\n\n\n\n" not in result
        # Should remove trailing spaces
        assert "Line 2   " not in result


class TestDiscordConversion:
    """Test Discord-specific markdown conversion"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("discord")
    
    def test_discord_preserves_standard_markdown(self):
        """Test that Discord preserves standard markdown"""
        text = "# Header\n**bold** *italic* ~~strike~~ `code`"
        result = self.converter.convert(text)
        # Discord supports standard markdown, so it should be mostly unchanged
        assert "# Header" in result
        assert "**bold**" in result
        assert "*italic*" in result
        assert "~~strike~~" in result
        assert "`code`" in result
    
    def test_discord_clean_whitespace(self):
        """Test that Discord conversion cleans whitespace"""
        text = "Line 1\n\n\n\nLine 2   "
        result = self.converter.convert(text)
        assert "\n\n\n\n" not in result
        assert result.endswith("Line 2")


class TestCodeBlockHandling:
    """Test code block extraction and restoration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")
    
    def test_extract_and_restore_fenced_code(self):
        """Test fenced code block extraction and restoration"""
        storage = []
        text = "Before\n```python\ncode here\n```\nAfter"
        
        # Test extraction
        extracted = self.converter._extract_code_blocks(text, storage)
        assert "###CODE_BLOCK_0###" in extracted
        assert len(storage) == 1
        assert "code here" in storage[0]
        
        # Test restoration
        restored = self.converter._restore_code_blocks_slack(extracted, storage)
        assert "code here" in restored
    
    def test_extract_and_restore_inline_code(self):
        """Test inline code extraction and restoration"""
        storage = []
        text = "Use `print()` function"
        
        # Test extraction
        extracted = self.converter._extract_code_blocks(text, storage)
        assert "###CODE_INLINE_0###" in extracted
        assert len(storage) == 1
        assert storage[0] == "`print()`"
        
        # Test restoration
        restored = self.converter._restore_code_blocks_slack(extracted, storage)
        assert "`print()`" in restored
    
    def test_multiple_code_blocks(self):
        """Test handling multiple code blocks"""
        text = "```python\ncode1\n```\nMiddle `inline` text\n```js\ncode2\n```"
        result = self.converter.convert(text)
        
        assert "code1" in result
        assert "code2" in result
        assert "`inline`" in result