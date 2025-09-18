"""
Comprehensive unit tests for markdown_converter.py module
Tests for improved coverage of markdown conversion functionality
"""
import pytest
from markdown_converter import MarkdownConverter


class TestMarkdownConverterComprehensive:
    """Comprehensive tests for MarkdownConverter for better coverage"""

    def test_initialization_default_platform(self):
        """Test initialization with default platform"""
        converter = MarkdownConverter()
        assert converter.platform == "slack"

    def test_initialization_case_insensitive(self):
        """Test platform name is case insensitive"""
        converter = MarkdownConverter("SLACK")
        assert converter.platform == "slack"

    def test_convert_none_input(self):
        """Test converting None input"""
        converter = MarkdownConverter("slack")
        result = converter.convert(None)
        assert result == ""

    def test_convert_empty_string(self):
        """Test converting empty string"""
        converter = MarkdownConverter("slack")
        result = converter.convert("")
        assert result == ""

    def test_convert_unknown_platform_returns_original(self):
        """Test conversion for unknown platform returns original text"""
        converter = MarkdownConverter("unknown_platform")
        text = "**bold** *italic* text"
        result = converter.convert(text)
        assert result == text


class TestSlackConversionComprehensive:
    """Comprehensive tests for Slack-specific markdown conversion"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")

    def test_convert_headers_all_levels(self):
        """Test all header levels conversion to bold"""
        text = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        result = self.converter.convert(text)

        # All headers should become bold
        assert "*H1*" in result
        assert "*H2*" in result
        assert "*H3*" in result
        assert "*H4*" in result
        assert "*H5*" in result
        assert "*H6*" in result

    def test_convert_bold_double_asterisk(self):
        """Test **bold** conversion"""
        text = "This is **bold** text"
        result = self.converter.convert(text)
        assert "*bold*" in result
        assert "**bold**" not in result

    def test_convert_bold_double_underscore(self):
        """Test __bold__ conversion"""
        text = "This is __bold__ text"
        result = self.converter.convert(text)
        assert "*bold*" in result
        assert "__bold__" not in result

    def test_convert_italic_single_asterisk(self):
        """Test *italic* conversion"""
        text = "This is *italic* text"
        result = self.converter.convert(text)
        assert "_italic_" in result

    def test_convert_italic_single_underscore(self):
        """Test _italic_ conversion"""
        text = "This is _italic_ text"
        result = self.converter.convert(text)
        assert "_italic_" in result

    def test_convert_strikethrough(self):
        """Test ~~strikethrough~~ conversion"""
        text = "This is ~~strikethrough~~ text"
        result = self.converter.convert(text)
        assert "~strikethrough~" in result
        assert "~~strikethrough~~" not in result

    def test_convert_links_markdown_format(self):
        """Test [text](url) to <url|text> conversion"""
        text = "[Google](https://google.com)"
        result = self.converter.convert(text)
        assert "<https://google.com|Google>" in result
        assert "[Google](https://google.com)" not in result

    def test_convert_bare_urls(self):
        """Test bare URL to <url> conversion"""
        text = "Visit https://example.com for more info"
        result = self.converter.convert(text)
        assert "<https://example.com>" in result

    def test_convert_bare_urls_no_double_wrap(self):
        """Test that already wrapped URLs aren't double-wrapped"""
        text = "Already wrapped <https://example.com> URL"
        result = self.converter.convert(text)
        assert "<<https://example.com>>" not in result
        assert "<https://example.com>" in result

    def test_convert_unordered_lists_dash(self):
        """Test - list item conversion"""
        text = "- Item 1\n- Item 2"
        result = self.converter.convert(text)
        assert "• Item 1" in result
        assert "• Item 2" in result

    def test_convert_unordered_lists_asterisk(self):
        """Test * list item conversion"""
        text = "* Item 1\n* Item 2"
        result = self.converter.convert(text)
        assert "• Item 1" in result
        assert "• Item 2" in result

    def test_convert_unordered_lists_plus(self):
        """Test + list item conversion"""
        text = "+ Item 1\n+ Item 2"
        result = self.converter.convert(text)
        assert "• Item 1" in result
        assert "• Item 2" in result

    def test_convert_ordered_lists_preserved(self):
        """Test that ordered lists are preserved"""
        text = "1. First item\n2. Second item\n3. Third item"
        result = self.converter.convert(text)
        # Ordered lists should remain unchanged
        assert "1. First item" in result
        assert "2. Second item" in result
        assert "3. Third item" in result

    def test_convert_blockquotes_single_line(self):
        """Test single line blockquote conversion"""
        text = "> This is a quote"
        result = self.converter.convert(text)
        assert "> This is a quote" in result

    def test_convert_blockquotes_multiple_lines(self):
        """Test multiple line blockquote conversion"""
        text = "> First line\n>Second line\n> Third line"
        result = self.converter.convert(text)
        assert "> First line" in result
        assert "> Second line" in result
        assert "> Third line" in result

    def test_convert_horizontal_rules_dashes(self):
        """Test --- horizontal rule conversion"""
        text = "Text\n---\nMore text"
        result = self.converter.convert(text)
        assert "———————————" in result

    def test_convert_horizontal_rules_asterisks(self):
        """Test *** horizontal rule conversion"""
        text = "Text\n***\nMore text"
        result = self.converter.convert(text)
        assert "———————————" in result

    def test_convert_horizontal_rules_underscores(self):
        """Test ___ horizontal rule conversion"""
        text = "Text\n___\nMore text"
        result = self.converter.convert(text)
        assert "———————————" in result

    def test_whitespace_cleanup_multiple_newlines(self):
        """Test cleanup of multiple consecutive newlines"""
        text = "Line 1\n\n\n\n\nLine 2"
        result = self.converter.convert(text)
        assert "\n\n\n" not in result
        assert "Line 1\n\nLine 2" in result

    def test_whitespace_cleanup_trailing_spaces(self):
        """Test removal of trailing spaces"""
        text = "Line with spaces   \nAnother line   "
        result = self.converter.convert(text)
        assert not any(line.endswith(' ') for line in result.split('\n'))

    def test_whitespace_cleanup_strip_text(self):
        """Test stripping leading/trailing whitespace"""
        text = "   Some text   "
        result = self.converter.convert(text)
        assert result == "Some text"

    def test_complex_markdown_conversion(self):
        """Test complex markdown with multiple elements"""
        text = """# Main Header

This is **bold** and this is *italic*.

- List item with `code`
- Another item

[Link to Google](https://google.com)

> This is a blockquote

```python
def hello():
    print("world")
```"""

        result = self.converter.convert(text)

        # Check various conversions
        assert "*Main Header*" in result
        assert "*bold*" in result
        assert "_italic_" in result
        assert "• List item" in result
        assert "<https://google.com|Link to Google>" in result
        assert "> This is a blockquote" in result
        assert "```python" in result
        assert "`code`" in result

    def test_nested_formatting(self):
        """Test nested formatting scenarios"""
        text = "# Header with **bold** text"
        result = self.converter.convert(text)
        # The conversion order may affect this - test that content is preserved
        assert "Header" in result
        assert "bold" in result
        assert "text" in result

    def test_italic_bold_interaction(self):
        """Test interaction between italic and bold conversion"""
        text = "***bold and italic***"
        result = self.converter.convert(text)
        # Should handle complex formatting without leaving original syntax
        assert "***" not in result


class TestDiscordConversion:
    """Test Discord-specific markdown conversion"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("discord")

    def test_discord_preserves_standard_markdown(self):
        """Test that Discord preserves standard markdown"""
        text = "**bold** *italic* ~~strikethrough~~"
        result = self.converter.convert(text)
        # Discord supports standard markdown, so should be preserved
        assert "**bold**" in result
        assert "*italic*" in result
        assert "~~strikethrough~~" in result

    def test_discord_headers_preserved(self):
        """Test that Discord headers are preserved"""
        text = "# Header 1\n## Header 2"
        result = self.converter.convert(text)
        assert "# Header 1" in result
        assert "## Header 2" in result

    def test_discord_code_blocks_preserved(self):
        """Test that Discord code blocks are preserved"""
        text = "```python\nprint('hello')\n```"
        result = self.converter.convert(text)
        assert "```python" in result
        assert "print('hello')" in result

    def test_discord_links_preserved(self):
        """Test that Discord links are preserved"""
        text = "[Google](https://google.com)"
        result = self.converter.convert(text)
        assert "[Google](https://google.com)" in result

    def test_discord_whitespace_cleanup(self):
        """Test Discord still cleans up whitespace"""
        text = "Line 1\n\n\n\nLine 2   "
        result = self.converter.convert(text)
        # Should still clean up excess whitespace
        assert result.count('\n\n\n') == 0
        assert not result.endswith(' ')


class TestCodeBlockExtraction:
    """Test code block extraction and restoration methods"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")

    def test_extract_fenced_code_blocks(self):
        """Test extraction of fenced code blocks"""
        storage = []
        text = "Some text ```code here``` more text"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_BLOCK_0###" in result
        assert len(storage) == 1
        assert storage[0] == "```code here```"

    def test_extract_inline_code(self):
        """Test extraction of inline code"""
        storage = []
        text = "Some text `inline code` more text"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_INLINE_0###" in result
        assert len(storage) == 1
        assert storage[0] == "`inline code`"

    def test_extract_mixed_code_types(self):
        """Test extraction of mixed code types"""
        storage = []
        text = "Text ```block``` and `inline` code"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_BLOCK_0###" in result
        assert "###CODE_INLINE_1###" in result
        assert len(storage) == 2

    def test_extract_multiple_code_blocks(self):
        """Test extraction of multiple code blocks"""
        storage = []
        text = "```first``` and ```second``` blocks"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_BLOCK_0###" in result
        assert "###CODE_BLOCK_1###" in result
        assert len(storage) == 2

    def test_restore_code_blocks_with_language(self):
        """Test restoration of code blocks with language specification"""
        storage = ["```python\nprint('hello')\n```"]
        text = "Text ###CODE_BLOCK_0### here"

        result = self.converter._restore_code_blocks_slack(text, storage)

        assert "```python" not in result  # Language hint is removed for Slack
        assert "```print('hello')" in result
        assert "###CODE_BLOCK_0###" not in result

    def test_restore_inline_code(self):
        """Test restoration of inline code"""
        storage = ["`variable`"]
        text = "Use ###CODE_INLINE_0### here"

        result = self.converter._restore_code_blocks_slack(text, storage)

        assert "`variable`" in result
        assert "###CODE_INLINE_0###" not in result

    def test_restore_mixed_code_types(self):
        """Test restoration of mixed code types"""
        storage = ["```python\ncode\n```", "`inline`"]
        text = "Block ###CODE_BLOCK_0### and ###CODE_INLINE_1### code"

        result = self.converter._restore_code_blocks_slack(text, storage)

        assert "```code```" in result
        assert "`inline`" in result
        assert "###CODE_BLOCK_0###" not in result
        assert "###CODE_INLINE_1###" not in result

    def test_code_block_preservation_during_conversion(self):
        """Test that code blocks are preserved during full conversion"""
        text = "Text with ```def function(**kwargs):\n    return *args``` code"
        result = self.converter.convert(text)

        # Code content should be preserved exactly
        assert "def function(**kwargs):" in result
        assert "return *args" in result
        assert "```" in result

    def test_inline_code_preservation_during_conversion(self):
        """Test that inline code is preserved during full conversion"""
        text = "Use `**not bold**` in code"
        result = self.converter.convert(text)

        # Inline code should not be processed for markdown
        assert "`**not bold**`" in result
        assert "*not bold*" not in result


class TestUtilityMethods:
    """Test utility methods in MarkdownConverter"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")

    def test_clean_whitespace_comprehensive(self):
        """Test comprehensive whitespace cleaning"""
        text = """


Line 1


Line 2

        """
        result = self.converter._clean_whitespace(text)

        # Should remove multiple newlines and trim
        assert result.count('\n\n\n') == 0
        assert not result.startswith(' ')
        assert not result.endswith(' ')
        assert "Line 1" in result
        assert "Line 2" in result

    def test_convert_headers_slack_specific(self):
        """Test header conversion specific to Slack"""
        text = "### Important Header"
        result = self.converter._convert_headers_slack(text)
        assert "*Important Header*" == result.strip()

    def test_convert_bold_slack_specific(self):
        """Test bold conversion specific to Slack"""
        text = "**bold** and __also bold__"
        result = self.converter._convert_bold_slack(text)
        assert "*bold* and *also bold*" == result

    def test_convert_italic_slack_specific(self):
        """Test italic conversion specific to Slack"""
        text = "*italic* and _also italic_"
        result = self.converter._convert_italic_slack(text)
        assert "_italic_ and _also italic_" == result

    def test_convert_strikethrough_slack_specific(self):
        """Test strikethrough conversion specific to Slack"""
        text = "~~strikethrough~~ text"
        result = self.converter._convert_strikethrough_slack(text)
        assert "~strikethrough~ text" == result

    def test_convert_links_slack_specific(self):
        """Test link conversion specific to Slack"""
        text = "[Google](https://google.com) and https://example.com"
        result = self.converter._convert_links_slack(text)
        assert "<https://google.com|Google>" in result
        assert "<https://example.com>" in result

    def test_convert_lists_slack_with_indentation(self):
        """Test list conversion with indentation"""
        text = "  - Indented item\n    * Double indented"
        result = self.converter._convert_lists_slack(text)
        assert "  • Indented item" in result
        assert "    • Double indented" in result

    def test_convert_blockquotes_multiple_formats(self):
        """Test blockquote conversion with different formats"""
        text = ">No space\n> With space\n>  Extra spaces"
        result = self.converter._convert_blockquotes(text)
        assert "> No space" in result
        assert "> With space" in result
        assert "> Extra spaces" in result

    def test_convert_horizontal_rules_multiline(self):
        """Test horizontal rule conversion in multiline context"""
        text = "Before\n---\nAfter\n***\nEnd"
        result = self.converter._convert_horizontal_rules(text)
        lines = result.split('\n')
        assert "———————————" in lines
        assert lines.count("———————————") == 2


@pytest.mark.critical
class TestMarkdownConverterCritical:
    """Critical tests for markdown converter functionality"""

    def test_critical_platform_routing(self):
        """Critical test for platform-specific routing"""
        slack_converter = MarkdownConverter("slack")
        discord_converter = MarkdownConverter("discord")

        text = "**bold** text"

        slack_result = slack_converter.convert(text)
        discord_result = discord_converter.convert(text)

        # Results should be different for different platforms
        assert slack_result != discord_result
        assert "*bold*" in slack_result  # Slack format
        assert "**bold**" in discord_result  # Discord preserves standard

    def test_critical_code_preservation(self):
        """Critical test that code blocks are not corrupted during conversion"""
        converter = MarkdownConverter("slack")

        original_code = "```python\ndef function(**kwargs):\n    return *args\n```"
        result = converter.convert(original_code)

        # Code content should be preserved exactly
        assert "def function(**kwargs):" in result
        assert "return *args" in result
        assert "```" in result

    def test_critical_no_data_loss(self):
        """Critical test that no content is lost during conversion"""
        converter = MarkdownConverter("slack")

        text = "Important **data** with `code` and [links](http://example.com)"
        result = converter.convert(text)

        # All content words should be preserved
        assert "Important" in result
        assert "data" in result
        assert "code" in result
        assert "links" in result
        assert "example.com" in result

    def test_critical_conversion_order(self):
        """Critical test for conversion order to prevent conflicts"""
        converter = MarkdownConverter("slack")

        # Test that italic conversion happens before bold to avoid conflicts
        text = "*italic* and **bold** text"
        result = converter.convert(text)

        assert "_italic_" in result
        assert "*bold*" in result
        # Should not have any leftover markdown syntax
        assert "**" not in result


@pytest.mark.smoke
class TestMarkdownConverterSmoke:
    """Smoke tests for markdown converter"""

    def test_smoke_basic_functionality(self):
        """Smoke test for basic converter functionality"""
        converter = MarkdownConverter("slack")

        # Should not raise exceptions
        assert converter.convert("") == ""
        assert converter.convert("plain text") == "plain text"
        assert isinstance(converter.convert("**bold**"), str)

    def test_smoke_all_platforms(self):
        """Smoke test for all supported platforms"""
        for platform in ["slack", "discord", "unknown"]:
            converter = MarkdownConverter(platform)
            result = converter.convert("**test**")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_smoke_large_text(self):
        """Smoke test for large text conversion"""
        converter = MarkdownConverter("slack")

        # Generate large text with various markdown elements
        large_text = "\n".join([
            f"# Header {i}",
            f"This is **bold {i}** and *italic {i}*",
            f"- List item {i}",
            f"> Quote {i}",
            f"```code block {i}```"
        ] for i in range(100))

        # Should handle large text without issues
        result = converter.convert(large_text)
        assert isinstance(result, str)
        assert len(result) > len(large_text) * 0.8  # Shouldn't shrink too much


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")

    def test_empty_markdown_elements(self):
        """Test empty markdown elements"""
        text = "** ** __ __ ~~ ~~ `` ``"
        result = self.converter.convert(text)
        # Should handle empty elements gracefully
        assert isinstance(result, str)

    def test_malformed_markdown(self):
        """Test malformed markdown"""
        text = "**unclosed bold _mixed *formatting [incomplete link"
        result = self.converter.convert(text)
        # Should not crash on malformed input
        assert isinstance(result, str)

    def test_special_characters_in_code(self):
        """Test special characters preserved in code blocks"""
        text = "```\n**bold** *italic* [link](url)\n```"
        result = self.converter.convert(text)
        # Special characters in code should be preserved
        assert "**bold**" in result
        assert "*italic*" in result
        assert "[link](url)" in result

    def test_unicode_text(self):
        """Test unicode text handling"""
        text = "**粗体** *斜体* 中文测试"
        result = self.converter.convert(text)
        # Should handle unicode correctly
        assert "*粗体*" in result
        assert "_斜体_" in result
        assert "中文测试" in result

    def test_url_edge_cases(self):
        """Test URL conversion edge cases"""
        text = "Visit https://example.com/path?param=value#fragment for info"
        result = self.converter.convert(text)
        assert "<https://example.com/path?param=value#fragment>" in result