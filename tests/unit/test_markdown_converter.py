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

    def test_convert_headers_all_levels(self):
        """Test all header levels conversion"""
        text = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        result = self.converter.convert(text)

        # All headers should become bold
        assert "*H1*" in result
        assert "*H2*" in result
        assert "*H3*" in result
        assert "*H4*" in result
        assert "*H5*" in result
        assert "*H6*" in result

    def test_convert_bold_markdown(self):
        """Test bold markdown conversion"""
        # Test **bold**
        text = "This is **bold** text"
        result = self.converter.convert(text)
        assert "*bold*" in result
        assert "**bold**" not in result

        # Test __bold__
        text = "This is __bold__ text"
        result = self.converter.convert(text)
        assert "*bold*" in result
        assert "__bold__" not in result

    def test_convert_italic_markdown(self):
        """Test italic markdown conversion"""
        # Test *italic*
        text = "This is *italic* text"
        result = self.converter.convert(text)
        assert "_italic_" in result

        # Test _italic_
        text = "This is _italic_ text"
        result = self.converter.convert(text)
        assert "_italic_" in result

    def test_convert_strikethrough(self):
        """Test strikethrough conversion"""
        text = "This is ~~strikethrough~~ text"
        result = self.converter.convert(text)
        assert "~strikethrough~" in result
        assert "~~strikethrough~~" not in result

    def test_convert_links(self):
        """Test link conversion"""
        text = "[Google](https://google.com)"
        result = self.converter.convert(text)
        assert "<https://google.com|Google>" in result

    def test_convert_bare_urls(self):
        """Test bare URL conversion"""
        text = "Visit https://example.com for more info"
        result = self.converter.convert(text)
        assert "<https://example.com>" in result

    def test_convert_unordered_lists(self):
        """Test unordered list conversion"""
        text = "- Item 1\n- Item 2\n* Item 3\n+ Item 4"
        result = self.converter.convert(text)
        assert "• Item 1" in result
        assert "• Item 2" in result
        assert "• Item 3" in result
        assert "• Item 4" in result

    def test_convert_ordered_lists(self):
        """Test ordered list preservation"""
        text = "1. First item\n2. Second item\n3. Third item"
        result = self.converter.convert(text)
        # Ordered lists should remain unchanged
        assert "1. First item" in result
        assert "2. Second item" in result
        assert "3. Third item" in result

    def test_convert_blockquotes(self):
        """Test blockquote conversion"""
        text = "> This is a quote\n>Another line"
        result = self.converter.convert(text)
        assert "> This is a quote" in result
        assert "> Another line" in result

    def test_convert_horizontal_rules(self):
        """Test horizontal rule conversion"""
        text = "---\n***\n___"
        result = self.converter.convert(text)
        # Should convert to em dashes
        assert "———————————" in result

    def test_code_block_preservation(self):
        """Test that code blocks are preserved during conversion"""
        text = "```python\nprint('hello')\n```"
        result = self.converter.convert(text)
        # Code block should remain unchanged
        assert "```" in result
        assert "print('hello')" in result

    def test_inline_code_preservation(self):
        """Test that inline code is preserved"""
        text = "Use `print()` function"
        result = self.converter.convert(text)
        assert "`print()`" in result

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

    def test_whitespace_cleanup(self):
        """Test whitespace cleanup"""
        text = "Line 1\n\n\n\nLine 2   \n   Line 3   "
        result = self.converter.convert(text)
        # Should reduce multiple newlines and trim whitespace
        assert result.count('\n\n\n') == 0
        assert not result.endswith(' ')

    def test_nested_formatting(self):
        """Test nested formatting scenarios"""
        # Bold inside headers
        text = "# Header with **bold** text"
        result = self.converter.convert(text)
        assert "*Header with *bold* text*" in result

    def test_edge_case_empty_elements(self):
        """Test edge cases with empty elements"""
        text = "**empty bold****another**"
        result = self.converter.convert(text)
        # Should handle empty bold elements
        assert "**empty bold**" not in result

    def test_code_block_with_language(self):
        """Test code block with language specification"""
        text = "```javascript\nconst x = 5;\n```"
        result = self.converter.convert(text)
        # Should preserve code block content
        assert "const x = 5;" in result
        assert "```" in result

    def test_mixed_list_types(self):
        """Test mixed list types"""
        text = """1. Ordered item
- Unordered item
2. Another ordered
* Another unordered"""

        result = self.converter.convert(text)
        assert "1. Ordered item" in result
        assert "• Unordered item" in result
        assert "2. Another ordered" in result
        assert "• Another unordered" in result

    def test_link_edge_cases(self):
        """Test link conversion edge cases"""
        # Link with existing angle brackets shouldn't be double-wrapped
        text = "Already formatted: <https://example.com>"
        result = self.converter.convert(text)
        # Should not become <<https://example.com>>
        assert "<<https://example.com>>" not in result
        assert "<https://example.com>" in result

    def test_italic_bold_interaction(self):
        """Test interaction between italic and bold conversion"""
        text = "***bold and italic***"
        result = self.converter.convert(text)
        # Should handle complex formatting
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
        """Test Discord whitespace cleanup"""
        text = "Line 1\n\n\n\nLine 2   "
        result = self.converter.convert(text)
        # Should still clean up excess whitespace
        assert result.count('\n\n\n') == 0
        assert not result.endswith(' ')


class TestCodeBlockExtraction:
    """Test code block extraction and restoration"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")

    def test_fenced_code_block_extraction(self):
        """Test extraction of fenced code blocks"""
        storage = []
        text = "Some text ```code here``` more text"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_BLOCK_0###" in result
        assert len(storage) == 1
        assert storage[0] == "```code here```"

    def test_inline_code_extraction(self):
        """Test extraction of inline code"""
        storage = []
        text = "Some text `inline code` more text"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_INLINE_0###" in result
        assert len(storage) == 1
        assert storage[0] == "`inline code`"

    def test_mixed_code_extraction(self):
        """Test extraction of mixed code types"""
        storage = []
        text = "Text ```block``` and `inline` code"
        result = self.converter._extract_code_blocks(text, storage)

        assert "###CODE_BLOCK_0###" in result
        assert "###CODE_INLINE_1###" in result
        assert len(storage) == 2

    def test_code_block_restoration(self):
        """Test restoration of code blocks"""
        storage = ["```python\nprint('hello')\n```", "`variable`"]
        text = "Text ###CODE_BLOCK_0### and ###CODE_INLINE_1### here"

        result = self.converter._restore_code_blocks_slack(text, storage)

        assert "```python" in result
        assert "print('hello')" in result
        assert "`variable`" in result
        assert "###CODE_BLOCK_0###" not in result
        assert "###CODE_INLINE_1###" not in result


class TestUtilityMethods:
    """Test utility methods in MarkdownConverter"""

    def setup_method(self):
        """Setup test fixtures"""
        self.converter = MarkdownConverter("slack")

    def test_clean_whitespace_multiple_newlines(self):
        """Test cleaning multiple consecutive newlines"""
        text = "Line 1\n\n\n\n\nLine 2"
        result = self.converter._clean_whitespace(text)
        assert "\n\n\n" not in result
        assert "Line 1\n\nLine 2" == result

    def test_clean_whitespace_trailing_spaces(self):
        """Test removing trailing spaces"""
        text = "Line with spaces   \nAnother line   "
        result = self.converter._clean_whitespace(text)
        assert not any(line.endswith(' ') for line in result.split('\n'))

    def test_clean_whitespace_strip_text(self):
        """Test stripping leading/trailing whitespace from entire text"""
        text = "   Some text   "
        result = self.converter._clean_whitespace(text)
        assert result == "Some text"


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


class TestCriticalFormatting:
    """Critical tests for formatting that must not break"""
    
    @pytest.mark.critical
    def test_critical_code_preservation(self):
        """Critical: Code blocks must be preserved exactly"""
        converter = MarkdownConverter("slack")
        code = "```python\ndef function():\n    return 'value'\n```"
        result = converter.convert(code)
        
        # Code content must be preserved
        assert "def function():" in result
        assert "return 'value'" in result
        # Indentation should be preserved
        assert "    return" in result
    
    @pytest.mark.critical
    def test_critical_url_formatting(self):
        """Critical: URLs must be properly formatted for Slack"""
        converter = MarkdownConverter("slack")
        
        # Markdown link
        text1 = "[OpenAI](https://openai.com)"
        result1 = converter.convert(text1)
        assert "<https://openai.com|OpenAI>" in result1
        
        # Bare URL
        text2 = "Visit https://example.com"
        result2 = converter.convert(text2)
        assert "<https://example.com>" in result2
    
    @pytest.mark.critical
    def test_critical_no_data_loss(self):
        """Critical: No text should be lost during conversion"""
        converter = MarkdownConverter("slack")
        text = "Line 1\n**Bold** text\n`code`\n[link](http://url.com)\n> Quote"
        result = converter.convert(text)
        
        # All content must be present (formatted differently is ok)
        assert "Line 1" in result
        assert "Bold" in result
        assert "text" in result
        assert "code" in result
        assert "link" in result or "http://url.com" in result
        assert "Quote" in result


class TestRegressionCases:
    """Regression tests for previously found issues"""
    
    def test_regression_nested_formatting(self):
        """Regression: Nested bold/italic should be handled correctly"""
        converter = MarkdownConverter("slack")
        text = "***bold and italic***"
        result = converter.convert(text)
        # Should handle as bold (Slack doesn't support nested)
        assert result.count("*") >= 2
    
    def test_regression_empty_link(self):
        """Regression: Empty links should not crash"""
        converter = MarkdownConverter("slack")
        text = "[](https://example.com)"
        result = converter.convert(text)
        # Should handle gracefully
        assert "https://example.com" in result
    
    def test_regression_special_chars_in_code(self):
        """Regression: Special chars in code blocks should be preserved"""
        converter = MarkdownConverter("slack")
        text = "```\n*bold* _italic_ <tag>\n```"
        result = converter.convert(text)
        # Should preserve special chars in code
        assert "*bold*" in result
        assert "_italic_" in result
        assert "<tag>" in result


class TestContractInterface:
    """Contract tests for interface stability"""
    
    @pytest.mark.smoke
    def test_contract_converter_interface(self):
        """Contract: MarkdownConverter must provide expected interface"""
        converter = MarkdownConverter("slack")
        
        # Required attributes
        assert hasattr(converter, 'platform')
        assert hasattr(converter, 'convert')
        
        # Convert method should accept string and return string
        result = converter.convert("test")
        assert isinstance(result, str)
        
        # Should handle None gracefully
        result_none = converter.convert(None)
        assert result_none == ""
    
    def test_contract_platform_support(self):
        """Contract: Must support slack and discord platforms"""
        # Slack converter should work
        slack_conv = MarkdownConverter("slack")
        assert slack_conv.platform == "slack"
        
        # Discord converter should work
        discord_conv = MarkdownConverter("discord")
        assert discord_conv.platform == "discord"
        
        # Unknown platform should not crash
        unknown_conv = MarkdownConverter("unknown")
        assert unknown_conv.platform == "unknown"


class TestScenarios:
    """Scenario tests for real-world use cases"""
    
    def test_scenario_bot_response_formatting(self):
        """Scenario: Format a typical bot response for Slack"""
        converter = MarkdownConverter("slack")
        
        response = """# Analysis Results
        
Here are your **key findings**:
- First item with `code`
- Second item with [documentation](https://docs.com)

> Note: Results may vary

```python
result = analyze(data)
print(result)
```

For more info, visit https://example.com"""
        
        result = converter.convert(response)
        
        # Headers become bold
        assert "*Analysis Results*" in result
        # Bold becomes Slack bold
        assert "*key findings*" in result
        # Lists get bullet points
        assert "• First item" in result
        # Links are formatted
        assert "<https://docs.com|documentation>" in result
        # Code blocks preserved
        assert "result = analyze(data)" in result
        # Bare URLs wrapped
        assert "<https://example.com>" in result
    
    def test_scenario_user_message_with_code(self):
        """Scenario: User message containing code examples"""
        converter = MarkdownConverter("slack")
        
        message = """I'm getting this error:
```
TypeError: unsupported operand type(s)
```

When I run `my_function()` with these parameters."""
        
        result = converter.convert(message)
        
        # Error message preserved
        assert "TypeError: unsupported operand type(s)" in result
        # Inline code preserved
        assert "`my_function()`" in result
    
    @pytest.mark.smoke
    def test_smoke_basic_conversion(self):
        """Smoke test: Basic conversion functionality works"""
        converter = MarkdownConverter("slack")
        
        # Should not crash on basic input
        try:
            result = converter.convert("Simple text")
            assert result == "Simple text"
            
            result = converter.convert("**bold**")
            assert "*bold*" in result
            
            result = converter.convert("[link](http://url.com)")
            assert "http://url.com" in result
        except Exception as e:
            pytest.fail(f"Basic conversion failed: {e}")


class TestDiagnostics:
    """Diagnostic tests for debugging"""
    
    def test_diagnostic_conversion_steps(self):
        """Diagnostic: Log conversion steps for debugging"""
        converter = MarkdownConverter("slack")
        
        text = "# Header\n**bold** *italic* `code` [link](url)"
        
        # Track transformations
        diagnostics = {
            "original": text,
            "after_conversion": converter.convert(text),
            "platform": converter.platform
        }
        
        print(f"\nDiagnostic Conversion Info:")
        print(f"  Original: {diagnostics['original']}")
        print(f"  Converted: {diagnostics['after_conversion']}")
        print(f"  Platform: {diagnostics['platform']}")
        
        # Verify transformations occurred
        result = diagnostics["after_conversion"]
        assert "*Header*" in result  # Header -> bold
        assert "*bold*" in result  # Bold converted
        assert "_italic_" in result  # Italic converted
        assert "`code`" in result  # Code preserved