"""
Markdown Converter for Multiple Platforms
Converts standard Markdown to platform-specific formats
"""
import re
from typing import List, Optional
from logger import LoggerMixin


class MarkdownConverter(LoggerMixin):
    """Convert Markdown to platform-specific formats"""
    
    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.log_debug(f"MarkdownConverter initialized for {platform}")
    
    def convert(self, text: str) -> str:
        """
        Convert Markdown text to platform-specific format
        
        Args:
            text: Markdown formatted text
        
        Returns:
            Platform-formatted text
        """
        if not text:
            return ""
        
        # Route to platform-specific converter
        if self.platform == "slack":
            return self._convert_to_slack(text)
        elif self.platform == "discord":
            return self._convert_to_discord(text)
        else:
            # Return original markdown for unknown platforms
            return text
    
    def _convert_to_slack(self, text: str) -> str:
        """Convert Markdown to Slack mrkdwn format"""
        # Store code blocks to protect them from conversion
        code_blocks = []
        text = self._extract_code_blocks(text, code_blocks)
        
        # Convert various Markdown elements for Slack
        text = self._convert_headers_slack(text)
        text = self._convert_bold_slack(text)
        text = self._convert_italic_slack(text)
        text = self._convert_strikethrough_slack(text)
        text = self._convert_links_slack(text)
        text = self._convert_lists_slack(text)
        text = self._convert_blockquotes(text)
        text = self._convert_horizontal_rules(text)
        
        # Restore code blocks
        text = self._restore_code_blocks_slack(text, code_blocks)
        
        # Clean up extra whitespace
        text = self._clean_whitespace(text)
        
        return text
    
    def _convert_to_discord(self, text: str) -> str:
        """Convert Markdown for Discord (Discord supports standard Markdown)"""
        # Discord supports standard markdown, so minimal conversion needed
        # Just ensure proper formatting
        
        # Discord supports:
        # - **bold**, *italic*, __underline__, ~~strikethrough~~
        # - # Headers (all levels)
        # - ```code blocks``` with language hints
        # - > blockquotes
        # - [links](url)
        
        # Clean up extra whitespace
        text = self._clean_whitespace(text)
        
        return text
    
    def _extract_code_blocks(self, text: str, storage: List[str]) -> str:
        """Extract code blocks to protect them from conversion"""
        
        # Extract fenced code blocks (```)
        def replace_fenced(match):
            storage.append(match.group(0))
            return f"###CODE_BLOCK_{len(storage) - 1}###"
        
        text = re.sub(r'```[\s\S]*?```', replace_fenced, text)
        
        # Extract inline code (`)
        def replace_inline(match):
            storage.append(match.group(0))
            return f"###CODE_INLINE_{len(storage) - 1}###"
        
        text = re.sub(r'`[^`]+`', replace_inline, text)
        
        return text
    
    def _restore_code_blocks_slack(self, text: str, storage: List[str]) -> str:
        """Restore code blocks after conversion"""
        
        # Restore fenced code blocks
        for i, block in enumerate(storage):
            if block.startswith('```'):
                # Convert to Slack code block format
                lang_match = re.match(r'```(\w+)?\n?([\s\S]*?)```', block)
                if lang_match:
                    lang = lang_match.group(1) or ""
                    code = lang_match.group(2)
                    # Slack doesn't support language hints in the same way
                    # Just use triple backticks
                    text = text.replace(f"###CODE_BLOCK_{i}###", f"```{code}```")
            elif block.startswith('`'):
                # Inline code remains the same in Slack
                text = text.replace(f"###CODE_INLINE_{i}###", block)
        
        return text
    
    def _convert_headers_slack(self, text: str) -> str:
        """Convert Markdown headers to Slack bold text"""
        # H1-H6 headers become bold in Slack
        text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
        return text
    
    def _convert_bold_slack(self, text: str) -> str:
        """Convert Markdown bold to Slack bold"""
        # **text** or __text__ to *text*
        text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
        text = re.sub(r'__(.+?)__', r'*\1*', text)
        return text
    
    def _convert_italic_slack(self, text: str) -> str:
        """Convert Markdown italic to Slack italic"""
        # *text* or _text_ to _text_
        # Need to be careful not to conflict with bold
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'_\1_', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'_\1_', text)
        return text
    
    def _convert_strikethrough_slack(self, text: str) -> str:
        """Convert Markdown strikethrough to Slack strikethrough"""
        # ~~text~~ to ~text~
        text = re.sub(r'~~(.+?)~~', r'~\1~', text)
        return text
    
    def _convert_links_slack(self, text: str) -> str:
        """Convert Markdown links to Slack links"""
        # [text](url) to <url|text>
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)
        
        # Bare URLs should be wrapped in <>
        # Match URLs that aren't already in Slack format
        url_pattern = r'(?<!<)(https?://[^\s<>]+)(?!>)'
        text = re.sub(url_pattern, r'<\1>', text)
        
        return text
    
    def _convert_lists_slack(self, text: str) -> str:
        """Convert Markdown lists to Slack format"""
        lines = text.split('\n')
        converted_lines = []
        
        for line in lines:
            # Convert unordered lists
            if re.match(r'^\s*[-*+]\s+', line):
                # Slack uses • for bullet points
                line = re.sub(r'^(\s*)[-*+]\s+', r'\1• ', line)
            
            # Convert ordered lists
            elif re.match(r'^\s*\d+\.\s+', line):
                # Keep numbered lists as-is
                pass
            
            converted_lines.append(line)
        
        return '\n'.join(converted_lines)
    
    def _convert_blockquotes(self, text: str) -> str:
        """Convert Markdown blockquotes to Slack format"""
        # > text to > text (Slack uses the same format)
        # Multi-line blockquotes
        lines = text.split('\n')
        converted_lines = []
        
        for line in lines:
            if line.startswith('>'):
                # Slack blockquotes use >
                line = re.sub(r'^>\s*', '> ', line)
            converted_lines.append(line)
        
        return '\n'.join(converted_lines)
    
    def _convert_horizontal_rules(self, text: str) -> str:
        """Convert Markdown horizontal rules"""
        # ---, ***, ___ to a line of dashes
        text = re.sub(r'^[-*_]{3,}$', '———————————', text, flags=re.MULTILINE)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace"""
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        return text.strip()


