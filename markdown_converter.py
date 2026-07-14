"""
Markdown Converter for Multiple Platforms
Converts standard Markdown to platform-specific formats
"""
import re
from typing import Dict, List, Tuple
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
        else:
            # Return original markdown for unknown platforms
            return text
    
    def _convert_to_slack(self, text: str) -> str:
        """Convert Markdown to Slack mrkdwn format"""
        # Store code blocks to protect them from conversion
        code_blocks = []
        text = self._extract_code_blocks(text, code_blocks)

        # Tables are converted immediately after code extraction and before every
        # other pass, for two reasons:
        #   1. Fenced/inline code is already stashed, so a pipe table that lives
        #      inside a ``` block is a placeholder here and is never touched.
        #   2. Cell contents must still be raw Markdown. We flatten them ourselves
        #      (a code block renders markup literally) and we need [label](url)
        #      intact to build footnotes -- _convert_links_slack() would already
        #      have rewritten it to <url|label>.
        # The rendered table is itself stashed as a placeholder (separate storage
        # from code_blocks, restored after them) so that the ``` fence we emit,
        # its padding, and its dashed rule survive the italic/list/rule passes.
        tables = []
        text = self._convert_tables_slack(text, tables, code_blocks)

        # Convert various Markdown elements for Slack
        # Important: Convert italic before headers/bold to avoid conflicts
        text = self._convert_italic_slack(text)
        text = self._convert_bold_slack(text)
        text = self._convert_headers_slack(text)
        text = self._convert_strikethrough_slack(text)
        text = self._convert_links_slack(text)
        text = self._convert_lists_slack(text)
        text = self._convert_blockquotes(text)
        text = self._convert_horizontal_rules(text)

        # Restore code blocks, then tables (tables last: their content is final
        # Slack output and must not be re-scanned by the code-block restore)
        text = self._restore_code_blocks_slack(text, code_blocks)
        text = self._restore_tables_slack(text, tables)

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
                    code = lang_match.group(2)
                    # Slack doesn't support language hints in the same way
                    # Just use triple backticks
                    text = text.replace(f"###CODE_BLOCK_{i}###", f"```{code}```")
            elif block.startswith('`'):
                # Inline code remains the same in Slack
                text = text.replace(f"###CODE_INLINE_{i}###", block)
        
        return text
    
    # --- Tables ---------------------------------------------------------------
    # Slack mrkdwn has no table syntax, so a Markdown pipe table would ship as raw
    # `| a | b |` rows. We re-emit it as a padded, column-aligned code block, which
    # Slack renders in a fixed-width font so the columns line up.

    # Cell delimiters are pipes that are not backslash-escaped (`\|` is content)
    _CELL_DELIM_RE = re.compile(r'(?<!\\)\|')
    _TRAILING_DELIM_RE = re.compile(r'(?<!\\)\|$')
    # Separator cells: ---, :---, ---:, :---: (surrounding spaces already stripped)
    _SEPARATOR_CELL_RE = re.compile(r'^:?-+:?$')
    # [label](url) with an optional "title"
    _LINK_RE = re.compile(r'\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)')
    # Code stashed by _extract_code_blocks() before this pass runs
    _CODE_PLACEHOLDER_RE = re.compile(r'###CODE_(?:BLOCK|INLINE)_(\d+)###')

    def _convert_tables_slack(self, text: str, storage: List[str], code_blocks: List[str]) -> str:
        """
        Replace GitHub-style pipe tables with placeholders for rendered code blocks

        A table is a row of cells, a separator row (`|---|:---:|`), and one or more
        body rows. Anything without a valid separator row is left untouched, so
        prose that merely contains pipes is safe.
        """
        lines = text.split('\n')
        converted: List[str] = []
        i = 0

        while i < len(lines):
            if (
                i + 2 < len(lines)
                and self._is_table_row(lines[i])
                and self._is_separator_row(lines[i + 1])
                and self._is_table_row(lines[i + 2])
                and self._split_table_row(lines[i])
            ):
                # Body runs until the first line that is not a table row
                end = i + 2
                while end < len(lines) and self._is_table_row(lines[end]):
                    end += 1

                storage.append(
                    self._render_table_slack(lines[i], lines[i + 2:end], code_blocks)
                )
                converted.append(f"###TABLE_BLOCK_{len(storage) - 1}###")
                self.log_debug(f"Converted Markdown table ({end - i - 2} rows) to code block")
                i = end
                continue

            converted.append(lines[i])
            i += 1

        return '\n'.join(converted)

    def _restore_tables_slack(self, text: str, storage: List[str]) -> str:
        """Restore rendered tables after conversion (content is already mrkdwn)"""
        for i, block in enumerate(storage):
            text = text.replace(f"###TABLE_BLOCK_{i}###", block)

        return text

    def _is_table_row(self, line: str) -> bool:
        """Check whether a line could be a table row (holds a cell delimiter)"""
        return bool(line.strip()) and bool(self._CELL_DELIM_RE.search(line))

    def _is_separator_row(self, line: str) -> bool:
        """Check whether a line is a table separator row (`|---|:---:|---:|`)"""
        if not self._is_table_row(line):
            # Requiring a pipe keeps a plain `---` horizontal rule out of tables
            return False

        cells = self._split_table_row(line)
        return bool(cells) and all(self._SEPARATOR_CELL_RE.match(cell) for cell in cells)

    def _split_table_row(self, line: str) -> List[str]:
        """Split a table row into cells, honoring escaped pipes and optional edge pipes"""
        line = line.strip()
        cells = self._CELL_DELIM_RE.split(line)

        # Leading/trailing pipes are optional; drop the empty cells they produce
        if line.startswith('|'):
            cells = cells[1:]
        if len(cells) > 1 and self._TRAILING_DELIM_RE.search(line):
            cells = cells[:-1]

        return [cell.strip() for cell in cells]

    def _render_table_slack(
        self, header_line: str, body_lines: List[str], code_blocks: List[str]
    ) -> str:
        """Render a pipe table as a padded code block plus an optional footnote line"""
        footnotes: Dict[str, Tuple[int, str]] = {}
        header = [
            self._flatten_cell(c, footnotes, code_blocks)
            for c in self._split_table_row(header_line)
        ]
        rows = [
            [self._flatten_cell(c, footnotes, code_blocks) for c in self._split_table_row(line)]
            for line in body_lines
        ]
        columns = len(header)

        # Ragged rows: pad short ones, fold any overflow back into the last column
        normalized: List[List[str]] = []
        for row in rows:
            if len(row) > columns:
                row = row[:columns - 1] + [' | '.join(row[columns - 1:])]
            elif len(row) < columns:
                row = row + [''] * (columns - len(row))
            normalized.append(row)

        widths = [len(cell) for cell in header]
        for row in normalized:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def format_row(cells: List[str]) -> str:
            # rstrip: trailing pad would be stripped by _clean_whitespace anyway
            return '  '.join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        rendered = [format_row(header), '  '.join('-' * width for width in widths).rstrip()]
        rendered += [format_row(row) for row in normalized]
        block = '```\n' + '\n'.join(rendered) + '\n```'

        if footnotes:
            links = sorted(footnotes.items(), key=lambda item: item[1][0])
            block += '\n_Sources:_ ' + ' · '.join(
                f"<{url}|[{number}] {label}>" for url, (number, label) in links
            )

        return block

    def _flatten_cell(
        self, cell: str, footnotes: Dict[str, Tuple[int, str]], code_blocks: List[str]
    ) -> str:
        """
        Flatten a cell to plain text, moving any links out to numbered footnotes

        A code block renders everything literally, so inline markup inside a cell is
        just noise. Links would lose their URL entirely, so `[label](url)` becomes
        `label [n]` and the URL is collected; identical URLs share a number.

        Inline code was stashed by _extract_code_blocks() before this pass, so cells
        may hold placeholders. They are resolved here -- after the row was split, so
        that a pipe inside a code span stays content -- and never reach the text that
        _restore_code_blocks_slack() sees.
        """

        def restore_code(match):
            index = int(match.group(1))
            return code_blocks[index] if index < len(code_blocks) else match.group(0)

        cell = self._CODE_PLACEHOLDER_RE.sub(restore_code, cell)

        def replace_link(match):
            label = self._strip_inline_markup(match.group(1)).strip()
            url = match.group(2).strip()

            if url not in footnotes:
                # Slack link syntax is <url|label>, so | and <> cannot ride along
                safe_label = re.sub(r'[<>|]', '', label).strip() or url
                footnotes[url] = (len(footnotes) + 1, safe_label)

            number = footnotes[url][0]
            return f"{label} [{number}]".strip()

        cell = self._LINK_RE.sub(replace_link, cell)
        cell = self._strip_inline_markup(cell)
        cell = cell.replace('\\|', '|')

        return re.sub(r'\s+', ' ', cell).strip()

    def _strip_inline_markup(self, text: str) -> str:
        """Unwrap bold/italic/code/strikethrough markers, keeping the text inside"""
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'`+([^`]*)`+', r'\1', text)
        text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!_)_(?!_)([^_]+?)_(?!_)', r'\1', text)

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


