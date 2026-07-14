"""
Unit tests for Markdown pipe-table conversion in markdown_converter.py

Slack mrkdwn has no table syntax, so pipe tables are re-emitted as padded,
column-aligned code blocks with links moved out to a footnote line.
"""
import re

from markdown_converter import MarkdownConverter


def code_block_lines(text: str) -> list:
    """Return the lines inside the first fenced code block of the output"""
    match = re.search(r'```\n?(.*?)```', text, re.DOTALL)
    assert match, f"no code block in output:\n{text}"
    return [line for line in match.group(1).split('\n') if line.strip()]


class TestTableDetection:
    """Tables are only tables when a valid separator row is present"""

    def setup_method(self):
        self.converter = MarkdownConverter("slack")

    def test_basic_three_column_table(self):
        """Test a basic 3-column table becomes a padded code block"""
        text = (
            "| Year | Model | Lab |\n"
            "| --- | --- | --- |\n"
            "| 2017 | Transformer | Google |\n"
            "| 2019 | GPT-2 | OpenAI |"
        )
        result = self.converter.convert(text)

        assert result.startswith("```")
        assert result.endswith("```")
        assert "|" not in result

        lines = code_block_lines(result)
        assert lines[0] == "Year  Model        Lab"
        assert lines[1] == "----  -----------  ------"
        assert lines[2] == "2017  Transformer  Google"
        assert lines[3] == "2019  GPT-2        OpenAI"

    def test_columns_are_padded_to_widest_cell(self):
        """Test every column is padded to its widest cell (header or body)"""
        text = (
            "| A | Long header |\n"
            "|---|---|\n"
            "| a very long cell | x |"
        )
        lines = code_block_lines(self.converter.convert(text))

        assert lines[0] == "A                 Long header"
        assert lines[2] == "a very long cell  x"

    def test_alignment_colons_in_separator(self):
        """Test a separator row with alignment colons is accepted and stripped"""
        text = (
            "| Item | Qty | Cost |\n"
            "|:--- | :---: | ---: |\n"
            "| Nail | 10 | 1.50 |"
        )
        result = self.converter.convert(text)
        lines = code_block_lines(result)

        assert ":---" not in result
        assert lines[0] == "Item  Qty  Cost"
        assert lines[2] == "Nail  10   1.50"

    def test_table_without_edge_pipes(self):
        """Test leading/trailing pipes are optional"""
        text = (
            "Name | Role\n"
            "--- | ---\n"
            "Ada | Engineer"
        )
        lines = code_block_lines(self.converter.convert(text))

        assert lines[0] == "Name  Role"
        assert lines[2] == "Ada   Engineer"

    def test_pipes_without_separator_row_left_alone(self):
        """Test prose that merely contains pipes is not converted"""
        text = "Use `a | b` for either, and | this | is not a table."
        result = self.converter.convert(text)

        assert "```" not in result
        assert "| this | is not a table." in result

    def test_two_pipe_rows_without_separator_left_alone(self):
        """Test a header-like row with no separator row is not a table"""
        text = (
            "| Year | Model |\n"
            "| 2017 | Transformer |"
        )
        result = self.converter.convert(text)

        assert "```" not in result
        assert "| 2017 | Transformer |" in result

    def test_separator_without_body_rows_left_alone(self):
        """Test a header + separator with no body rows is not a table"""
        text = (
            "| Year | Model |\n"
            "| --- | --- |"
        )
        result = self.converter.convert(text)

        assert "```" not in result

    def test_horizontal_rule_is_not_a_separator_row(self):
        """Test a bare --- rule after a pipe-free line stays a horizontal rule"""
        text = (
            "Some heading\n"
            "---\n"
            "Body text"
        )
        result = self.converter.convert(text)

        assert "```" not in result
        assert "———————————" in result


class TestTableCellFlattening:
    """Cells are flattened to plain text before padding"""

    def setup_method(self):
        self.converter = MarkdownConverter("slack")

    def test_bold_italic_code_strike_are_unwrapped(self):
        """Test inline markup inside cells is stripped, not converted"""
        text = (
            "| Name | Note |\n"
            "|---|---|\n"
            "| **Bold** | `code` |\n"
            "| __Also bold__ | *ital* |\n"
            "| _ital_ | ~~gone~~ |"
        )
        result = self.converter.convert(text)
        lines = code_block_lines(result)

        # column 0 is padded to 'Also bold' (9 chars), then cells joined by 2 spaces
        assert lines[2] == "Bold       code"
        assert lines[3] == "Also bold  ital"
        assert lines[4] == "ital       gone"
        for marker in ("**", "__", "`code`", "~~", "~gone~"):
            assert marker not in result

    def test_flattened_width_ignores_markup_characters(self):
        """Test padding is computed from the flattened text, not the raw markup"""
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| **xx** | y |"
        )
        lines = code_block_lines(self.converter.convert(text))

        # 'xx' is 2 chars once flattened, so the rule under column A is 2 dashes
        assert lines[1] == "--  -"
        assert lines[2] == "xx  y"

    def test_escaped_pipes_are_literal_content(self):
        r"""Test \| inside a cell is content, not a delimiter"""
        text = (
            "| Op | Meaning |\n"
            "|---|---|\n"
            r"| a \| b | logical or |"
        )
        lines = code_block_lines(self.converter.convert(text))

        assert lines[2] == "a | b  logical or"

    def test_pipe_inside_inline_code_is_literal_content(self):
        """Test a pipe inside a code span does not split the cell"""
        text = (
            "| Op | Meaning |\n"
            "|---|---|\n"
            "| `a | b` | logical or |"
        )
        result = self.converter.convert(text)
        lines = code_block_lines(result)

        assert "###CODE_INLINE" not in result
        assert lines[2] == "a | b  logical or"

    def test_ragged_rows_do_not_crash(self):
        """Test short rows are padded and overflow cells fold into the last column"""
        text = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| 1 |\n"
            "| 1 | 2 | 3 | 4 |\n"
            "| 1 | 2 | 3 |"
        )
        lines = code_block_lines(self.converter.convert(text))

        assert lines[0] == "A  B  C"
        assert lines[2] == "1"           # padded, then rstripped
        assert lines[3] == "1  2  3 | 4"  # extras kept in the last column
        assert lines[4] == "1  2  3"


class TestTableLinks:
    """Links become numbered footnotes posted after the code block"""

    def setup_method(self):
        self.converter = MarkdownConverter("slack")

    def test_link_becomes_footnote_reference(self):
        """Test a [label](url) cell renders as 'label [1]' with a footnote line"""
        text = (
            "| Year | Paper |\n"
            "|---|---|\n"
            "| 2017 | [Attention](https://arxiv.org/abs/1706.03762) |"
        )
        result = self.converter.convert(text)
        lines = code_block_lines(result)

        assert lines[2] == "2017  Attention [1]"
        assert "_Sources:_ <https://arxiv.org/abs/1706.03762|[1] Attention>" in result
        # the footnote sits immediately after the code block
        assert result.index("```", 3) < result.index("_Sources:_")

    def test_footnotes_numbered_in_document_order_and_deduped(self):
        """Test footnote numbering follows document order and dedupes identical URLs"""
        text = (
            "| Model | Paper | Blog |\n"
            "|---|---|---|\n"
            "| A | [P1](https://a.example/p1) | [Blog](https://a.example/blog) |\n"
            "| B | [Again](https://a.example/p1) | [Other](https://b.example/x) |"
        )
        result = self.converter.convert(text)
        lines = code_block_lines(result)

        assert lines[2] == "A      P1 [1]     Blog [2]"
        assert lines[3] == "B      Again [1]  Other [3]"

        footnote = result.split("_Sources:_ ")[1]
        assert footnote == (
            "<https://a.example/p1|[1] P1> · "
            "<https://a.example/blog|[2] Blog> · "
            "<https://b.example/x|[3] Other>"
        )
        assert footnote.count("https://a.example/p1") == 1

    def test_no_links_means_no_footnote_line(self):
        """Test a table without links emits no footnote line"""
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |"
        )
        result = self.converter.convert(text)

        assert "_Sources:_" not in result

    def test_footnote_urls_are_not_double_wrapped(self):
        """Test the link pass does not re-wrap URLs already in Slack format"""
        text = (
            "| Paper |\n"
            "|---|\n"
            "| [Attention](https://arxiv.org/abs/1706.03762) |"
        )
        result = self.converter.convert(text)

        assert "<<" not in result
        assert ">>" not in result
        assert result.count("https://arxiv.org/abs/1706.03762") == 1

    def test_bold_link_label_is_flattened(self):
        """Test markup inside a link label is stripped in cell and footnote"""
        text = (
            "| Paper |\n"
            "|---|\n"
            "| [**Attention**](https://arxiv.org/abs/1706.03762) |"
        )
        result = self.converter.convert(text)

        assert code_block_lines(result)[2] == "Attention [1]"
        assert "|[1] Attention>" in result


class TestTablesInContext:
    """Tables coexist with code blocks, prose and other tables"""

    def setup_method(self):
        self.converter = MarkdownConverter("slack")

    def test_table_inside_fenced_code_block_untouched(self):
        """Test a pipe table inside a ``` block keeps its raw pipes"""
        text = (
            "Here is markdown source:\n\n"
            "```\n"
            "| Year | Model |\n"
            "| --- | --- |\n"
            "| 2017 | Transformer |\n"
            "```"
        )
        result = self.converter.convert(text)

        assert "| Year | Model |" in result
        assert "| --- | --- |" in result
        assert "| 2017 | Transformer |" in result
        assert "###TABLE_BLOCK" not in result

    def test_prose_around_table_is_preserved(self):
        """Test surrounding prose survives and is still converted"""
        text = (
            "Here is the **summary** you asked for:\n\n"
            "| Year | Model |\n"
            "|---|---|\n"
            "| 2017 | Transformer |\n\n"
            "Let me know if you want _more_ detail."
        )
        result = self.converter.convert(text)

        assert result.startswith("Here is the *summary* you asked for:")
        assert result.endswith("Let me know if you want _more_ detail.")
        assert "2017  Transformer" in result
        assert "###TABLE_BLOCK" not in result

    def test_table_after_a_list_keeps_the_list(self):
        """Test a table adjacent to a bullet list leaves the list conversion intact"""
        text = (
            "- first\n"
            "- second\n\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |"
        )
        result = self.converter.convert(text)

        assert "• first" in result
        assert "• second" in result
        assert "1  2" in result

    def test_multiple_tables_each_get_their_own_footnotes(self):
        """Test each table numbers its footnotes independently, starting at 1"""
        text = (
            "| Paper |\n"
            "|---|\n"
            "| [One](https://example.com/one) |\n"
            "\n"
            "Some prose in between.\n"
            "\n"
            "| Paper |\n"
            "|---|\n"
            "| [Two](https://example.com/two) |"
        )
        result = self.converter.convert(text)

        assert result.count("```") == 4
        assert "One [1]" in result
        assert "Two [1]" in result
        assert "<https://example.com/one|[1] One>" in result
        assert "<https://example.com/two|[1] Two>" in result
        assert "Some prose in between." in result

    def test_table_dashes_survive_the_horizontal_rule_pass(self):
        """Test the emitted dashed rule is not eaten by _convert_horizontal_rules"""
        text = (
            "| Header |\n"
            "|---|\n"
            "| value |"
        )
        result = self.converter.convert(text)

        assert "------" in result
        assert "———————————" not in result

    def test_real_world_release_table(self):
        """Test the shipped regression: a model-release table renders as aligned text"""
        text = (
            "| Date | Model | Lab | Link |\n"
            "| --- | --- | --- | --- |\n"
            "| 2017-06-12 | Transformer | Google Research "
            "| [Paper](https://arxiv.org/abs/1706.03762) |\n"
            "| 2019-02-14 | GPT-2 | OpenAI | [Post](https://openai.com/index/gpt-2) |"
        )
        result = self.converter.convert(text)
        lines = code_block_lines(result)

        assert lines[0] == "Date        Model        Lab              Link"
        assert lines[2] == "2017-06-12  Transformer  Google Research  Paper [1]"
        assert lines[3] == "2019-02-14  GPT-2        OpenAI           Post [2]"
        assert "_Sources:_ <https://arxiv.org/abs/1706.03762|[1] Paper> · " \
               "<https://openai.com/index/gpt-2|[2] Post>" in result
