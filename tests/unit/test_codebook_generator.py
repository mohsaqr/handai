"""
Tests for Codebook Generator Tool
"""

import pytest
from tools.codebook_generator import (
    Code, Codebook, CodebookGeneratorTool, SAMPLE_QUALITATIVE_DATA
)


class TestCodeDataclass:
    """Tests for Code dataclass"""

    def test_code_creation_minimal(self):
        """Code can be created with just name and definition"""
        code = Code(name="Test Code", definition="A test definition")
        assert code.name == "Test Code"
        assert code.definition == "A test definition"
        assert code.inclusion_criteria == []
        assert code.exclusion_criteria == []
        assert code.examples == []
        assert code.parent_category == ""
        assert code.is_borderline is False
        assert code.borderline_notes is None

    def test_code_creation_full(self):
        """Code can be created with all fields"""
        code = Code(
            name="Positive Experience",
            definition="Participant expresses satisfaction or enjoyment",
            inclusion_criteria=["Uses positive language", "Mentions satisfaction"],
            exclusion_criteria=["Mixed feelings", "Neutral statements"],
            examples=["I loved this experience", "It was fantastic"],
            parent_category="Sentiment",
            is_borderline=True,
            borderline_notes="May overlap with neutral when qualified"
        )
        assert code.name == "Positive Experience"
        assert len(code.inclusion_criteria) == 2
        assert len(code.exclusion_criteria) == 2
        assert len(code.examples) == 2
        assert code.parent_category == "Sentiment"
        assert code.is_borderline is True
        assert "overlap" in code.borderline_notes


class TestCodebookDataclass:
    """Tests for Codebook dataclass"""

    def test_codebook_creation_empty(self):
        """Codebook can be created with minimal fields"""
        codebook = Codebook(title="Test Codebook", approach="inductive")
        assert codebook.title == "Test Codebook"
        assert codebook.approach == "inductive"
        assert codebook.codes == []
        assert codebook.categories == {}
        assert codebook.data_source == ""
        assert codebook.total_items_analyzed == 0

    def test_codebook_to_dict(self):
        """Codebook can be converted to dictionary"""
        code = Code(
            name="Test Code",
            definition="Test definition",
            parent_category="Category A"
        )
        codebook = Codebook(
            title="Test Codebook",
            approach="inductive",
            codes=[code],
            categories={"Category A": ["Test Code"]},
            data_source="test.csv",
            total_items_analyzed=100
        )
        result = codebook.to_dict()

        assert result["title"] == "Test Codebook"
        assert result["approach"] == "inductive"
        assert len(result["codes"]) == 1
        assert result["codes"][0]["name"] == "Test Code"
        assert result["categories"] == {"Category A": ["Test Code"]}
        assert result["data_source"] == "test.csv"
        assert result["total_items_analyzed"] == 100

    def test_codebook_to_markdown(self):
        """Codebook can be converted to Markdown format"""
        code = Code(
            name="Test Code",
            definition="Test definition",
            inclusion_criteria=["Criterion 1"],
            exclusion_criteria=["Exclusion 1"],
            examples=["Example quote"],
            parent_category="Category A"
        )
        codebook = Codebook(
            title="Test Codebook",
            approach="inductive",
            codes=[code],
            categories={"Category A": ["Test Code"]},
            data_source="test.csv",
            total_items_analyzed=100
        )
        markdown = codebook.to_markdown()

        assert "# Test Codebook" in markdown
        assert "**Approach:** Inductive" in markdown
        assert "## Category A" in markdown
        assert "### Test Code" in markdown
        assert "**Definition:** Test definition" in markdown
        assert "**Apply when:**" in markdown
        assert "- Criterion 1" in markdown
        assert "**Do NOT apply when:**" in markdown
        assert "- Exclusion 1" in markdown
        assert '"Example quote"' in markdown

    def test_codebook_to_markdown_borderline(self):
        """Borderline codes are marked with warning in Markdown"""
        code = Code(
            name="Borderline Code",
            definition="Ambiguous definition",
            parent_category="Category A",
            is_borderline=True,
            borderline_notes="This is ambiguous"
        )
        codebook = Codebook(
            title="Test",
            approach="inductive",
            codes=[code],
            categories={"Category A": ["Borderline Code"]}
        )
        markdown = codebook.to_markdown()

        assert ":warning:" in markdown
        assert "**Borderline Notes:**" in markdown

    def test_codebook_to_csv_rows(self):
        """Codebook can be converted to CSV row format"""
        code = Code(
            name="Test Code",
            definition="Test definition",
            inclusion_criteria=["Criterion 1", "Criterion 2"],
            exclusion_criteria=["Exclusion 1"],
            examples=["Example 1", "Example 2"],
            parent_category="Category A",
            is_borderline=True,
            borderline_notes="Some notes"
        )
        codebook = Codebook(
            title="Test",
            approach="inductive",
            codes=[code]
        )
        rows = codebook.to_csv_rows()

        assert len(rows) == 1
        row = rows[0]
        assert row["code_name"] == "Test Code"
        assert row["definition"] == "Test definition"
        assert row["parent_category"] == "Category A"
        assert "Criterion 1; Criterion 2" == row["inclusion_criteria"]
        assert "Exclusion 1" == row["exclusion_criteria"]
        assert "Example 1 | Example 2" == row["examples"]
        assert row["is_borderline"] is True
        assert row["borderline_notes"] == "Some notes"

    def test_codebook_to_qualitative_coder_prompt(self):
        """Codebook generates valid Qualitative Coder prompt"""
        code = Code(
            name="Test Code",
            definition="Test definition",
            inclusion_criteria=["Apply when X"],
            exclusion_criteria=["Do not apply when Y"],
            parent_category="Category A"
        )
        codebook = Codebook(
            title="Test",
            approach="inductive",
            codes=[code],
            categories={"Category A": ["Test Code"]}
        )
        prompt = codebook.to_qualitative_coder_prompt()

        assert "You are a qualitative coder" in prompt
        assert "## Codebook" in prompt
        assert "### Category: Category A" in prompt
        assert "**Test Code**" in prompt
        assert "Definition: Test definition" in prompt
        assert "Apply when: Apply when X" in prompt
        assert "Do NOT apply when: Do not apply when Y" in prompt
        assert "## Instructions" in prompt
        assert "comma-separated" in prompt


class TestCodebookGeneratorTool:
    """Tests for CodebookGeneratorTool class"""

    def test_tool_metadata(self):
        """Tool has correct metadata"""
        tool = CodebookGeneratorTool()
        assert tool.id == "codebook-generator"
        assert tool.name == "Codebook Generator"
        assert tool.category == "Analysis"
        assert tool.icon == ":material/book:"

    def test_tool_info(self):
        """Tool info returns correct structure"""
        tool = CodebookGeneratorTool()
        info = tool.get_info()
        assert info["id"] == "codebook-generator"
        assert info["name"] == "Codebook Generator"
        assert info["category"] == "Analysis"


class TestSampleData:
    """Tests for sample qualitative data"""

    def test_sample_data_structure(self):
        """Sample data has expected structure"""
        assert "response" in SAMPLE_QUALITATIVE_DATA
        assert "participant_id" in SAMPLE_QUALITATIVE_DATA
        assert "context" in SAMPLE_QUALITATIVE_DATA

    def test_sample_data_lengths_match(self):
        """All columns in sample data have same length"""
        lengths = [len(v) for v in SAMPLE_QUALITATIVE_DATA.values()]
        assert all(l == lengths[0] for l in lengths)

    def test_sample_data_not_empty(self):
        """Sample data contains entries"""
        assert len(SAMPLE_QUALITATIVE_DATA["response"]) > 0
        assert len(SAMPLE_QUALITATIVE_DATA["response"]) >= 10


class TestPromptGeneration:
    """Tests for LLM prompt generation"""

    def test_qualitative_coder_prompt_format(self):
        """Generated prompt follows expected format for Qualitative Coder"""
        codes = [
            Code(
                name="Technical Issues",
                definition="Problems with technology or platforms",
                inclusion_criteria=["Mentions software bugs", "Connection problems"],
                exclusion_criteria=["User error"],
                parent_category="Challenges"
            ),
            Code(
                name="Social Learning",
                definition="Learning through peer interaction",
                inclusion_criteria=["Group work", "Discussion"],
                exclusion_criteria=["Individual study"],
                parent_category="Learning Styles"
            )
        ]
        codebook = Codebook(
            title="Learning Experience Codebook",
            approach="inductive",
            codes=codes,
            categories={
                "Challenges": ["Technical Issues"],
                "Learning Styles": ["Social Learning"]
            }
        )

        prompt = codebook.to_qualitative_coder_prompt()

        # Check structure
        assert "You are a qualitative coder" in prompt
        assert "## Codebook" in prompt
        assert "## Instructions" in prompt

        # Check categories
        assert "### Category: Challenges" in prompt
        assert "### Category: Learning Styles" in prompt

        # Check code definitions
        assert "**Technical Issues**" in prompt
        assert "Definition: Problems with technology or platforms" in prompt
        assert "**Social Learning**" in prompt

        # Check criteria
        assert "Apply when:" in prompt
        assert "Do NOT apply when:" in prompt

        # Check instructions
        assert "comma-separated" in prompt
        assert "none" in prompt.lower()


class TestExportFormats:
    """Tests for export format generation"""

    def test_json_export_is_valid_json(self):
        """to_dict produces JSON-serializable output"""
        import json

        code = Code(name="Test", definition="Test def", parent_category="Cat")
        codebook = Codebook(
            title="Test",
            approach="hybrid",
            codes=[code],
            categories={"Cat": ["Test"]}
        )

        # Should not raise
        json_str = json.dumps(codebook.to_dict())
        parsed = json.loads(json_str)

        assert parsed["title"] == "Test"
        assert len(parsed["codes"]) == 1

    def test_csv_export_creates_flat_structure(self):
        """CSV export creates flat rows suitable for DataFrame"""
        import pandas as pd

        codes = [
            Code(name="Code1", definition="Def1", parent_category="Cat1"),
            Code(name="Code2", definition="Def2", parent_category="Cat2")
        ]
        codebook = Codebook(
            title="Test",
            approach="inductive",
            codes=codes,
            categories={"Cat1": ["Code1"], "Cat2": ["Code2"]}
        )

        rows = codebook.to_csv_rows()
        df = pd.DataFrame(rows)

        assert len(df) == 2
        assert "code_name" in df.columns
        assert "definition" in df.columns
        assert "parent_category" in df.columns
        assert list(df["code_name"]) == ["Code1", "Code2"]

    def test_markdown_export_is_valid_markdown(self):
        """Markdown export produces valid Markdown structure"""
        code = Code(
            name="Test Code",
            definition="Test definition",
            examples=["Quote 1"],
            parent_category="Category"
        )
        codebook = Codebook(
            title="Test Codebook",
            approach="deductive",
            codes=[code],
            categories={"Category": ["Test Code"]}
        )

        md = codebook.to_markdown()

        # Check Markdown elements
        assert md.startswith("# ")  # H1 heading
        assert "## " in md  # H2 heading
        assert "### " in md  # H3 heading
        assert "**" in md  # Bold text
        assert "> " in md  # Blockquote
        assert "---" in md  # Horizontal rule
