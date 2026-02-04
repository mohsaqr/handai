"""
Tests for Automator Tool
"""

import pytest
import json
from tools.automator import (
    OutputField, FewShotExample, AutomatorConfig, AutomatorTool,
    SAMPLE_AUTOMATOR_DATA, build_system_prompt, extract_json,
    extract_xml, extract_csv_row, validate_output, sanitize_llm_output
)


class TestOutputFieldDataclass:
    """Tests for OutputField dataclass"""

    def test_output_field_minimal(self):
        """OutputField can be created with just name and type"""
        field = OutputField(name="sentiment", field_type="text")
        assert field.name == "sentiment"
        assert field.field_type == "text"
        assert field.required is True
        assert field.constraints is None
        assert field.default is None

    def test_output_field_full(self):
        """OutputField can be created with all fields"""
        field = OutputField(
            name="sentiment",
            field_type="text",
            required=True,
            constraints="one of: positive, negative, neutral",
            default="neutral"
        )
        assert field.name == "sentiment"
        assert field.field_type == "text"
        assert field.required is True
        assert "positive" in field.constraints
        assert field.default == "neutral"

    def test_output_field_optional(self):
        """OutputField can be marked as optional"""
        field = OutputField(name="notes", field_type="text", required=False)
        assert field.required is False


class TestFewShotExampleDataclass:
    """Tests for FewShotExample dataclass"""

    def test_few_shot_example_creation(self):
        """FewShotExample can be created with input and output"""
        example = FewShotExample(
            input_data={"text": "I love this product!"},
            output_data={"sentiment": "positive", "confidence": 95}
        )
        assert example.input_data["text"] == "I love this product!"
        assert example.output_data["sentiment"] == "positive"
        assert example.output_data["confidence"] == 95


class TestAutomatorConfigDataclass:
    """Tests for AutomatorConfig dataclass"""

    def test_config_minimal(self):
        """AutomatorConfig can be created with required fields"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="json"
        )
        assert config.task_description == "Classify sentiment"
        assert config.input_columns == ["text"]
        assert len(config.output_fields) == 1
        assert config.output_format == "json"
        assert config.few_shot_examples == []
        assert config.confidence_enabled is False
        assert config.confidence_threshold == 70
        assert config.consistency_prompt is None
        assert config.handle_empty == "skip"
        assert config.system_prompt_override is None

    def test_config_full(self):
        """AutomatorConfig can be created with all fields"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text", "source"],
            output_fields=[
                OutputField(name="sentiment", field_type="text", constraints="one of: positive, negative, neutral"),
                OutputField(name="confidence", field_type="number")
            ],
            output_format="json",
            few_shot_examples=[
                FewShotExample({"text": "Great!"}, {"sentiment": "positive", "confidence": 90})
            ],
            confidence_enabled=True,
            confidence_threshold=80,
            consistency_prompt="Always use lowercase labels",
            handle_empty="default",
            system_prompt_override=None
        )
        assert len(config.output_fields) == 2
        assert len(config.few_shot_examples) == 1
        assert config.confidence_enabled is True
        assert config.confidence_threshold == 80


class TestSanitizeLlmOutput:
    """Tests for LLM output sanitization"""

    def test_sanitize_removes_channel_tags(self):
        """Removes <|channel|> style tags"""
        text = '<|channel|>final <|constrain|>result some actual content'
        result = sanitize_llm_output(text)
        assert '<|channel|>' not in result
        assert '<|constrain|>' not in result
        assert 'actual content' in result

    def test_sanitize_removes_message_tags(self):
        """Removes <|message|> tags"""
        text = '<|message|>{"sentiment": "positive"}'
        result = sanitize_llm_output(text)
        assert '<|message|>' not in result
        assert '{"sentiment": "positive"}' in result

    def test_sanitize_removes_result_tags(self):
        """Removes <|result|> and <|final|> tags"""
        text = '<|final|><|result|>positive'
        result = sanitize_llm_output(text)
        assert '<|final|>' not in result
        assert '<|result|>' not in result
        assert 'positive' in result

    def test_sanitize_removes_repeated_result(self):
        """Removes repeated 'result' artifacts"""
        text = 'result result result actual output'
        result = sanitize_llm_output(text)
        assert 'result result result' not in result
        assert 'actual output' in result

    def test_sanitize_preserves_normal_text(self):
        """Preserves normal text without tags"""
        text = '{"sentiment": "positive", "confidence": 95}'
        result = sanitize_llm_output(text)
        assert result == text

    def test_sanitize_handles_arabic_with_tags(self):
        """Handles Arabic text mixed with injection tags"""
        text = '<|channel|>final <|constrain|>result<|message|>جميع الصور بين الثانية والرابعة'
        result = sanitize_llm_output(text)
        assert '<|channel|>' not in result
        assert '<|constrain|>' not in result
        assert '<|message|>' not in result

    def test_sanitize_handles_empty_input(self):
        """Handles empty input"""
        assert sanitize_llm_output('') == ''
        assert sanitize_llm_output(None) is None

    def test_sanitize_complex_injection(self):
        """Handles complex injection attempts"""
        text = '"<|channel|>final <|constrain|>=""result"">some content"'
        result = sanitize_llm_output(text)
        assert '<|channel|>' not in result
        assert '<|constrain|>' not in result


class TestExtractJson:
    """Tests for JSON extraction function"""

    def test_extract_json_direct(self):
        """Extracts JSON from clean JSON string"""
        text = '{"sentiment": "positive", "score": 95}'
        result = extract_json(text)
        assert result == {"sentiment": "positive", "score": 95}

    def test_extract_json_with_code_block(self):
        """Extracts JSON from markdown code block"""
        text = '```json\n{"sentiment": "positive"}\n```'
        result = extract_json(text)
        assert result == {"sentiment": "positive"}

    def test_extract_json_with_surrounding_text(self):
        """Extracts JSON when surrounded by other text"""
        text = 'Here is the result: {"sentiment": "positive"} Thank you!'
        result = extract_json(text)
        assert result == {"sentiment": "positive"}

    def test_extract_json_empty(self):
        """Returns None for empty input"""
        assert extract_json("") is None
        assert extract_json(None) is None

    def test_extract_json_invalid(self):
        """Returns None for invalid JSON"""
        assert extract_json("not json at all") is None
        assert extract_json("{invalid json}") is None


class TestExtractXml:
    """Tests for XML extraction function"""

    def test_extract_xml_simple(self):
        """Extracts data from simple XML"""
        text = '<sentiment>positive</sentiment><confidence>95</confidence>'
        fields = [
            OutputField(name="sentiment", field_type="text"),
            OutputField(name="confidence", field_type="number")
        ]
        result = extract_xml(text, fields)
        assert result == {"sentiment": "positive", "confidence": "95"}

    def test_extract_xml_with_root(self):
        """Extracts data from XML with root element"""
        text = '<root><sentiment>negative</sentiment></root>'
        fields = [OutputField(name="sentiment", field_type="text")]
        result = extract_xml(text, fields)
        assert result == {"sentiment": "negative"}

    def test_extract_xml_empty(self):
        """Returns None for empty input"""
        fields = [OutputField(name="test", field_type="text")]
        assert extract_xml("", fields) is None
        assert extract_xml(None, fields) is None

    def test_extract_xml_invalid(self):
        """Returns None for invalid XML"""
        fields = [OutputField(name="test", field_type="text")]
        assert extract_xml("not xml", fields) is None


class TestExtractCsvRow:
    """Tests for CSV row extraction function"""

    def test_extract_csv_simple(self):
        """Extracts data from simple CSV row"""
        text = 'positive,95,technology'
        fields = [
            OutputField(name="sentiment", field_type="text"),
            OutputField(name="confidence", field_type="number"),
            OutputField(name="category", field_type="text")
        ]
        result = extract_csv_row(text, fields)
        assert result == {"sentiment": "positive", "confidence": "95", "category": "technology"}

    def test_extract_csv_quoted(self):
        """Extracts data from CSV with quoted values"""
        text = '"positive","high confidence","tech, news"'
        fields = [
            OutputField(name="sentiment", field_type="text"),
            OutputField(name="note", field_type="text"),
            OutputField(name="category", field_type="text")
        ]
        result = extract_csv_row(text, fields)
        assert result["sentiment"] == "positive"
        assert result["category"] == "tech, news"

    def test_extract_csv_with_code_block(self):
        """Extracts CSV from markdown code block"""
        text = '```csv\npositive,90\n```'
        fields = [
            OutputField(name="sentiment", field_type="text"),
            OutputField(name="confidence", field_type="number")
        ]
        result = extract_csv_row(text, fields)
        assert result == {"sentiment": "positive", "confidence": "90"}

    def test_extract_csv_empty(self):
        """Returns None for empty input"""
        fields = [OutputField(name="test", field_type="text")]
        assert extract_csv_row("", fields) is None
        assert extract_csv_row(None, fields) is None


class TestValidateOutput:
    """Tests for output validation function"""

    def test_validate_all_required_present(self):
        """Validates successfully when all required fields present"""
        data = {"sentiment": "positive", "confidence": 90}
        fields = [
            OutputField(name="sentiment", field_type="text", required=True),
            OutputField(name="confidence", field_type="number", required=True)
        ]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert errors == []

    def test_validate_missing_required(self):
        """Fails validation when required field missing"""
        data = {"sentiment": "positive"}
        fields = [
            OutputField(name="sentiment", field_type="text", required=True),
            OutputField(name="confidence", field_type="number", required=True)
        ]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is False
        assert any("confidence" in e for e in errors)

    def test_validate_missing_optional(self):
        """Passes validation when optional field missing"""
        data = {"sentiment": "positive"}
        fields = [
            OutputField(name="sentiment", field_type="text", required=True),
            OutputField(name="notes", field_type="text", required=False)
        ]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True

    def test_validate_default_applied(self):
        """Default value applied when required field missing"""
        data = {"sentiment": "positive"}
        fields = [
            OutputField(name="sentiment", field_type="text", required=True),
            OutputField(name="confidence", field_type="number", required=True, default="50")
        ]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert data["confidence"] == "50"

    def test_validate_number_type(self):
        """Number fields are converted correctly"""
        data = {"count": "42"}
        fields = [OutputField(name="count", field_type="number", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert data["count"] == 42

    def test_validate_decimal_type(self):
        """Decimal fields are converted correctly"""
        data = {"score": "3.14"}
        fields = [OutputField(name="score", field_type="decimal", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert data["score"] == 3.14

    def test_validate_boolean_type(self):
        """Boolean fields are converted correctly"""
        data = {"is_valid": "true"}
        fields = [OutputField(name="is_valid", field_type="boolean", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert data["is_valid"] is True

    def test_validate_list_type_json(self):
        """List fields parse JSON arrays"""
        data = {"tags": '["tech", "news"]'}
        fields = [OutputField(name="tags", field_type="list", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert data["tags"] == ["tech", "news"]

    def test_validate_list_type_csv(self):
        """List fields parse comma-separated values"""
        data = {"tags": "tech, news, science"}
        fields = [OutputField(name="tags", field_type="list", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
        assert data["tags"] == ["tech", "news", "science"]

    def test_validate_constraints_one_of(self):
        """Validates against 'one of' constraints"""
        fields = [
            OutputField(
                name="sentiment",
                field_type="text",
                required=True,
                constraints="one of: positive, negative, neutral"
            )
        ]

        # Valid value
        data = {"sentiment": "positive"}
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True

        # Invalid value
        data = {"sentiment": "unknown"}
        is_valid, errors = validate_output(data, fields)
        assert is_valid is False
        assert any("sentiment" in e for e in errors)


class TestBuildSystemPrompt:
    """Tests for system prompt generation"""

    def test_build_prompt_basic(self):
        """Builds basic system prompt"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="json"
        )
        prompt = build_system_prompt(config)

        assert "TASK: Classify sentiment" in prompt
        assert "OUTPUT SCHEMA:" in prompt
        assert "sentiment: text, required" in prompt
        assert "OUTPUT FORMAT: JSON" in prompt
        assert "CRITICAL RULES:" in prompt

    def test_build_prompt_with_constraints(self):
        """Includes constraints in prompt"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[
                OutputField(
                    name="sentiment",
                    field_type="text",
                    constraints="one of: positive, negative, neutral"
                )
            ],
            output_format="json"
        )
        prompt = build_system_prompt(config)

        assert "one of: positive, negative, neutral" in prompt

    def test_build_prompt_with_few_shot(self):
        """Includes few-shot examples in prompt"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="json",
            few_shot_examples=[
                FewShotExample(
                    input_data={"text": "I love this!"},
                    output_data={"sentiment": "positive"}
                )
            ]
        )
        prompt = build_system_prompt(config)

        assert "EXAMPLES:" in prompt
        assert "Example 1:" in prompt
        assert "I love this!" in prompt
        assert "positive" in prompt

    def test_build_prompt_with_confidence(self):
        """Includes confidence instruction when enabled"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="json",
            confidence_enabled=True
        )
        prompt = build_system_prompt(config)

        assert "CONFIDENCE:" in prompt
        assert "0-100" in prompt

    def test_build_prompt_with_consistency(self):
        """Includes consistency rules when provided"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="json",
            consistency_prompt="Always use lowercase labels"
        )
        prompt = build_system_prompt(config)

        assert "CONSISTENCY RULES:" in prompt
        assert "Always use lowercase labels" in prompt

    def test_build_prompt_override(self):
        """Uses override when provided"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="json",
            system_prompt_override="Custom system prompt here"
        )
        prompt = build_system_prompt(config)

        assert prompt == "Custom system prompt here"

    def test_build_prompt_csv_format(self):
        """Builds correct prompt for CSV output format"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[
                OutputField(name="sentiment", field_type="text"),
                OutputField(name="confidence", field_type="number")
            ],
            output_format="csv"
        )
        prompt = build_system_prompt(config)

        assert "OUTPUT FORMAT: CSV" in prompt
        assert "sentiment, confidence" in prompt
        assert "CSV row" in prompt

    def test_build_prompt_xml_format(self):
        """Builds correct prompt for XML output format"""
        config = AutomatorConfig(
            task_description="Classify sentiment",
            input_columns=["text"],
            output_fields=[OutputField(name="sentiment", field_type="text")],
            output_format="xml"
        )
        prompt = build_system_prompt(config)

        assert "OUTPUT FORMAT: XML" in prompt
        assert "XML" in prompt


class TestAutomatorTool:
    """Tests for AutomatorTool class"""

    def test_tool_metadata(self):
        """Tool has correct metadata"""
        tool = AutomatorTool()
        assert tool.id == "automator"
        assert tool.name == "General Purpose Automator"
        assert tool.category == "Processing"
        assert tool.icon == ":material/precision_manufacturing:"

    def test_tool_info(self):
        """Tool info returns correct structure"""
        tool = AutomatorTool()
        info = tool.get_info()
        assert info["id"] == "automator"
        assert info["name"] == "General Purpose Automator"
        assert info["category"] == "Processing"


class TestSampleData:
    """Tests for sample automator data"""

    def test_sample_data_structure(self):
        """Sample data has expected structure"""
        assert "text" in SAMPLE_AUTOMATOR_DATA
        assert "source" in SAMPLE_AUTOMATOR_DATA
        assert "date" in SAMPLE_AUTOMATOR_DATA

    def test_sample_data_lengths_match(self):
        """All columns in sample data have same length"""
        lengths = [len(v) for v in SAMPLE_AUTOMATOR_DATA.values()]
        assert all(l == lengths[0] for l in lengths)

    def test_sample_data_not_empty(self):
        """Sample data contains entries"""
        assert len(SAMPLE_AUTOMATOR_DATA["text"]) > 0
        assert len(SAMPLE_AUTOMATOR_DATA["text"]) >= 10


class TestEdgeCases:
    """Tests for edge case handling"""

    def test_validate_empty_data(self):
        """Handles empty data dict"""
        data = {}
        fields = [OutputField(name="test", field_type="text", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is False

    def test_validate_none_value(self):
        """Handles None values"""
        data = {"test": None}
        fields = [OutputField(name="test", field_type="text", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is False

    def test_extract_json_nested(self):
        """Extracts nested JSON correctly"""
        text = '{"result": {"sentiment": "positive", "entities": ["Apple", "iPhone"]}}'
        result = extract_json(text)
        assert result["result"]["sentiment"] == "positive"
        assert "Apple" in result["result"]["entities"]

    def test_validate_invalid_number(self):
        """Handles invalid number conversion"""
        data = {"count": "not a number"}
        fields = [OutputField(name="count", field_type="number", required=True)]
        is_valid, errors = validate_output(data, fields)
        assert is_valid is False
        assert any("count" in e for e in errors)

    def test_validate_case_insensitive_constraints(self):
        """Constraint validation is case insensitive"""
        fields = [
            OutputField(
                name="sentiment",
                field_type="text",
                required=True,
                constraints="one of: Positive, Negative, Neutral"
            )
        ]

        # Lowercase should match
        data = {"sentiment": "positive"}
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True

        # Uppercase should match
        data = {"sentiment": "POSITIVE"}
        is_valid, errors = validate_output(data, fields)
        assert is_valid is True
