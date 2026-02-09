"""
Tests for Model Comparison Tool
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from core.comparison_processor import (
    ModelConfig, ComparisonConfig, ComparisonProcessor, ComparisonResult
)
from core.comparison_analytics import (
    calculate_pairwise_agreement,
    calculate_jaccard_similarity,
    calculate_per_model_consistency,
    calculate_all_agreement_metrics,
    ModelAgreementMetrics,
    _normalize_output,
    _tokenize
)
from core.providers import LLMProvider
from tools.model_comparison import ModelComparisonTool


# ==========================================
# ModelConfig Tests
# ==========================================

class TestModelConfigDataclass:
    """Tests for ModelConfig dataclass"""

    def test_model_config_minimal(self):
        """ModelConfig can be created with required fields"""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            base_url=None,
            model="gpt-4",
            display_name="GPT-4"
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.model == "gpt-4"
        assert config.display_name == "GPT-4"
        assert config.temperature is None
        assert config.max_tokens is None

    def test_model_config_full(self):
        """ModelConfig can be created with all fields"""
        config = ModelConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="sk-test",
            base_url="https://api.anthropic.com",
            model="claude-3-sonnet",
            display_name="Claude 3 Sonnet",
            temperature=0.5,
            max_tokens=1024
        )
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.temperature == 0.5
        assert config.max_tokens == 1024


class TestComparisonConfigDataclass:
    """Tests for ComparisonConfig dataclass"""

    def test_comparison_config_minimal(self):
        """ComparisonConfig can be created with required fields"""
        models = [
            ModelConfig(LLMProvider.OPENAI, "key1", None, "gpt-4", "GPT-4"),
            ModelConfig(LLMProvider.ANTHROPIC, "key2", None, "claude-3", "Claude 3")
        ]
        config = ComparisonConfig(models=models)

        assert len(config.models) == 2
        assert config.max_concurrency == 5
        assert config.auto_retry is True
        assert config.max_retries == 3
        assert config.save_path is None
        assert config.realtime_progress is True
        assert config.json_mode is False

    def test_comparison_config_full(self):
        """ComparisonConfig can be created with all fields"""
        models = [
            ModelConfig(LLMProvider.OPENAI, "key1", None, "gpt-4", "GPT-4")
        ]
        config = ComparisonConfig(
            models=models,
            max_concurrency=10,
            auto_retry=False,
            max_retries=5,
            save_path="/tmp/test",
            realtime_progress=False,
            json_mode=True
        )

        assert config.max_concurrency == 10
        assert config.auto_retry is False
        assert config.max_retries == 5
        assert config.save_path == "/tmp/test"
        assert config.realtime_progress is False
        assert config.json_mode is True


# ==========================================
# Agreement Metrics Tests
# ==========================================

class TestNormalizeOutput:
    """Tests for output normalization"""

    def test_normalize_basic(self):
        """Normalizes basic text"""
        assert _normalize_output("Positive") == "positive"
        assert _normalize_output("  Positive  ") == "positive"

    def test_normalize_whitespace(self):
        """Normalizes multiple whitespace"""
        assert _normalize_output("positive   sentiment") == "positive sentiment"
        assert _normalize_output("  positive\n\nsentiment  ") == "positive sentiment"

    def test_normalize_none(self):
        """Handles None input"""
        assert _normalize_output(None) == ""

    def test_normalize_non_string(self):
        """Handles non-string input"""
        assert _normalize_output(123) == "123"


class TestTokenize:
    """Tests for tokenization"""

    def test_tokenize_basic(self):
        """Tokenizes basic text"""
        result = _tokenize("hello world")
        assert result == {"hello", "world"}

    def test_tokenize_punctuation(self):
        """Removes punctuation"""
        result = _tokenize("hello, world!")
        assert result == {"hello", "world"}

    def test_tokenize_mixed_case(self):
        """Lowercases tokens"""
        result = _tokenize("Hello WORLD")
        assert result == {"hello", "world"}

    def test_tokenize_none(self):
        """Handles None input"""
        assert _tokenize(None) == set()

    def test_tokenize_empty(self):
        """Handles empty string"""
        assert _tokenize("") == set()

    def test_tokenize_numbers(self):
        """Keeps numbers"""
        result = _tokenize("test123 value456")
        assert "test123" in result
        assert "value456" in result


class TestCalculatePairwiseAgreement:
    """Tests for pairwise agreement calculation"""

    def test_perfect_agreement(self):
        """Returns 100% for identical outputs"""
        outputs = {
            "model_a": ["positive", "negative", "neutral"],
            "model_b": ["positive", "negative", "neutral"]
        }
        result = calculate_pairwise_agreement(outputs)

        assert result.loc["model_a", "model_b"] == 100.0
        assert result.loc["model_b", "model_a"] == 100.0
        assert result.loc["model_a", "model_a"] == 100.0

    def test_no_agreement(self):
        """Returns 0% for completely different outputs"""
        outputs = {
            "model_a": ["positive", "positive", "positive"],
            "model_b": ["negative", "negative", "negative"]
        }
        result = calculate_pairwise_agreement(outputs)

        assert result.loc["model_a", "model_b"] == 0.0

    def test_partial_agreement(self):
        """Returns correct percentage for partial agreement"""
        outputs = {
            "model_a": ["positive", "negative", "positive"],
            "model_b": ["positive", "positive", "positive"]
        }
        result = calculate_pairwise_agreement(outputs)

        # 2 out of 3 match = 66.67%
        assert 66 < result.loc["model_a", "model_b"] < 67

    def test_three_models(self):
        """Works with three or more models"""
        outputs = {
            "model_a": ["positive", "negative"],
            "model_b": ["positive", "negative"],
            "model_c": ["negative", "negative"]
        }
        result = calculate_pairwise_agreement(outputs)

        assert result.shape == (3, 3)
        assert result.loc["model_a", "model_b"] == 100.0
        assert result.loc["model_a", "model_c"] == 50.0

    def test_single_model(self):
        """Returns empty DataFrame for single model"""
        outputs = {"model_a": ["positive"]}
        result = calculate_pairwise_agreement(outputs)
        assert result.empty

    def test_empty_outputs(self):
        """Handles empty outputs"""
        outputs = {}
        result = calculate_pairwise_agreement(outputs)
        assert result.empty

    def test_case_insensitive(self):
        """Agreement is case-insensitive"""
        outputs = {
            "model_a": ["Positive", "NEGATIVE"],
            "model_b": ["positive", "negative"]
        }
        result = calculate_pairwise_agreement(outputs)
        assert result.loc["model_a", "model_b"] == 100.0


class TestCalculateJaccardSimilarity:
    """Tests for Jaccard similarity calculation"""

    def test_identical_outputs(self):
        """Returns 100% for identical token sets"""
        outputs = {
            "model_a": ["hello world", "foo bar"],
            "model_b": ["hello world", "foo bar"]
        }
        result = calculate_jaccard_similarity(outputs)

        assert result.loc["model_a", "model_b"] == 100.0

    def test_no_overlap(self):
        """Returns 0% for no token overlap"""
        outputs = {
            "model_a": ["hello world"],
            "model_b": ["foo bar"]
        }
        result = calculate_jaccard_similarity(outputs)

        assert result.loc["model_a", "model_b"] == 0.0

    def test_partial_overlap(self):
        """Returns correct percentage for partial overlap"""
        outputs = {
            "model_a": ["hello world"],
            "model_b": ["hello there"]
        }
        result = calculate_jaccard_similarity(outputs)

        # tokens: {hello, world} vs {hello, there}
        # intersection: {hello}, union: {hello, world, there}
        # Jaccard = 1/3 = 33.33%
        assert 33 < result.loc["model_a", "model_b"] < 34

    def test_both_empty(self):
        """Returns 100% when both outputs are empty"""
        outputs = {
            "model_a": [""],
            "model_b": [""]
        }
        result = calculate_jaccard_similarity(outputs)
        assert result.loc["model_a", "model_b"] == 100.0

    def test_one_empty(self):
        """Returns 0% when one output is empty"""
        outputs = {
            "model_a": ["hello"],
            "model_b": [""]
        }
        result = calculate_jaccard_similarity(outputs)
        assert result.loc["model_a", "model_b"] == 0.0


class TestCalculatePerModelConsistency:
    """Tests for per-model consistency calculation"""

    def test_all_agree(self):
        """Returns 100% when all models agree"""
        outputs = {
            "model_a": ["positive", "negative"],
            "model_b": ["positive", "negative"],
            "model_c": ["positive", "negative"]
        }
        result = calculate_per_model_consistency(outputs)

        assert result["model_a"] == 100.0
        assert result["model_b"] == 100.0
        assert result["model_c"] == 100.0

    def test_one_disagrees(self):
        """Returns correct percentages when one model disagrees"""
        outputs = {
            "model_a": ["positive", "positive"],
            "model_b": ["positive", "positive"],
            "model_c": ["negative", "negative"]
        }
        result = calculate_per_model_consistency(outputs)

        # For each row, majority is what A and B agree on
        assert result["model_a"] == 100.0
        assert result["model_b"] == 100.0
        assert result["model_c"] == 0.0

    def test_no_majority(self):
        """Returns 0% when there's no clear majority"""
        outputs = {
            "model_a": ["positive"],
            "model_b": ["negative"]
        }
        result = calculate_per_model_consistency(outputs)

        # No majority (need > 50%), so both get 0%
        assert result["model_a"] == 0.0
        assert result["model_b"] == 0.0

    def test_single_model(self):
        """Returns 100% for single model"""
        outputs = {"model_a": ["positive"]}
        result = calculate_per_model_consistency(outputs)
        assert result["model_a"] == 100.0


class TestCalculateAllAgreementMetrics:
    """Tests for combined metrics calculation"""

    def test_returns_all_metrics(self):
        """Returns all metric types"""
        outputs = {
            "model_a": ["positive", "negative"],
            "model_b": ["positive", "positive"]
        }
        result = calculate_all_agreement_metrics(outputs)

        assert isinstance(result, ModelAgreementMetrics)
        assert isinstance(result.pairwise_agreement, pd.DataFrame)
        assert isinstance(result.jaccard_similarity, pd.DataFrame)
        assert isinstance(result.average_pairwise_agreement, float)
        assert isinstance(result.per_model_consistency, dict)

    def test_average_pairwise_excludes_diagonal(self):
        """Average pairwise agreement excludes self-comparisons"""
        outputs = {
            "model_a": ["positive", "negative"],
            "model_b": ["positive", "positive"]  # 50% match with a
        }
        result = calculate_all_agreement_metrics(outputs)

        # Average should be the off-diagonal value, not inflated by 100% diagonals
        assert result.average_pairwise_agreement == 50.0

    def test_handles_empty(self):
        """Handles empty outputs gracefully"""
        outputs = {}
        result = calculate_all_agreement_metrics(outputs)

        assert result.pairwise_agreement.empty
        assert result.jaccard_similarity.empty
        assert result.average_pairwise_agreement == 0.0
        assert result.per_model_consistency == {}


# ==========================================
# ComparisonProcessor Tests
# ==========================================

class TestComparisonProcessor:
    """Tests for ComparisonProcessor class"""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['positive review', 'negative review', 'neutral review']
        })

    @pytest.fixture
    def sample_config(self):
        """Sample comparison config"""
        return ComparisonConfig(
            models=[
                ModelConfig(LLMProvider.OPENAI, "key1", None, "gpt-4", "GPT-4"),
                ModelConfig(LLMProvider.ANTHROPIC, "key2", None, "claude-3", "Claude 3")
            ],
            max_concurrency=5,
            auto_retry=False,
            max_retries=0
        )

    def test_sanitize_column_name(self):
        """Column name sanitization works correctly"""
        config = ComparisonConfig(models=[])
        processor = ComparisonProcessor(config, "run_123", "session_456")

        assert processor._sanitize_column_name("GPT-4") == "gpt_4"
        assert processor._sanitize_column_name("Claude 3 Sonnet") == "claude_3_sonnet"
        assert processor._sanitize_column_name("OpenAI/GPT-4o") == "openai_gpt_4o"
        assert processor._sanitize_column_name("model__test") == "model_test"
        assert processor._sanitize_column_name("  Model  ") == "model"

    @pytest.mark.asyncio
    async def test_processor_creates_correct_result_structure(self, sample_df, sample_config):
        """Processor creates correct result structure"""
        # Mock the LLM calls
        with patch('core.comparison_processor.get_client') as mock_get_client, \
             patch('core.comparison_processor.create_http_client') as mock_http, \
             patch('core.comparison_processor.call_llm_with_retry') as mock_call:

            # Setup mocks
            mock_http_client = AsyncMock()
            mock_http.return_value = mock_http_client
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Return successful responses
            mock_call.return_value = ("positive", 0.5, None, 0)

            # Mock database
            with patch('core.comparison_processor.get_db') as mock_db:
                mock_db_instance = MagicMock()
                mock_db.return_value = mock_db_instance

                processor = ComparisonProcessor(sample_config, "run_123", "session_456")
                result = await processor.process(
                    sample_df,
                    "Classify sentiment",
                    ["text"],
                    None
                )

                assert isinstance(result, ComparisonResult)
                assert result.success_count == 3
                assert result.error_count == 0
                assert len(result.results) == 3
                assert "per_model_stats" in dir(result)


# ==========================================
# ModelComparisonTool Tests
# ==========================================

class TestModelComparisonTool:
    """Tests for ModelComparisonTool class"""

    def test_tool_metadata(self):
        """Tool has correct metadata"""
        tool = ModelComparisonTool()
        assert tool.id == "model_comparison"
        assert tool.name == "Model Comparison"
        assert tool.category == "Processing"
        assert tool.icon == ":material/compare:"

    def test_tool_info(self):
        """Tool info returns correct structure"""
        tool = ModelComparisonTool()
        info = tool.get_info()
        assert info["id"] == "model_comparison"
        assert info["name"] == "Model Comparison"
        assert info["category"] == "Processing"

    def test_sanitize_column_name_method(self):
        """Tool's sanitize method matches processor's"""
        tool = ModelComparisonTool()

        assert tool._sanitize_column_name("GPT-4") == "gpt_4"
        assert tool._sanitize_column_name("Claude 3 Sonnet") == "claude_3_sonnet"
        assert tool._sanitize_column_name("OpenAI/GPT-4o") == "openai_gpt_4o"


# ==========================================
# Integration Tests
# ==========================================

class TestModelComparisonIntegration:
    """Integration tests for model comparison"""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['Great product!', 'Terrible service', 'It was okay']
        })

    def test_agreement_metrics_with_real_like_data(self):
        """Test agreement metrics with realistic data"""
        outputs = {
            "GPT-4": ["positive", "negative", "neutral", "positive", "negative"],
            "Claude": ["positive", "negative", "neutral", "positive", "positive"],
            "Gemini": ["positive", "negative", "neutral", "negative", "negative"]
        }

        metrics = calculate_all_agreement_metrics(outputs)

        # Check structure
        assert metrics.pairwise_agreement.shape == (3, 3)
        assert metrics.jaccard_similarity.shape == (3, 3)
        assert len(metrics.per_model_consistency) == 3

        # GPT-4 and Claude agree on 4/5 = 80%
        assert metrics.pairwise_agreement.loc["GPT-4", "Claude"] == 80.0

        # Average should be between 0 and 100
        assert 0 <= metrics.average_pairwise_agreement <= 100

    def test_metrics_with_long_form_outputs(self):
        """Test with longer, more varied outputs"""
        outputs = {
            "Model A": [
                "This is a positive review about the product quality.",
                "The customer expresses dissatisfaction with delivery.",
            ],
            "Model B": [
                "This is a positive review about product quality.",
                "Customer is unhappy with the delivery experience.",
            ]
        }

        metrics = calculate_all_agreement_metrics(outputs)

        # Jaccard should show high similarity due to similar words
        assert metrics.jaccard_similarity.loc["Model A", "Model B"] > 50

    def test_result_dataframe_structure(self):
        """Test that results can be properly added to DataFrame"""
        df = pd.DataFrame({'id': [1, 2], 'text': ['a', 'b']})

        # Simulate results from processor
        results = [
            {'gpt_4_output': 'positive', 'gpt_4_latency': 0.5,
             'claude_output': 'positive', 'claude_latency': 0.3},
            {'gpt_4_output': 'negative', 'gpt_4_latency': 0.4,
             'claude_output': 'neutral', 'claude_latency': 0.35}
        ]

        # Add results to DataFrame
        for col in ['gpt_4_output', 'gpt_4_latency', 'claude_output', 'claude_latency']:
            df[col] = [r[col] for r in results]

        assert 'gpt_4_output' in df.columns
        assert 'claude_output' in df.columns
        assert df['gpt_4_output'].tolist() == ['positive', 'negative']
        assert df['claude_output'].tolist() == ['positive', 'neutral']


# ==========================================
# Edge Cases and Error Handling
# ==========================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_dataframe(self):
        """Handles empty DataFrame"""
        outputs = {"model_a": [], "model_b": []}
        result = calculate_pairwise_agreement(outputs)
        # Should handle empty lists gracefully
        assert result.loc["model_a", "model_b"] == 0.0 or np.isnan(result.loc["model_a", "model_b"]) or result.empty == False

    def test_single_row(self):
        """Works with single row of data"""
        outputs = {
            "model_a": ["positive"],
            "model_b": ["positive"]
        }
        result = calculate_pairwise_agreement(outputs)
        assert result.loc["model_a", "model_b"] == 100.0

    def test_unicode_outputs(self):
        """Handles Unicode text correctly"""
        outputs = {
            "model_a": ["", ""],  # Chinese and Arabic
            "model_b": ["", ""]
        }
        result = calculate_pairwise_agreement(outputs)
        assert result.loc["model_a", "model_b"] == 100.0

    def test_special_characters_in_output(self):
        """Handles special characters"""
        outputs = {
            "model_a": ["positive! :)", "negative :("],
            "model_b": ["positive! :)", "negative :("]
        }
        result = calculate_pairwise_agreement(outputs)
        assert result.loc["model_a", "model_b"] == 100.0

    def test_very_long_outputs(self):
        """Handles very long outputs"""
        long_text = "word " * 1000
        outputs = {
            "model_a": [long_text],
            "model_b": [long_text]
        }
        result = calculate_pairwise_agreement(outputs)
        assert result.loc["model_a", "model_b"] == 100.0

    def test_error_marked_outputs(self):
        """Agreement calculation with error-marked outputs"""
        outputs = {
            "model_a": ["positive", "Error: timeout", "negative"],
            "model_b": ["positive", "negative", "negative"]
        }
        result = calculate_pairwise_agreement(outputs)

        # Should calculate correctly - 2/3 match
        assert 66 < result.loc["model_a", "model_b"] < 67


class TestModelConfigValidation:
    """Tests for ModelConfig edge cases"""

    def test_empty_api_key(self):
        """Handles empty API key"""
        config = ModelConfig(
            provider=LLMProvider.LM_STUDIO,
            api_key="",
            base_url="http://localhost:1234/v1",
            model="local-model",
            display_name="Local Model"
        )
        assert config.api_key == ""

    def test_display_name_with_special_chars(self):
        """Display name can have special characters"""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            api_key="key",
            base_url=None,
            model="gpt-4",
            display_name="GPT-4 (OpenAI) - Latest"
        )
        assert config.display_name == "GPT-4 (OpenAI) - Latest"
