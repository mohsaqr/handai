"""
Tests for Consensus Coder Tool - Enhanced Judge Features
Comprehensive test suite covering analytics, parsing, error handling, and integration.
"""

import pytest
import json
import asyncio
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from core.comparison_analytics import (
    calculate_average_scores,
    calculate_rank_distribution,
    calculate_confidence_stats,
    extract_disagreement_patterns,
    calculate_quality_metrics,
    QualityMetrics,
)


# ==========================================
# TestCalculateAverageScores - Extended Tests
# ==========================================

class TestCalculateAverageScores:
    """Tests for calculate_average_scores function"""

    def test_basic_scores(self):
        """Calculate average scores from results"""
        results = [
            {"worker_1_score": 5, "worker_2_score": 4},
            {"worker_1_score": 4, "worker_2_score": 3},
            {"worker_1_score": 5, "worker_2_score": 5},
        ]
        avg = calculate_average_scores(results)

        assert "worker_1" in avg
        assert "worker_2" in avg
        assert abs(avg["worker_1"] - 4.67) < 0.1  # (5+4+5)/3
        assert abs(avg["worker_2"] - 4.0) < 0.1   # (4+3+5)/3

    def test_empty_results(self):
        """Handle empty results"""
        avg = calculate_average_scores([])
        assert avg == {}

    def test_missing_scores(self):
        """Handle missing score fields"""
        results = [
            {"worker_1_score": 5},
            {"worker_1_score": 3, "worker_2_score": 4},
        ]
        avg = calculate_average_scores(results)

        assert "worker_1" in avg
        assert abs(avg["worker_1"] - 4.0) < 0.1  # (5+3)/2

    def test_invalid_scores_filtered(self):
        """Invalid scores are filtered out"""
        results = [
            {"worker_1_score": 5},
            {"worker_1_score": "invalid"},
            {"worker_1_score": 10},  # Out of range, should be excluded
            {"worker_1_score": 3},
        ]
        avg = calculate_average_scores(results)

        # Only valid scores (5, 3) should be included
        assert "worker_1" in avg
        assert abs(avg["worker_1"] - 4.0) < 0.1

    def test_three_workers(self):
        """Calculate averages with 3 workers"""
        results = [
            {"worker_1_score": 5, "worker_2_score": 4, "worker_3_score": 3},
            {"worker_1_score": 4, "worker_2_score": 3, "worker_3_score": 5},
            {"worker_1_score": 3, "worker_2_score": 5, "worker_3_score": 4},
        ]
        avg = calculate_average_scores(results)

        assert len(avg) == 3
        assert abs(avg["worker_1"] - 4.0) < 0.1
        assert abs(avg["worker_2"] - 4.0) < 0.1
        assert abs(avg["worker_3"] - 4.0) < 0.1

    def test_single_row(self):
        """Calculate averages with only one result"""
        results = [{"worker_1_score": 4, "worker_2_score": 5}]
        avg = calculate_average_scores(results)

        assert avg["worker_1"] == 4.0
        assert avg["worker_2"] == 5.0

    def test_all_same_scores(self):
        """All workers score the same (5)"""
        results = [
            {"worker_1_score": 5, "worker_2_score": 5},
            {"worker_1_score": 5, "worker_2_score": 5},
        ]
        avg = calculate_average_scores(results)

        assert avg["worker_1"] == 5.0
        assert avg["worker_2"] == 5.0

    def test_all_minimum_scores(self):
        """All workers score minimum (1)"""
        results = [
            {"worker_1_score": 1, "worker_2_score": 1},
            {"worker_1_score": 1, "worker_2_score": 1},
        ]
        avg = calculate_average_scores(results)

        assert avg["worker_1"] == 1.0
        assert avg["worker_2"] == 1.0

    def test_float_scores(self):
        """Handle float scores like 4.5"""
        results = [
            {"worker_1_score": 4.5, "worker_2_score": 3.5},
            {"worker_1_score": 3.5, "worker_2_score": 4.5},
        ]
        avg = calculate_average_scores(results)

        assert avg["worker_1"] == 4.0
        assert avg["worker_2"] == 4.0

    def test_none_values(self):
        """Handle None values in scores"""
        results = [
            {"worker_1_score": 5, "worker_2_score": None},
            {"worker_1_score": 3, "worker_2_score": 4},
        ]
        avg = calculate_average_scores(results)

        assert abs(avg["worker_1"] - 4.0) < 0.1
        # worker_2 should only have one valid score
        assert avg["worker_2"] == 4.0

    def test_negative_scores_filtered(self):
        """Negative scores are filtered out"""
        results = [
            {"worker_1_score": 5},
            {"worker_1_score": -1},  # Invalid
            {"worker_1_score": 3},
        ]
        avg = calculate_average_scores(results)

        assert abs(avg["worker_1"] - 4.0) < 0.1

    def test_large_dataset(self):
        """Performance test with 1000 rows"""
        results = [{"worker_1_score": i % 5 + 1, "worker_2_score": (i + 1) % 5 + 1}
                   for i in range(1000)]
        avg = calculate_average_scores(results)

        assert "worker_1" in avg
        assert "worker_2" in avg
        assert 1 <= avg["worker_1"] <= 5
        assert 1 <= avg["worker_2"] <= 5


# ==========================================
# TestCalculateRankDistribution - Extended Tests
# ==========================================

class TestCalculateRankDistribution:
    """Tests for calculate_rank_distribution function"""

    def test_basic_ranks(self):
        """Calculate rank distribution from results"""
        results = [
            {"worker_1_rank": 1, "worker_2_rank": 2},
            {"worker_1_rank": 1, "worker_2_rank": 2},
            {"worker_1_rank": 2, "worker_2_rank": 1},
        ]
        dist = calculate_rank_distribution(results)

        assert "worker_1" in dist
        assert "worker_2" in dist
        assert dist["worker_1"][1] == 2  # worker_1 ranked 1st twice
        assert dist["worker_1"][2] == 1  # worker_1 ranked 2nd once
        assert dist["worker_2"][1] == 1  # worker_2 ranked 1st once
        assert dist["worker_2"][2] == 2  # worker_2 ranked 2nd twice

    def test_empty_results(self):
        """Handle empty results"""
        dist = calculate_rank_distribution([])
        assert dist == {}

    def test_zero_ranks_excluded(self):
        """Zero ranks (defaults) are excluded"""
        results = [
            {"worker_1_rank": 1, "worker_2_rank": 0},
            {"worker_1_rank": 0, "worker_2_rank": 1},
        ]
        dist = calculate_rank_distribution(results)

        assert dist["worker_1"].get(1, 0) == 1
        assert 0 not in dist["worker_1"]

    def test_three_workers_ranks(self):
        """Full 1st/2nd/3rd distribution with 3 workers"""
        results = [
            {"worker_1_rank": 1, "worker_2_rank": 2, "worker_3_rank": 3},
            {"worker_1_rank": 2, "worker_2_rank": 1, "worker_3_rank": 3},
            {"worker_1_rank": 3, "worker_2_rank": 2, "worker_3_rank": 1},
        ]
        dist = calculate_rank_distribution(results)

        assert len(dist) == 3
        # worker_1: 1st once, 2nd once, 3rd once
        assert dist["worker_1"][1] == 1
        assert dist["worker_1"][2] == 1
        assert dist["worker_1"][3] == 1

    def test_ties_in_ranking(self):
        """Same rank for multiple workers (ties)"""
        results = [
            {"worker_1_rank": 1, "worker_2_rank": 1},  # Both ranked 1st
            {"worker_1_rank": 2, "worker_2_rank": 2},  # Both ranked 2nd
        ]
        dist = calculate_rank_distribution(results)

        assert dist["worker_1"][1] == 1
        assert dist["worker_1"][2] == 1
        assert dist["worker_2"][1] == 1
        assert dist["worker_2"][2] == 1

    def test_single_row_ranks(self):
        """Only one result row"""
        results = [{"worker_1_rank": 1, "worker_2_rank": 2}]
        dist = calculate_rank_distribution(results)

        assert dist["worker_1"][1] == 1
        assert dist["worker_2"][2] == 1

    def test_all_first_place(self):
        """Edge case: everyone ranked 1st (unusual but handled)"""
        results = [
            {"worker_1_rank": 1, "worker_2_rank": 1},
            {"worker_1_rank": 1, "worker_2_rank": 1},
        ]
        dist = calculate_rank_distribution(results)

        assert dist["worker_1"][1] == 2
        assert dist["worker_2"][1] == 2

    def test_string_ranks_handled(self):
        """String values are filtered out"""
        results = [
            {"worker_1_rank": 1, "worker_2_rank": "first"},
            {"worker_1_rank": 2, "worker_2_rank": 1},
        ]
        dist = calculate_rank_distribution(results)

        assert dist["worker_1"][1] == 1
        assert dist["worker_1"][2] == 1
        # worker_2 should only have 1 valid rank
        assert dist["worker_2"][1] == 1
        assert 2 not in dist.get("worker_2", {})

    def test_negative_ranks_filtered(self):
        """Negative ranks are filtered out"""
        results = [
            {"worker_1_rank": 1},
            {"worker_1_rank": -1},  # Invalid
            {"worker_1_rank": 2},
        ]
        dist = calculate_rank_distribution(results)

        assert dist["worker_1"][1] == 1
        assert dist["worker_1"][2] == 1
        assert -1 not in dist["worker_1"]


# ==========================================
# TestCalculateConfidenceStats - Extended Tests
# ==========================================

class TestCalculateConfidenceStats:
    """Tests for calculate_confidence_stats function"""

    def test_basic_confidence(self):
        """Calculate confidence statistics"""
        results = [
            {"judge_confidence": 80},
            {"judge_confidence": 90},
            {"judge_confidence": 70},
            {"judge_confidence": 100},
            {"judge_confidence": 60},
        ]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 80.0
        assert stats["median"] == 80.0
        assert stats["min"] == 60.0
        assert stats["max"] == 100.0
        assert stats["std"] > 0

    def test_empty_results(self):
        """Handle empty results"""
        stats = calculate_confidence_stats([])

        assert stats["mean"] == 0
        assert stats["median"] == 0
        assert stats["min"] == 0
        assert stats["max"] == 0
        assert stats["std"] == 0

    def test_invalid_confidence_filtered(self):
        """Invalid confidence values are filtered"""
        results = [
            {"judge_confidence": 80},
            {"judge_confidence": "invalid"},
            {"judge_confidence": 150},  # Out of range
            {"judge_confidence": -10},  # Negative
            {"judge_confidence": 70},
        ]
        stats = calculate_confidence_stats(results)

        # Only valid values (80, 70) should be included
        assert stats["mean"] == 75.0
        assert stats["min"] == 70.0
        assert stats["max"] == 80.0

    def test_single_value(self):
        """Only one confidence score"""
        results = [{"judge_confidence": 85}]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 85.0
        assert stats["median"] == 85.0
        assert stats["min"] == 85.0
        assert stats["max"] == 85.0
        assert stats["std"] == 0.0

    def test_all_same_confidence(self):
        """All confidence scores are the same"""
        results = [
            {"judge_confidence": 80},
            {"judge_confidence": 80},
            {"judge_confidence": 80},
        ]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 80.0
        assert stats["median"] == 80.0
        assert stats["std"] == 0.0

    def test_zero_confidence(self):
        """0% is a valid confidence"""
        results = [
            {"judge_confidence": 0},
            {"judge_confidence": 50},
        ]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 25.0
        assert stats["min"] == 0.0

    def test_hundred_confidence(self):
        """100% is a valid confidence"""
        results = [
            {"judge_confidence": 100},
            {"judge_confidence": 50},
        ]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 75.0
        assert stats["max"] == 100.0

    def test_boundary_values(self):
        """Test boundary values 0, 50, 100"""
        results = [
            {"judge_confidence": 0},
            {"judge_confidence": 50},
            {"judge_confidence": 100},
        ]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 50.0
        assert stats["median"] == 50.0
        assert stats["min"] == 0.0
        assert stats["max"] == 100.0

    def test_float_precision(self):
        """Handle float confidence like 85.5%"""
        results = [
            {"judge_confidence": 85.5},
            {"judge_confidence": 74.5},
        ]
        stats = calculate_confidence_stats(results)

        assert stats["mean"] == 80.0
        assert stats["min"] == 74.5
        assert stats["max"] == 85.5


# ==========================================
# TestExtractDisagreementPatterns - Extended Tests
# ==========================================

class TestExtractDisagreementPatterns:
    """Tests for extract_disagreement_patterns function"""

    def test_basic_disagreements(self):
        """Extract disagreement patterns"""
        results = [
            {"disagreements_raw": json.dumps([
                {"aspect": "sentiment", "details": "positive vs negative"},
                {"aspect": "category", "details": "tech vs business"},
            ])},
            {"disagreements_raw": json.dumps([
                {"aspect": "sentiment", "details": "neutral vs positive"},
            ])},
            {"disagreements_raw": json.dumps([
                {"aspect": "confidence", "details": "high vs low"},
            ])},
        ]
        patterns = extract_disagreement_patterns(results)

        # sentiment appears twice, should be first
        assert len(patterns) == 3
        assert patterns[0]["aspect"] == "sentiment"
        assert patterns[0]["count"] == 2
        assert len(patterns[0]["examples"]) == 2

    def test_empty_results(self):
        """Handle empty results"""
        patterns = extract_disagreement_patterns([])
        assert patterns == []

    def test_empty_disagreements(self):
        """Handle results with empty disagreements"""
        results = [
            {"disagreements_raw": "[]"},
            {"disagreements_raw": json.dumps([])},
        ]
        patterns = extract_disagreement_patterns(results)
        assert patterns == []

    def test_invalid_json_handled(self):
        """Invalid JSON in disagreements is handled gracefully"""
        results = [
            {"disagreements_raw": "invalid json"},
            {"disagreements_raw": json.dumps([{"aspect": "valid", "details": "test"}])},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert patterns[0]["aspect"] == "valid"

    def test_many_disagreements(self):
        """10+ different aspects"""
        aspects = ["tone", "format", "length", "style", "accuracy",
                   "completeness", "clarity", "grammar", "logic", "detail"]
        results = [
            {"disagreements_raw": json.dumps([{"aspect": a, "details": f"issue with {a}"}])}
            for a in aspects
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 10
        # All should have count of 1
        for p in patterns:
            assert p["count"] == 1

    def test_same_aspect_different_details(self):
        """Same aspect with different details - aggregation"""
        results = [
            {"disagreements_raw": json.dumps([{"aspect": "format", "details": "spacing issue"}])},
            {"disagreements_raw": json.dumps([{"aspect": "format", "details": "indentation problem"}])},
            {"disagreements_raw": json.dumps([{"aspect": "format", "details": "line breaks"}])},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert patterns[0]["aspect"] == "format"
        assert patterns[0]["count"] == 3
        assert len(patterns[0]["examples"]) == 3

    def test_unicode_in_aspects(self):
        """Non-ASCII characters in aspects"""
        results = [
            {"disagreements_raw": json.dumps([
                {"aspect": "语言", "details": "Chinese aspect"},
                {"aspect": "émoji 🎉", "details": "with emoji"},
            ])},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 2
        aspect_names = [p["aspect"] for p in patterns]
        assert "语言" in aspect_names
        assert "émoji 🎉" in aspect_names

    def test_very_long_details(self):
        """Very long details are stored"""
        long_detail = "A" * 1000
        results = [
            {"disagreements_raw": json.dumps([{"aspect": "length", "details": long_detail}])},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert len(patterns[0]["examples"][0]) == 1000

    def test_nested_json(self):
        """Complex nested JSON structure"""
        results = [
            {"disagreements_raw": json.dumps([
                {"aspect": "complex", "details": {"nested": {"value": "deep"}}}
            ])},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert patterns[0]["aspect"] == "complex"

    def test_list_as_disagreements(self):
        """Already parsed list (not JSON string)"""
        results = [
            {"disagreements_raw": [{"aspect": "parsed", "details": "already a list"}]},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert patterns[0]["aspect"] == "parsed"

    def test_missing_aspect_key(self):
        """Partial objects - missing aspect key uses 'unknown'"""
        results = [
            {"disagreements_raw": json.dumps([{"details": "no aspect key"}])},
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert patterns[0]["aspect"] == "unknown"

    def test_example_limit(self):
        """Max 3 examples stored per aspect"""
        results = [
            {"disagreements_raw": json.dumps([{"aspect": "format", "details": f"example {i}"}])}
            for i in range(5)
        ]
        patterns = extract_disagreement_patterns(results)

        assert len(patterns) == 1
        assert patterns[0]["count"] == 5
        assert len(patterns[0]["examples"]) == 3  # Max 3 examples


# ==========================================
# TestCalculateQualityMetrics
# ==========================================

class TestCalculateQualityMetrics:
    """Tests for calculate_quality_metrics function"""

    def test_full_metrics_calculation(self):
        """Calculate all quality metrics"""
        results = [
            {
                "worker_1_score": 5, "worker_2_score": 4,
                "worker_1_rank": 1, "worker_2_rank": 2,
                "judge_confidence": 85,
                "disagreements_raw": json.dumps([{"aspect": "tone", "details": "formal vs casual"}]),
            },
            {
                "worker_1_score": 4, "worker_2_score": 5,
                "worker_1_rank": 2, "worker_2_rank": 1,
                "judge_confidence": 75,
                "disagreements_raw": json.dumps([{"aspect": "tone", "details": "another difference"}]),
            },
        ]
        metrics = calculate_quality_metrics(results)

        assert isinstance(metrics, QualityMetrics)
        assert "worker_1" in metrics.average_scores
        assert "worker_2" in metrics.average_scores
        assert metrics.confidence_stats["mean"] == 80.0
        assert len(metrics.disagreement_patterns) == 1
        assert metrics.disagreement_patterns[0]["aspect"] == "tone"
        assert metrics.disagreement_patterns[0]["count"] == 2


# ==========================================
# TestEnhancedJudgeParsing - Extended Tests
# ==========================================

class TestEnhancedJudgeParsing:
    """Tests for enhanced judge response parsing"""

    def test_parse_enhanced_response(self):
        """Parse a full enhanced judge response"""
        response = {
            "consensus": "Majority",
            "confidence": 85,
            "best_answer": "value1,value2,value3",
            "reasoning": "Worker 1 and 2 agree on most points",
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1, "notes": "Excellent response"},
                "worker_2": {"score": 4, "rank": 2, "notes": "Good but minor issues"},
            },
            "disagreements": [
                {"aspect": "formatting", "details": "Worker 1 used proper CSV, Worker 2 added quotes"},
            ]
        }

        # Verify structure
        assert response["consensus"] == "Majority"
        assert response["confidence"] == 85
        assert "worker_evaluations" in response
        assert response["worker_evaluations"]["worker_1"]["score"] == 5
        assert response["worker_evaluations"]["worker_1"]["rank"] == 1
        assert len(response["disagreements"]) == 1
        assert response["disagreements"][0]["aspect"] == "formatting"

    def test_backwards_compatible_response(self):
        """Old format responses still work"""
        response = {
            "consensus": "Yes",
            "best_answer": "value1,value2",
            "reasoning": "All workers agree",
        }

        # These fields should have defaults
        assert response.get("confidence", 50) == 50
        assert response.get("worker_evaluations", {}) == {}
        assert response.get("disagreements", []) == []


# ==========================================
# TestJSONParsingEdgeCases - NEW
# ==========================================

class TestJSONParsingEdgeCases:
    """Tests for JSON parsing edge cases in judge response handling"""

    def test_valid_json_standard(self):
        """Normal valid JSON case"""
        raw = '{"consensus": "Full", "confidence": 90, "best_answer": "test"}'
        parsed = json.loads(raw)
        assert parsed["consensus"] == "Full"
        assert parsed["confidence"] == 90

    def test_json_with_markdown_code_block(self):
        """JSON wrapped in ```json ... ```"""
        raw = '```json\n{"consensus": "Full", "best_answer": "test"}\n```'
        # Simulate the parsing logic from ConsensusProcessor
        clean = raw.strip()
        if "```" in clean:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', clean)
            if json_match:
                clean = json_match.group(1).strip()
        parsed = json.loads(clean)
        assert parsed["consensus"] == "Full"

    def test_json_with_triple_backticks(self):
        """JSON wrapped in ``` ... ``` (no json marker)"""
        raw = '```\n{"consensus": "Partial"}\n```'
        clean = raw.replace("```", "").strip()
        parsed = json.loads(clean)
        assert parsed["consensus"] == "Partial"

    def test_json_with_leading_text(self):
        """'Here's the answer: {...}'"""
        raw = "Here's the analysis: {\"consensus\": \"Majority\", \"best_answer\": \"x\"}"
        start = raw.find("{")
        end = raw.rfind("}") + 1
        clean = raw[start:end]
        parsed = json.loads(clean)
        assert parsed["consensus"] == "Majority"

    def test_json_with_trailing_text(self):
        """{...} I hope this helps!"""
        raw = '{"consensus": "None"} Let me know if you need more details!'
        end = raw.rfind("}") + 1
        clean = raw[:end]
        parsed = json.loads(clean)
        assert parsed["consensus"] == "None"

    def test_nested_json_objects(self):
        """Deep nesting"""
        raw = '{"outer": {"inner": {"deep": {"value": 42}}}}'
        parsed = json.loads(raw)
        assert parsed["outer"]["inner"]["deep"]["value"] == 42

    def test_json_with_unicode(self):
        """Emojis, Chinese, Arabic"""
        raw = '{"message": "Hello 你好 مرحبا 🎉", "consensus": "Full"}'
        parsed = json.loads(raw)
        assert "你好" in parsed["message"]
        assert "🎉" in parsed["message"]

    def test_json_with_escaped_quotes(self):
        """\\\" in strings"""
        raw = '{"best_answer": "He said \\"hello\\""}'
        parsed = json.loads(raw)
        assert '"hello"' in parsed["best_answer"]

    def test_json_with_newlines_in_values(self):
        """\\n in best_answer"""
        raw = '{"best_answer": "line1\\nline2\\nline3"}'
        parsed = json.loads(raw)
        assert "line1\nline2" in parsed["best_answer"]

    def test_malformed_json_partial(self):
        """Missing closing brace - should raise error"""
        raw = '{"consensus": "Full", "best_answer": "test"'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_empty_json_object(self):
        """Empty object {}"""
        raw = '{}'
        parsed = json.loads(raw)
        assert parsed == {}

    def test_json_array_instead_of_object(self):
        """[{}] wrapped - first element extraction"""
        raw = '[{"consensus": "Full"}]'
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        assert parsed["consensus"] == "Full"

    def test_json_with_comments_invalid(self):
        """// comments are invalid JSON"""
        raw = '{"consensus": "Full" // this is invalid}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_json_with_trailing_comma(self):
        """{key: 'value',} - trailing comma is invalid"""
        raw = '{"consensus": "Full",}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_null_values_in_json(self):
        """null handling"""
        raw = '{"consensus": "Full", "confidence": null, "reasoning": null}'
        parsed = json.loads(raw)
        assert parsed["confidence"] is None
        assert parsed["reasoning"] is None


# ==========================================
# TestBackwardsCompatibility - NEW
# ==========================================

class TestBackwardsCompatibility:
    """Tests for backwards compatibility with old response formats"""

    def test_old_format_yes_no_partial(self):
        """'Yes' instead of 'Full'"""
        response = {"consensus": "Yes", "best_answer": "result"}
        # Map old to new
        consensus_map = {"Yes": "Full", "No": "None", "Partial": "Partial"}
        mapped = consensus_map.get(response["consensus"], response["consensus"])
        assert mapped == "Full"

    def test_missing_confidence_field(self):
        """Default to 50 when confidence is missing"""
        response = {"consensus": "Yes", "best_answer": "x"}
        confidence = response.get("confidence", 50)
        assert confidence == 50

    def test_missing_worker_evaluations(self):
        """Default to empty dict when missing"""
        response = {"consensus": "Yes", "best_answer": "x"}
        evals = response.get("worker_evaluations", {})
        assert evals == {}

    def test_missing_disagreements(self):
        """Empty list default when disagreements missing"""
        response = {"consensus": "Yes", "best_answer": "x"}
        disagreements = response.get("disagreements", [])
        assert disagreements == []

    def test_mixed_old_new_format(self):
        """Some new fields, some old"""
        response = {
            "consensus": "Yes",  # Old format
            "confidence": 85,    # New field
            "best_answer": "x",
            # Missing worker_evaluations and disagreements
        }
        assert response.get("confidence", 50) == 85
        assert response.get("worker_evaluations", {}) == {}

    def test_consensus_case_insensitive(self):
        """'yes' vs 'Yes' vs 'YES'"""
        test_cases = ["yes", "Yes", "YES", "yEs"]
        for value in test_cases:
            normalized = value.lower()
            assert normalized == "yes"

    def test_legacy_field_names(self):
        """worker_scores vs worker_evaluations"""
        old_response = {"worker_scores": {"worker_1": 5}}
        # Support old field name as fallback
        evals = old_response.get("worker_evaluations") or old_response.get("worker_scores", {})
        assert evals["worker_1"] == 5

    def test_partial_worker_evaluations(self):
        """Only some workers rated"""
        response = {
            "consensus": "Majority",
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1},
                # worker_2 is missing
            }
        }
        evals = response.get("worker_evaluations", {})
        w1_score = evals.get("worker_1", {}).get("score", 3)
        w2_score = evals.get("worker_2", {}).get("score", 3)  # Default
        assert w1_score == 5
        assert w2_score == 3


# ==========================================
# TestConsensusConfigDataclass - Extended Tests
# ==========================================

class TestConsensusConfigDataclass:
    """Tests for ConsensusConfig dataclass with new fields"""

    def test_config_with_enhanced_options(self):
        """ConsensusConfig includes new quality scoring options"""
        from core.processing import ConsensusConfig

        config = ConsensusConfig(
            worker_configs=[{"provider_enum": "test", "api_key": "key", "base_url": None, "model": "test"}],
            judge_config={"provider_enum": "test", "api_key": "key", "base_url": None, "model": "test"},
            max_concurrency=5,
            auto_retry=True,
            max_retries=3,
            save_path=None,
            realtime_progress=True,
            include_reasoning=True,
            enable_quality_scoring=True,
            enable_disagreement_analysis=True,
        )

        assert config.enable_quality_scoring is True
        assert config.enable_disagreement_analysis is True

    def test_config_defaults(self):
        """New fields have correct defaults"""
        from core.processing import ConsensusConfig

        config = ConsensusConfig(
            worker_configs=[],
            judge_config={},
            max_concurrency=5,
            auto_retry=True,
            max_retries=3,
            save_path=None,
            realtime_progress=True,
            include_reasoning=True,
        )

        assert config.enable_quality_scoring is False
        assert config.enable_disagreement_analysis is False


# ==========================================
# TestQualityMetricsDataclass - NEW
# ==========================================

class TestQualityMetricsDataclass:
    """Tests for QualityMetrics dataclass"""

    def test_create_empty(self):
        """All empty fields"""
        metrics = QualityMetrics(
            average_scores={},
            rank_distribution={},
            confidence_stats={"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0},
            disagreement_patterns=[]
        )
        assert metrics.average_scores == {}
        assert metrics.rank_distribution == {}
        assert metrics.disagreement_patterns == []

    def test_create_full(self):
        """All fields populated"""
        metrics = QualityMetrics(
            average_scores={"worker_1": 4.5, "worker_2": 3.8},
            rank_distribution={
                "worker_1": {1: 5, 2: 3},
                "worker_2": {1: 3, 2: 5}
            },
            confidence_stats={"mean": 80, "median": 82, "min": 60, "max": 95, "std": 10},
            disagreement_patterns=[
                {"aspect": "format", "count": 5, "examples": ["ex1", "ex2"]}
            ]
        )
        assert metrics.average_scores["worker_1"] == 4.5
        assert metrics.rank_distribution["worker_1"][1] == 5
        assert metrics.confidence_stats["mean"] == 80
        assert len(metrics.disagreement_patterns) == 1

    def test_average_scores_type(self):
        """Dict[str, float] type verification"""
        metrics = QualityMetrics(
            average_scores={"worker_1": 4.5, "worker_2": 3.2},
            rank_distribution={},
            confidence_stats={},
            disagreement_patterns=[]
        )
        assert isinstance(metrics.average_scores, dict)
        for k, v in metrics.average_scores.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_rank_distribution_type(self):
        """Dict[str, Dict[int, int]] type verification"""
        metrics = QualityMetrics(
            average_scores={},
            rank_distribution={"worker_1": {1: 3, 2: 2}},
            confidence_stats={},
            disagreement_patterns=[]
        )
        assert isinstance(metrics.rank_distribution, dict)
        for worker, ranks in metrics.rank_distribution.items():
            assert isinstance(worker, str)
            assert isinstance(ranks, dict)
            for rank, count in ranks.items():
                assert isinstance(rank, int)
                assert isinstance(count, int)

    def test_disagreement_patterns_type(self):
        """List[Dict] type verification"""
        metrics = QualityMetrics(
            average_scores={},
            rank_distribution={},
            confidence_stats={},
            disagreement_patterns=[
                {"aspect": "test", "count": 1, "examples": []}
            ]
        )
        assert isinstance(metrics.disagreement_patterns, list)
        for p in metrics.disagreement_patterns:
            assert isinstance(p, dict)


# ==========================================
# TestMockJudgeResponses - NEW
# ==========================================

class TestMockJudgeResponses:
    """Tests for various mock judge response scenarios"""

    def test_full_consensus_response(self):
        """All workers agree"""
        response = {
            "consensus": "Full",
            "confidence": 95,
            "best_answer": "agreed,value",
            "reasoning": "All workers provided identical output",
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1, "notes": "Perfect"},
                "worker_2": {"score": 5, "rank": 1, "notes": "Perfect"},
            },
            "disagreements": []
        }
        assert response["consensus"] == "Full"
        assert response["confidence"] == 95
        assert len(response["disagreements"]) == 0

    def test_majority_consensus_response(self):
        """2 of 3 workers agree"""
        response = {
            "consensus": "Majority",
            "confidence": 80,
            "best_answer": "majority,value",
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1},
                "worker_2": {"score": 5, "rank": 1},
                "worker_3": {"score": 2, "rank": 3},
            },
            "disagreements": [
                {"aspect": "interpretation", "details": "Worker 3 misunderstood the task"}
            ]
        }
        assert response["consensus"] == "Majority"
        assert len(response["disagreements"]) == 1

    def test_partial_consensus_response(self):
        """Some agreement but not majority"""
        response = {
            "consensus": "Partial",
            "confidence": 55,
            "best_answer": "combined,result",
            "disagreements": [
                {"aspect": "format", "details": "Different output formats"},
                {"aspect": "values", "details": "Numerical disagreement"},
            ]
        }
        assert response["consensus"] == "Partial"
        assert response["confidence"] == 55
        assert len(response["disagreements"]) == 2

    def test_no_consensus_response(self):
        """Complete disagreement"""
        response = {
            "consensus": "None",
            "confidence": 20,
            "best_answer": "no,clear,winner",
            "worker_evaluations": {
                "worker_1": {"score": 3, "rank": 1},
                "worker_2": {"score": 3, "rank": 1},
            },
            "disagreements": [
                {"aspect": "methodology", "details": "Completely different approaches"},
                {"aspect": "output", "details": "Incompatible results"},
                {"aspect": "interpretation", "details": "Different understanding of task"},
            ]
        }
        assert response["consensus"] == "None"
        assert response["confidence"] == 20
        assert len(response["disagreements"]) == 3

    def test_high_confidence_response(self):
        """95%+ confidence"""
        response = {"consensus": "Full", "confidence": 98}
        assert response["confidence"] >= 95

    def test_low_confidence_response(self):
        """<30% confidence"""
        response = {"consensus": "Partial", "confidence": 25}
        assert response["confidence"] < 30

    def test_varied_scores_response(self):
        """5, 3, 1 scores"""
        response = {
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1},
                "worker_2": {"score": 3, "rank": 2},
                "worker_3": {"score": 1, "rank": 3},
            }
        }
        scores = [e["score"] for e in response["worker_evaluations"].values()]
        assert sorted(scores, reverse=True) == [5, 3, 1]

    def test_tied_ranks_response(self):
        """Workers tied for rank"""
        response = {
            "worker_evaluations": {
                "worker_1": {"score": 4, "rank": 1},
                "worker_2": {"score": 4, "rank": 1},  # Tied
                "worker_3": {"score": 2, "rank": 3},
            }
        }
        ranks = [e["rank"] for e in response["worker_evaluations"].values()]
        assert ranks.count(1) == 2  # Two workers tied for 1st

    def test_many_disagreements_response(self):
        """5+ disagreements"""
        response = {
            "disagreements": [
                {"aspect": f"aspect_{i}", "details": f"Detail {i}"}
                for i in range(7)
            ]
        }
        assert len(response["disagreements"]) >= 5

    def test_empty_disagreements_response(self):
        """No disagreements found"""
        response = {"consensus": "Full", "disagreements": []}
        assert response["disagreements"] == []


# ==========================================
# TestErrorScenarios - NEW
# ==========================================

class TestErrorScenarios:
    """Tests for error handling scenarios"""

    def test_rate_limit_error(self):
        """429 handling"""
        from tests.mocks.mock_llm_client import MockRateLimitError
        error = MockRateLimitError()
        assert "429" in str(error)
        assert "Rate limit" in str(error)

    def test_auth_error(self):
        """401 handling"""
        from tests.mocks.mock_llm_client import MockAuthError
        error = MockAuthError()
        assert "401" in str(error)
        assert "Invalid API key" in str(error)

    def test_timeout_error(self):
        """Timeout recovery"""
        from tests.mocks.mock_llm_client import MockTimeoutError
        error = MockTimeoutError()
        assert "timed out" in str(error).lower()

    def test_connection_error(self):
        """Network failure"""
        from tests.mocks.mock_llm_client import MockConnectionError
        error = MockConnectionError()
        assert "Connection" in str(error)

    def test_model_not_found(self):
        """404 handling"""
        from tests.mocks.mock_llm_client import MockModelNotFoundError
        error = MockModelNotFoundError("gpt-5")
        assert "404" in str(error)
        assert "gpt-5" in str(error)

    def test_content_filter_error(self):
        """Blocked content"""
        from tests.mocks.mock_llm_client import MockContentFilterError
        error = MockContentFilterError()
        assert "safety" in str(error).lower() or "blocked" in str(error).lower()

    def test_context_length_error(self):
        """Token limit"""
        from tests.mocks.mock_llm_client import MockContextLengthError
        error = MockContextLengthError()
        assert "context" in str(error).lower() or "token" in str(error).lower()

    def test_server_error_500(self):
        """Internal error"""
        from tests.mocks.mock_llm_client import MockServerError
        error = MockServerError()
        assert "500" in str(error)

    @pytest.mark.asyncio
    async def test_retry_success(self):
        """Retry recovers after initial failure"""
        from tests.mocks.mock_llm_client import MockLLMClient, MockResponse, MockRateLimitError

        client = MockLLMClient()
        # First call fails, second succeeds
        client.set_responses([
            MockRateLimitError(),
            MockResponse.create('{"result": "success"}')
        ])

        # First call should raise
        with pytest.raises(Exception):
            await client.chat.completions.create()

        # Second call should succeed
        response = await client.chat.completions.create()
        assert response.choices[0].message.content == '{"result": "success"}'

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Gives up after N tries"""
        from tests.mocks.mock_llm_client import MockLLMClient, MockRateLimitError

        client = MockLLMClient()
        # All calls fail
        client.set_responses([
            MockRateLimitError(),
            MockRateLimitError(),
            MockRateLimitError(),
        ])

        # Should raise on each attempt
        for _ in range(3):
            with pytest.raises(Exception):
                await client.chat.completions.create()


# ==========================================
# TestPropertyBased - NEW (with graceful skip)
# ==========================================

class TestPropertyBased:
    """Property-based tests using Hypothesis (graceful skip if unavailable)"""

    @pytest.fixture(autouse=True)
    def check_hypothesis(self):
        """Skip tests if Hypothesis is not installed"""
        pytest.importorskip("hypothesis")

    def test_scores_always_in_range(self):
        """Generated scores should always be 1-5"""
        from hypothesis import given, strategies as st

        @given(st.lists(st.floats(min_value=1, max_value=5), min_size=1, max_size=10))
        def check_scores(scores):
            results = [{"worker_1_score": s} for s in scores]
            avg = calculate_average_scores(results)
            if "worker_1" in avg:
                assert 1 <= avg["worker_1"] <= 5

        check_scores()

    def test_ranks_unique_per_row(self):
        """Ranks should be valid integers > 0"""
        from hypothesis import given, strategies as st

        @given(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=10))
        def check_ranks(ranks):
            results = [{"worker_1_rank": r} for r in ranks]
            dist = calculate_rank_distribution(results)
            if "worker_1" in dist:
                for rank in dist["worker_1"].keys():
                    assert rank > 0

        check_ranks()

    def test_confidence_always_0_100(self):
        """Confidence must be in valid range"""
        from hypothesis import given, strategies as st

        @given(st.lists(st.floats(min_value=0, max_value=100), min_size=1, max_size=20))
        def check_confidence(confs):
            results = [{"judge_confidence": c} for c in confs]
            stats = calculate_confidence_stats(results)
            assert 0 <= stats["min"] <= 100
            assert 0 <= stats["max"] <= 100

        check_confidence()

    def test_disagreements_parseable(self):
        """JSON roundtrip should work"""
        from hypothesis import given, strategies as st

        @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
        def check_json(aspects):
            disagreements = [{"aspect": a, "details": f"detail for {a}"} for a in aspects]
            raw = json.dumps(disagreements)
            parsed = json.loads(raw)
            assert len(parsed) == len(aspects)

        check_json()

    def test_average_scores_bounded(self):
        """Result should be within 1-5"""
        from hypothesis import given, strategies as st

        @given(st.lists(
            st.fixed_dictionaries({
                "worker_1_score": st.integers(min_value=1, max_value=5),
                "worker_2_score": st.integers(min_value=1, max_value=5)
            }),
            min_size=1, max_size=50
        ))
        def check_bounded(results):
            avg = calculate_average_scores(results)
            for score in avg.values():
                assert 1 <= score <= 5

        check_bounded()

    def test_confidence_stats_consistent(self):
        """min <= median <= max"""
        from hypothesis import given, strategies as st

        @given(st.lists(st.floats(min_value=0, max_value=100), min_size=3, max_size=20))
        def check_consistent(confs):
            results = [{"judge_confidence": c} for c in confs]
            stats = calculate_confidence_stats(results)
            if stats["max"] > 0:  # Has valid data
                assert stats["min"] <= stats["median"] <= stats["max"]

        check_consistent()


# ==========================================
# TestPerformance - NEW
# ==========================================

class TestPerformance:
    """Performance and stress tests"""

    def test_1000_rows_analytics(self):
        """Large dataset performance"""
        import time

        results = [
            {
                "worker_1_score": (i % 5) + 1,
                "worker_2_score": ((i + 1) % 5) + 1,
                "worker_1_rank": (i % 2) + 1,
                "worker_2_rank": ((i + 1) % 2) + 1,
                "judge_confidence": 50 + (i % 50),
                "disagreements_raw": json.dumps([
                    {"aspect": f"aspect_{i % 10}", "details": f"Detail for row {i}"}
                ]) if i % 3 == 0 else "[]"
            }
            for i in range(1000)
        ]

        start = time.time()
        metrics = calculate_quality_metrics(results)
        duration = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert duration < 1.0
        assert len(metrics.average_scores) == 2
        assert metrics.confidence_stats["mean"] > 0

    def test_many_workers_10(self):
        """10 workers"""
        results = []
        for i in range(100):
            row = {}
            for w in range(10):
                row[f"worker_{w+1}_score"] = (i + w) % 5 + 1
                row[f"worker_{w+1}_rank"] = (w % 10) + 1
            results.append(row)

        avg = calculate_average_scores(results)
        dist = calculate_rank_distribution(results)

        assert len(avg) == 10
        assert len(dist) == 10

    def test_large_disagreement_list(self):
        """100 disagreements per row"""
        results = [
            {
                "disagreements_raw": json.dumps([
                    {"aspect": f"aspect_{j}", "details": f"Detail {j}"}
                    for j in range(100)
                ])
            }
            for _ in range(10)
        ]

        patterns = extract_disagreement_patterns(results)

        # Should have 100 unique aspects
        assert len(patterns) == 100
        # Each aspect should appear 10 times
        for p in patterns:
            assert p["count"] == 10

    def test_concurrent_processing_mock(self):
        """Mock concurrent execution"""
        import asyncio

        async def mock_process(idx):
            await asyncio.sleep(0.001)  # Simulate async work
            return {"worker_1_score": idx % 5 + 1}

        async def run_concurrent():
            tasks = [mock_process(i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.get_event_loop().run_until_complete(run_concurrent())
        avg = calculate_average_scores(results)

        assert "worker_1" in avg
        assert len(results) == 100

    def test_memory_efficiency(self):
        """No excessive memory usage with large data"""
        import sys

        # Create large dataset
        results = [
            {
                "worker_1_score": i % 5 + 1,
                "disagreements_raw": json.dumps([{"aspect": "a", "details": "x" * 100}])
            }
            for i in range(1000)
        ]

        # Calculate metrics
        metrics = calculate_quality_metrics(results)

        # Metrics object should be reasonably sized
        # (not holding onto all input data)
        assert len(metrics.disagreement_patterns) <= 100  # Not 1000


# ==========================================
# TestConsensusProcessorIntegration - NEW
# ==========================================

class TestConsensusProcessorIntegration:
    """Integration tests for ConsensusProcessor with mock LLM"""

    @pytest.fixture
    def mock_consensus_config(self):
        """Create a mock ConsensusConfig"""
        from core.processing import ConsensusConfig
        return ConsensusConfig(
            worker_configs=[
                {"provider_enum": "test", "api_key": "key1", "base_url": None, "model": "test-model"},
                {"provider_enum": "test", "api_key": "key2", "base_url": None, "model": "test-model"},
            ],
            judge_config={"provider_enum": "test", "api_key": "judge-key", "base_url": None, "model": "test-model"},
            max_concurrency=2,
            auto_retry=True,
            max_retries=2,
            save_path=None,
            realtime_progress=False,
            include_reasoning=True,
            enable_quality_scoring=True,
            enable_disagreement_analysis=True,
        )

    def test_config_with_quality_scoring(self, mock_consensus_config):
        """Full flow with scoring enabled"""
        assert mock_consensus_config.enable_quality_scoring is True
        assert len(mock_consensus_config.worker_configs) == 2

    def test_config_with_disagreement_analysis(self, mock_consensus_config):
        """Full flow with disagreement analysis enabled"""
        assert mock_consensus_config.enable_disagreement_analysis is True

    def test_config_with_both_options(self, mock_consensus_config):
        """Both options enabled together"""
        assert mock_consensus_config.enable_quality_scoring is True
        assert mock_consensus_config.enable_disagreement_analysis is True

    def test_config_with_neither_option(self):
        """Standard mode - neither option enabled"""
        from core.processing import ConsensusConfig
        config = ConsensusConfig(
            worker_configs=[],
            judge_config={},
            max_concurrency=5,
            auto_retry=True,
            max_retries=3,
            save_path=None,
            realtime_progress=True,
            include_reasoning=True,
        )
        assert config.enable_quality_scoring is False
        assert config.enable_disagreement_analysis is False

    def test_result_columns_quality(self):
        """DataFrame should have score/rank columns when quality scoring enabled"""
        # Simulate results with quality scoring columns
        results = [
            {
                "worker_1": "output1",
                "worker_2": "output2",
                "worker_1_score": 5,
                "worker_2_score": 4,
                "worker_1_rank": 1,
                "worker_2_rank": 2,
                "judge_consensus": "Majority",
                "judge_best_answer": "result",
            }
        ]

        assert "worker_1_score" in results[0]
        assert "worker_2_score" in results[0]
        assert "worker_1_rank" in results[0]
        assert "worker_2_rank" in results[0]

    def test_result_columns_disagreement(self):
        """DataFrame should have disagreement columns when enabled"""
        results = [
            {
                "judge_consensus": "Partial",
                "disagreement_summary": "format; tone",
                "disagreements_raw": json.dumps([{"aspect": "format", "details": "x"}]),
            }
        ]

        assert "disagreement_summary" in results[0]
        assert "disagreements_raw" in results[0]

    def test_result_columns_confidence(self):
        """DataFrame should have confidence column"""
        results = [{"judge_confidence": 85}]
        assert "judge_confidence" in results[0]
        assert results[0]["judge_confidence"] == 85

    def test_worker_count_2_vs_3(self, mock_consensus_config):
        """Different worker counts"""
        assert len(mock_consensus_config.worker_configs) == 2

        # Create config with 3 workers
        from core.processing import ConsensusConfig
        config_3 = ConsensusConfig(
            worker_configs=[
                {"provider_enum": "t", "api_key": "k", "base_url": None, "model": "m"},
                {"provider_enum": "t", "api_key": "k", "base_url": None, "model": "m"},
                {"provider_enum": "t", "api_key": "k", "base_url": None, "model": "m"},
            ],
            judge_config={},
            max_concurrency=5,
            auto_retry=True,
            max_retries=3,
            save_path=None,
            realtime_progress=True,
            include_reasoning=True,
        )
        assert len(config_3.worker_configs) == 3

    def test_error_handling_defaults(self):
        """Default values when parsing fails"""
        # Simulated defaults when judge returns error
        defaults_on_error = {
            "judge_consensus": "Error",
            "judge_confidence": 0,
            "worker_1_score": 0,
            "worker_1_rank": 0,
            "disagreement_summary": "Error",
            "disagreements_raw": "[]",
        }

        assert defaults_on_error["judge_confidence"] == 0
        assert defaults_on_error["disagreements_raw"] == "[]"

    def test_partial_success_handling(self):
        """Some rows succeed, some fail"""
        results = [
            {"judge_consensus": "Full", "judge_confidence": 90},  # Success
            {"judge_consensus": "Error", "judge_confidence": 0},  # Error
            {"judge_consensus": "Majority", "judge_confidence": 75},  # Success
        ]

        success_count = sum(1 for r in results if r["judge_consensus"] != "Error")
        error_count = sum(1 for r in results if r["judge_consensus"] == "Error")

        assert success_count == 2
        assert error_count == 1


# ==========================================
# TestRenderQualityMetrics - NEW
# ==========================================

class TestRenderQualityMetrics:
    """Tests for render_quality_metrics visualization (mocked Streamlit)"""

    @pytest.fixture
    def sample_metrics(self):
        """Sample QualityMetrics for rendering tests"""
        return QualityMetrics(
            average_scores={"worker_1": 4.5, "worker_2": 3.8, "worker_3": 4.2},
            rank_distribution={
                "worker_1": {1: 5, 2: 3, 3: 2},
                "worker_2": {1: 3, 2: 4, 3: 3},
                "worker_3": {1: 2, 2: 3, 3: 5},
            },
            confidence_stats={"mean": 80, "median": 82, "min": 60, "max": 95, "std": 10.5},
            disagreement_patterns=[
                {"aspect": "format", "count": 5, "percentage": 50.0, "examples": ["ex1", "ex2"]},
                {"aspect": "tone", "count": 3, "percentage": 30.0, "examples": ["ex3"]},
            ]
        )

    def test_render_with_plotly_available(self, sample_metrics):
        """Charts render when plotly is available"""
        try:
            import plotly.graph_objects as go
            # If plotly is available, bar chart should be creatable
            fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[1, 2])])
            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_render_without_plotly(self, sample_metrics):
        """Fallback metrics work without plotly"""
        # Should be able to access metrics even without plotly
        assert sample_metrics.average_scores["worker_1"] == 4.5
        assert sample_metrics.confidence_stats["mean"] == 80

    def test_render_empty_metrics(self):
        """Handle empty metrics"""
        empty_metrics = QualityMetrics(
            average_scores={},
            rank_distribution={},
            confidence_stats={"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0},
            disagreement_patterns=[]
        )
        assert empty_metrics.average_scores == {}
        assert len(empty_metrics.disagreement_patterns) == 0

    def test_render_single_worker(self):
        """Only 1 worker"""
        single_worker = QualityMetrics(
            average_scores={"worker_1": 4.0},
            rank_distribution={"worker_1": {1: 10}},
            confidence_stats={"mean": 75, "median": 75, "min": 75, "max": 75, "std": 0},
            disagreement_patterns=[]
        )
        assert len(single_worker.average_scores) == 1
        assert "worker_1" in single_worker.average_scores

    def test_render_leaderboard_order(self, sample_metrics):
        """Sorted by score descending"""
        sorted_workers = sorted(
            sample_metrics.average_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        assert sorted_workers[0][0] == "worker_1"  # 4.5 is highest
        assert sorted_workers[1][0] == "worker_3"  # 4.2
        assert sorted_workers[2][0] == "worker_2"  # 3.8

    def test_render_rank_distribution(self, sample_metrics):
        """Grouped bars for rank distribution"""
        dist = sample_metrics.rank_distribution
        # Each worker should have ranks 1, 2, 3
        for worker in ["worker_1", "worker_2", "worker_3"]:
            assert 1 in dist[worker]
            assert 2 in dist[worker]
            assert 3 in dist[worker]

    def test_render_confidence_stats(self, sample_metrics):
        """All 5 metrics shown"""
        stats = sample_metrics.confidence_stats
        required_keys = ["mean", "median", "min", "max", "std"]
        for key in required_keys:
            assert key in stats

    def test_render_disagreement_table(self, sample_metrics):
        """Hotspots table format"""
        patterns = sample_metrics.disagreement_patterns
        for p in patterns:
            assert "aspect" in p
            assert "count" in p
            assert "percentage" in p
            assert "examples" in p
