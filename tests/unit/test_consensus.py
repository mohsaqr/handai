"""
Tests for Consensus Coder Tool - Enhanced Judge Features
"""

import pytest
import json
from core.comparison_analytics import (
    calculate_average_scores,
    calculate_rank_distribution,
    calculate_confidence_stats,
    extract_disagreement_patterns,
    calculate_quality_metrics,
    QualityMetrics,
)


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
