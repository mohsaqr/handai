"""
Pytest Configuration and Shared Fixtures
"""

import pytest
import tempfile
import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mocks.mock_llm_client import MockLLMClient, MockResponse


@pytest.fixture
def mock_llm_client():
    """Basic mock LLM client"""
    return MockLLMClient(default_response="Test response")


@pytest.fixture
def mock_llm_client_json():
    """Mock LLM client that returns JSON"""
    return MockLLMClient(default_response='{"result": "success", "value": 42}')


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            'This is a positive review',
            'This product is terrible',
            'Average quality, nothing special',
            'Absolutely love it!',
            'Would not recommend'
        ],
        'rating': [5, 1, 3, 5, 2]
    })


@pytest.fixture
def sample_dataframe_large():
    """Larger DataFrame for testing"""
    return pd.DataFrame({
        'id': list(range(100)),
        'text': [f'Sample text {i}' for i in range(100)],
        'value': [i * 10 for i in range(100)]
    })


@pytest.fixture
def sample_csv_content():
    """Sample CSV content"""
    return """id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300
"""


@pytest.fixture
def sample_json_content():
    """Sample JSON content"""
    return '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'


@pytest.fixture
def mock_responses_sequence():
    """Sequence of mock responses for testing retries"""
    return [
        MockResponse.create("First response"),
        MockResponse.create("Second response"),
        MockResponse.create("Third response"),
    ]


# ==========================================
# Consensus Coder Fixtures
# ==========================================

@pytest.fixture
def sample_consensus_results():
    """Standard consensus results for testing analytics"""
    import json
    return [
        {
            "worker_1_score": 5, "worker_2_score": 4,
            "worker_1_rank": 1, "worker_2_rank": 2,
            "judge_confidence": 85,
            "disagreements_raw": json.dumps([
                {"aspect": "sentiment", "details": "positive vs neutral"}
            ]),
        },
        {
            "worker_1_score": 4, "worker_2_score": 5,
            "worker_1_rank": 2, "worker_2_rank": 1,
            "judge_confidence": 75,
            "disagreements_raw": json.dumps([
                {"aspect": "sentiment", "details": "neutral vs positive"}
            ]),
        },
        {
            "worker_1_score": 5, "worker_2_score": 5,
            "worker_1_rank": 1, "worker_2_rank": 1,
            "judge_confidence": 95,
            "disagreements_raw": "[]",
        },
    ]


@pytest.fixture
def sample_enhanced_judge_response():
    """Full enhanced judge JSON response"""
    return {
        "consensus": "Majority",
        "confidence": 85,
        "best_answer": "value1,value2",
        "reasoning": "Workers agree on most points",
        "worker_evaluations": {
            "worker_1": {"score": 5, "rank": 1, "notes": "Excellent"},
            "worker_2": {"score": 4, "rank": 2, "notes": "Good"},
        },
        "disagreements": [
            {"aspect": "format", "details": "Minor formatting differences"}
        ]
    }


@pytest.fixture
def sample_quality_metrics():
    """Sample QualityMetrics for testing"""
    from core.comparison_analytics import QualityMetrics
    return QualityMetrics(
        average_scores={"worker_1": 4.5, "worker_2": 4.0},
        rank_distribution={
            "worker_1": {1: 2, 2: 1},
            "worker_2": {1: 1, 2: 2}
        },
        confidence_stats={"mean": 85, "median": 85, "min": 75, "max": 95, "std": 8.2},
        disagreement_patterns=[
            {"aspect": "sentiment", "count": 2, "percentage": 66.7, "examples": ["positive vs neutral", "neutral vs positive"]}
        ]
    )


@pytest.fixture
def mock_consensus_processor_config():
    """Mock ConsensusConfig for integration tests"""
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


@pytest.fixture
def sample_judge_responses():
    """Various judge response scenarios for testing"""
    import json
    return {
        "full_consensus": json.dumps({
            "consensus": "Full",
            "confidence": 95,
            "best_answer": "agreed,value",
            "reasoning": "All workers agree",
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1},
                "worker_2": {"score": 5, "rank": 1},
            },
            "disagreements": []
        }),
        "majority_consensus": json.dumps({
            "consensus": "Majority",
            "confidence": 80,
            "best_answer": "majority,value",
            "worker_evaluations": {
                "worker_1": {"score": 5, "rank": 1},
                "worker_2": {"score": 4, "rank": 2},
            },
            "disagreements": [
                {"aspect": "minor", "details": "Small differences"}
            ]
        }),
        "no_consensus": json.dumps({
            "consensus": "None",
            "confidence": 30,
            "best_answer": "unclear",
            "worker_evaluations": {
                "worker_1": {"score": 3, "rank": 1},
                "worker_2": {"score": 3, "rank": 1},
            },
            "disagreements": [
                {"aspect": "major", "details": "Completely different approaches"},
                {"aspect": "format", "details": "Different output formats"},
            ]
        }),
        "with_markdown": '```json\n{"consensus": "Full", "best_answer": "test"}\n```',
        "with_leading_text": 'Here is my analysis: {"consensus": "Partial", "best_answer": "x"}',
    }
