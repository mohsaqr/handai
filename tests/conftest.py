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
