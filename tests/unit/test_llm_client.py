"""
Tests for LLM Client Module
"""

import pytest
from unittest.mock import patch, MagicMock
from core.llm_client import (
    create_http_client, get_client, fetch_local_models
)
from core.providers import LLMProvider, PROVIDER_CONFIGS


class TestHttpClient:
    """Tests for HTTP client creation"""

    def test_create_http_client(self):
        """HTTP client should be created with correct settings"""
        client = create_http_client()
        assert client is not None
        # Client should be an AsyncClient
        import httpx
        assert isinstance(client, httpx.AsyncClient)


class TestGetClient:
    """Tests for LLM client creation"""

    def test_get_client_openai(self):
        """OpenAI client should use default base URL"""
        client = get_client(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            base_url=None
        )
        assert client is not None
        assert client.api_key == "test-key"

    def test_get_client_anthropic(self):
        """Anthropic client should use configured base URL"""
        config = PROVIDER_CONFIGS[LLMProvider.ANTHROPIC]
        client = get_client(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key",
            base_url=config.base_url
        )
        assert client is not None
        assert "anthropic" in str(client.base_url)

    def test_get_client_local_no_key(self):
        """Local providers should work without API key"""
        client = get_client(
            provider=LLMProvider.LM_STUDIO,
            api_key="",
            base_url="http://localhost:1234/v1"
        )
        assert client is not None
        # Should use dummy key
        assert client.api_key == "dummy"

    def test_get_client_custom_base_url(self):
        """Custom base URL should override default"""
        custom_url = "http://my-custom-api.com/v1"
        client = get_client(
            provider=LLMProvider.CUSTOM,
            api_key="key",
            base_url=custom_url
        )
        assert client is not None
        assert str(client.base_url) == custom_url + "/"

    def test_get_client_openrouter_headers(self):
        """OpenRouter client should have custom headers"""
        client = get_client(
            provider=LLMProvider.OPENROUTER,
            api_key="or-test-key",
            base_url="https://openrouter.ai/api/v1"
        )
        assert client is not None
        # OpenRouter requires specific headers
        assert client.default_headers is not None


class TestFetchLocalModels:
    """Tests for local model fetching"""

    @patch('core.llm_client.httpx.get')
    def test_fetch_local_models_success(self, mock_get):
        """Should parse models from API response"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-3"},
                {"id": "mistral-7b"},
                {"id": "codellama"}
            ]
        }
        mock_get.return_value = mock_response

        models = fetch_local_models("http://localhost:1234/v1")
        assert models == ["llama-3", "mistral-7b", "codellama"]

    @patch('core.llm_client.httpx.get')
    def test_fetch_local_models_connection_error(self, mock_get):
        """Should return empty list on connection error"""
        mock_get.side_effect = Exception("Connection refused")

        models = fetch_local_models("http://localhost:1234/v1")
        assert models == []

    @patch('core.llm_client.httpx.get')
    def test_fetch_local_models_bad_response(self, mock_get):
        """Should return empty list on bad response"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        models = fetch_local_models("http://localhost:1234/v1")
        assert models == []


class TestMockLLMClient:
    """Tests using the mock LLM client"""

    @pytest.mark.asyncio
    async def test_mock_client_basic_call(self, mock_llm_client):
        """Mock client should return configured response"""
        response = await mock_llm_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response.choices[0].message.content == "Test response"
        assert mock_llm_client.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_client_tracks_kwargs(self, mock_llm_client):
        """Mock client should track call kwargs"""
        await mock_llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            max_tokens=100
        )

        assert mock_llm_client.last_kwargs["model"] == "gpt-4o"
        assert mock_llm_client.last_kwargs["temperature"] == 0.5
        assert mock_llm_client.last_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_mock_client_raises_error(self):
        """Mock client should raise configured error"""
        from tests.mocks.mock_llm_client import MockLLMClient, MockRateLimitError

        client = MockLLMClient(error_to_raise=MockRateLimitError())

        with pytest.raises(MockRateLimitError):
            await client.chat.completions.create(
                model="test",
                messages=[]
            )

    @pytest.mark.asyncio
    async def test_mock_client_response_sequence(self, mock_responses_sequence):
        """Mock client should return responses in sequence"""
        from tests.mocks.mock_llm_client import MockLLMClient

        client = MockLLMClient()
        client.set_responses(mock_responses_sequence.copy())

        r1 = await client.chat.completions.create(model="test", messages=[])
        r2 = await client.chat.completions.create(model="test", messages=[])
        r3 = await client.chat.completions.create(model="test", messages=[])

        assert r1.choices[0].message.content == "First response"
        assert r2.choices[0].message.content == "Second response"
        assert r3.choices[0].message.content == "Third response"

    @pytest.mark.asyncio
    async def test_mock_client_reset(self, mock_llm_client):
        """Mock client reset should clear tracking"""
        await mock_llm_client.chat.completions.create(model="test", messages=[])
        assert mock_llm_client.call_count == 1

        mock_llm_client.reset()
        assert mock_llm_client.call_count == 0
        assert mock_llm_client.last_kwargs == {}
