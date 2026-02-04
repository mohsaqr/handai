"""
Tests for LLM Provider Module
"""

import pytest
from core.providers import (
    LLMProvider, PROVIDER_CONFIGS, ProviderConfig,
    get_provider_by_name, get_provider_names, is_local_provider,
    is_reasoning_model, supports_json_mode, uses_completion_tokens,
    requires_max_tokens
)


class TestLLMProvider:
    """Tests for LLMProvider enum"""

    def test_all_providers_have_configs(self):
        """Every provider should have a configuration"""
        for provider in LLMProvider:
            assert provider in PROVIDER_CONFIGS

    def test_provider_configs_have_required_fields(self):
        """All configs should have required fields"""
        for provider, config in PROVIDER_CONFIGS.items():
            assert isinstance(config, ProviderConfig)
            assert config.name
            assert isinstance(config.requires_api_key, bool)
            assert config.description


class TestProviderLookup:
    """Tests for provider lookup functions"""

    def test_get_provider_by_name_valid(self):
        """Should find provider by display name"""
        provider = get_provider_by_name("OpenAI")
        assert provider == LLMProvider.OPENAI

    def test_get_provider_by_name_anthropic(self):
        """Should find Anthropic provider"""
        provider = get_provider_by_name("Anthropic (Claude)")
        assert provider == LLMProvider.ANTHROPIC

    def test_get_provider_by_name_invalid(self):
        """Should raise error for unknown provider"""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider_by_name("NonexistentProvider")

    def test_get_provider_names(self):
        """Should return all provider names"""
        names = get_provider_names()
        assert "OpenAI" in names
        assert "Anthropic (Claude)" in names
        assert "Groq" in names
        assert len(names) == len(LLMProvider)


class TestLocalProviders:
    """Tests for local provider detection"""

    def test_lm_studio_is_local(self):
        """LM Studio should be local"""
        assert is_local_provider(LLMProvider.LM_STUDIO) is True

    def test_ollama_is_local(self):
        """Ollama should be local"""
        assert is_local_provider(LLMProvider.OLLAMA) is True

    def test_custom_is_local(self):
        """Custom endpoint should be local"""
        assert is_local_provider(LLMProvider.CUSTOM) is True

    def test_openai_is_not_local(self):
        """OpenAI should not be local"""
        assert is_local_provider(LLMProvider.OPENAI) is False

    def test_anthropic_is_not_local(self):
        """Anthropic should not be local"""
        assert is_local_provider(LLMProvider.ANTHROPIC) is False


class TestReasoningModels:
    """Tests for reasoning model detection"""

    def test_o1_is_reasoning(self):
        """o1 models should be reasoning"""
        assert is_reasoning_model("o1") is True
        assert is_reasoning_model("o1-mini") is True
        assert is_reasoning_model("o1-pro") is True

    def test_o3_is_reasoning(self):
        """o3 models should be reasoning"""
        assert is_reasoning_model("o3") is True
        assert is_reasoning_model("o3-mini") is True

    def test_gpt5_is_reasoning(self):
        """GPT-5 models should be reasoning"""
        assert is_reasoning_model("gpt-5") is True
        assert is_reasoning_model("gpt-5-mini") is True
        assert is_reasoning_model("gpt-5.1") is True

    def test_gpt4_is_not_reasoning(self):
        """GPT-4 models should not be reasoning"""
        assert is_reasoning_model("gpt-4o") is False
        assert is_reasoning_model("gpt-4-turbo") is False
        assert is_reasoning_model("gpt-4.1") is False

    def test_claude_is_not_reasoning(self):
        """Claude models should not be reasoning"""
        assert is_reasoning_model("claude-sonnet-4") is False
        assert is_reasoning_model("claude-3-5-sonnet") is False

    def test_gemini_is_not_reasoning(self):
        """Gemini models should not be reasoning"""
        assert is_reasoning_model("gemini-2.0-flash") is False


class TestJsonModeSupport:
    """Tests for JSON mode support detection"""

    def test_openai_gpt4_supports_json(self):
        """OpenAI GPT-4 should support JSON mode"""
        assert supports_json_mode(LLMProvider.OPENAI, "gpt-4o") is True

    def test_reasoning_models_no_json(self):
        """Reasoning models should not support JSON mode"""
        assert supports_json_mode(LLMProvider.OPENAI, "o1") is False
        assert supports_json_mode(LLMProvider.OPENAI, "o3-mini") is False
        assert supports_json_mode(LLMProvider.OPENAI, "gpt-5") is False

    def test_local_providers_no_json(self):
        """Local providers should not support JSON mode"""
        assert supports_json_mode(LLMProvider.LM_STUDIO, "any-model") is False
        assert supports_json_mode(LLMProvider.OLLAMA, "llama3") is False
        assert supports_json_mode(LLMProvider.CUSTOM, "custom") is False


class TestCompletionTokens:
    """Tests for completion tokens parameter detection"""

    def test_reasoning_models_use_completion_tokens(self):
        """Reasoning models should use max_completion_tokens"""
        assert uses_completion_tokens("o1") is True
        assert uses_completion_tokens("o1-mini") is True
        assert uses_completion_tokens("o3") is True
        assert uses_completion_tokens("gpt-5") is True

    def test_regular_models_use_max_tokens(self):
        """Regular models should use max_tokens"""
        assert uses_completion_tokens("gpt-4o") is False
        assert uses_completion_tokens("gpt-4-turbo") is False
        assert uses_completion_tokens("claude-3-5-sonnet") is False


class TestMaxTokensRequirement:
    """Tests for max_tokens requirement detection"""

    def test_anthropic_requires_max_tokens(self):
        """Anthropic should require max_tokens"""
        assert requires_max_tokens(LLMProvider.ANTHROPIC) is True

    def test_openai_does_not_require_max_tokens(self):
        """OpenAI should not require max_tokens"""
        assert requires_max_tokens(LLMProvider.OPENAI) is False

    def test_other_providers_do_not_require_max_tokens(self):
        """Other providers should not require max_tokens"""
        assert requires_max_tokens(LLMProvider.GOOGLE) is False
        assert requires_max_tokens(LLMProvider.GROQ) is False
        assert requires_max_tokens(LLMProvider.TOGETHER) is False


class TestProviderConfigs:
    """Tests for specific provider configurations"""

    def test_openai_config(self):
        """OpenAI config should be correct"""
        config = PROVIDER_CONFIGS[LLMProvider.OPENAI]
        assert config.name == "OpenAI"
        assert config.base_url is None  # Uses default
        assert config.default_model == "gpt-4o"
        assert config.requires_api_key is True
        assert "gpt-4o" in config.models

    def test_anthropic_config(self):
        """Anthropic config should be correct"""
        config = PROVIDER_CONFIGS[LLMProvider.ANTHROPIC]
        assert config.name == "Anthropic"
        assert "anthropic.com" in config.base_url
        assert config.requires_api_key is True

    def test_ollama_config(self):
        """Ollama config should be correct"""
        config = PROVIDER_CONFIGS[LLMProvider.OLLAMA]
        assert config.base_url == "http://localhost:11434/v1"
        assert config.requires_api_key is False

    def test_lm_studio_config(self):
        """LM Studio config should be correct"""
        config = PROVIDER_CONFIGS[LLMProvider.LM_STUDIO]
        assert config.base_url == "http://localhost:1234/v1"
        assert config.requires_api_key is False
