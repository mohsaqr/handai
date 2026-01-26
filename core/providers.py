"""
Handai LLM Provider Definitions
Provider configurations and model lists
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic (Claude)"
    GOOGLE = "Google (Gemini)"
    GROQ = "Groq"
    TOGETHER = "Together AI"
    AZURE = "Azure OpenAI"
    OPENROUTER = "OpenRouter"
    LM_STUDIO = "LM Studio (Local)"
    OLLAMA = "Ollama (Local)"
    CUSTOM = "Custom Endpoint"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider"""
    name: str
    base_url: Optional[str]
    default_model: str
    models: List[str]
    requires_api_key: bool
    description: str


PROVIDER_CONFIGS = {
    LLMProvider.OPENAI: ProviderConfig(
        name="OpenAI",
        base_url=None,
        default_model="gpt-4o",
        models=[
            # GPT-5 series
            "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
            "gpt-5.1", "gpt-5.2", "gpt-5.2-pro",
            # GPT-4 series
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            # GPT-3.5
            "gpt-3.5-turbo",
            # Reasoning models (o-series)
            "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o3-pro",
        ],
        requires_api_key=True,
        description="OpenAI's GPT models - industry standard"
    ),
    LLMProvider.ANTHROPIC: ProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-20250514",
        models=["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
        requires_api_key=True,
        description="Anthropic's Claude models - excellent reasoning"
    ),
    LLMProvider.GOOGLE: ProviderConfig(
        name="Google",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.0-flash",
        models=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
        requires_api_key=True,
        description="Google's Gemini models - multimodal capable"
    ),
    LLMProvider.GROQ: ProviderConfig(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        requires_api_key=True,
        description="Groq's ultra-fast inference - blazing speed"
    ),
    LLMProvider.TOGETHER: ProviderConfig(
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        models=[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen2.5-72B-Instruct-Turbo"
        ],
        requires_api_key=True,
        description="Together AI - wide model selection"
    ),
    LLMProvider.AZURE: ProviderConfig(
        name="Azure OpenAI",
        base_url=None,
        default_model="gpt-4o",
        models=["gpt-4o", "gpt-4", "gpt-35-turbo"],
        requires_api_key=True,
        description="Azure-hosted OpenAI models"
    ),
    LLMProvider.OPENROUTER: ProviderConfig(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4",
        models=[
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct"
        ],
        requires_api_key=True,
        description="OpenRouter - access many providers via one API"
    ),
    LLMProvider.LM_STUDIO: ProviderConfig(
        name="LM Studio",
        base_url="http://localhost:1234/v1",
        default_model="",  # Populated dynamically via fetch_local_models
        models=[],  # Populated dynamically via fetch_local_models
        requires_api_key=False,
        description="Local models via LM Studio"
    ),
    LLMProvider.OLLAMA: ProviderConfig(
        name="Ollama",
        base_url="http://localhost:11434/v1",
        default_model="",  # Populated dynamically via fetch_local_models
        models=[],  # Populated dynamically via fetch_local_models
        requires_api_key=False,
        description="Local models via Ollama"
    ),
    LLMProvider.CUSTOM: ProviderConfig(
        name="Custom",
        base_url=None,
        default_model="custom-model",
        models=["custom-model"],
        requires_api_key=False,
        description="Any OpenAI-compatible endpoint"
    ),
}


def get_provider_by_name(name: str) -> LLMProvider:
    """Get provider enum by display name"""
    for provider in LLMProvider:
        if provider.value == name:
            return provider
    raise ValueError(f"Unknown provider: {name}")


def get_provider_names() -> List[str]:
    """Get list of all provider display names"""
    return [p.value for p in LLMProvider]


def is_local_provider(provider: LLMProvider) -> bool:
    """Check if provider is a local model provider"""
    return provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.CUSTOM]


def is_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model (o1, o3, gpt-5)"""
    return any(x in model_name.lower() for x in ["gpt-5", "o1", "o3"])


def supports_json_mode(provider: LLMProvider, model_name: str) -> bool:
    """Check if provider/model combination supports JSON mode"""
    if is_local_provider(provider):
        return False
    if is_reasoning_model(model_name):
        return False
    return True


def uses_completion_tokens(model_name: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens"""
    return any(x in model_name.lower() for x in ["gpt-5", "o1", "o3"])
