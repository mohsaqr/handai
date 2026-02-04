"""
Handai v3.0 - AI Data Transformer & Generator
With sessions, logging, error handling, and auto-retry
"""

import streamlit as st
import pandas as pd
import asyncio
from openai import AsyncOpenAI
import time
import os
import json
import io
# tenacity removed - using custom retry logic in call_llm_with_retry
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Local imports
from handai_db import get_db, RunStatus, ResultStatus, LogLevel, RunResult
from handai_errors import ErrorClassifier, ErrorType, format_error_for_display

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Handai: AI Data Transformer", layout="wide", page_icon="🔄")

# Initialize database
db = get_db()

# ==========================================
# 1.5 PERSISTENT SETTINGS
# ==========================================

def load_persistent_settings():
    """Load settings from database into session state on app startup."""
    if "settings_loaded" not in st.session_state:
        saved = db.get_all_settings()
        if saved:
            for key, value in saved.items():
                if key not in st.session_state:
                    st.session_state[key] = value
        st.session_state.settings_loaded = True

def save_setting(key: str):
    """Callback to save a setting when it changes."""
    if key in st.session_state:
        db.save_setting(key, st.session_state[key])

def save_all_current_settings():
    """Save all current settings to database."""
    settings_keys = [
        "selected_provider", "api_key", "base_url", "model_name", "custom_model",
        "temperature", "max_tokens", "json_mode", "max_concurrency", "test_batch_size",
        "realtime_progress", "save_path", "auto_retry", "max_retries",
        "dataset_mode", "schema_mode", "generation_prompt", "system_prompt"
    ]
    settings = {k: st.session_state.get(k) for k in settings_keys if k in st.session_state}
    db.save_all_settings(settings)

# Load settings at startup
load_persistent_settings()

# ==========================================
# 2. LLM PROVIDER DEFINITIONS
# ==========================================

class LLMProvider(Enum):
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
        models=["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.1-8B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
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
        models=["anthropic/claude-sonnet-4", "openai/gpt-4o", "google/gemini-2.0-flash-001", "meta-llama/llama-3.3-70b-instruct"],
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

# ==========================================
# 3. DATASET GENERATOR TEMPLATES
# ==========================================

DATASET_TEMPLATES = {
    "Custom (Define Your Own)": {
        "description": "Create your own dataset schema",
        "schema": {},
        "example_prompt": "Generate a row about..."
    },
    "Interview Questions & Answers": {
        "description": "Q&A pairs for training or evaluation",
        "schema": {"question": "str", "answer": "str", "category": "str", "difficulty": "str"},
        "example_prompt": "Generate an interview question and ideal answer about {topic}. Include category and difficulty level."
    },
    "Product Reviews": {
        "description": "Synthetic product reviews with sentiment",
        "schema": {"product_name": "str", "review_text": "str", "rating": "int", "sentiment": "str", "helpful_votes": "int"},
        "example_prompt": "Generate a realistic product review for {product_type}. Include rating 1-5 and sentiment analysis."
    },
    "Customer Support Tickets": {
        "description": "Support conversations for training",
        "schema": {"ticket_id": "str", "customer_message": "str", "category": "str", "priority": "str", "suggested_response": "str"},
        "example_prompt": "Generate a customer support ticket about {issue_type}. Include category, priority, and ideal response."
    },
    "Text Classification Dataset": {
        "description": "Labeled text samples for classification",
        "schema": {"text": "str", "label": "str", "confidence": "float"},
        "example_prompt": "Generate a text sample that belongs to the category '{category}'. The text should be realistic and clearly classifiable."
    },
    "Instruction-Response Pairs": {
        "description": "Training data for instruction-following",
        "schema": {"instruction": "str", "input": "str", "output": "str", "category": "str"},
        "example_prompt": "Generate an instruction-following example for {task_type}. Include clear instruction, optional input, and ideal output."
    },
    "Code Snippets": {
        "description": "Code examples with explanations",
        "schema": {"language": "str", "code": "str", "explanation": "str", "complexity": "str", "use_case": "str"},
        "example_prompt": "Generate a {language} code snippet that demonstrates {concept}. Include explanation and use case."
    },
}

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def get_client(provider: LLMProvider, api_key: str, base_url: str, http_client) -> AsyncOpenAI:
    """Create appropriate client for provider."""
    config = PROVIDER_CONFIGS[provider]
    effective_key = api_key if config.requires_api_key else "dummy"
    effective_url = base_url or config.base_url

    # OpenRouter requires additional headers
    if provider == LLMProvider.OPENROUTER:
        return AsyncOpenAI(
            api_key=effective_key,
            base_url=effective_url,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://handai.app",
                "X-Title": "Handai Data Transformer"
            }
        )

    # Other providers use the custom http_client
    return AsyncOpenAI(
        api_key=effective_key,
        base_url=effective_url,
        http_client=http_client,
        max_retries=0
    )

def fetch_local_models(base_url: str) -> List[str]:
    """Fetch available models from LM Studio or Ollama /v1/models endpoint."""
    try:
        response = httpx.get(f"{base_url}/models", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []

def fetch_openrouter_models() -> List[str]:
    """Fetch available models from OpenRouter API."""
    try:
        response = httpx.get("https://openrouter.ai/api/v1/models", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            # Return top 50 models sorted by ID (OpenRouter returns them sorted by popularity)
            models = [m["id"] for m in data.get("data", [])[:50]]
            return models if models else []
    except Exception:
        pass
    # Fallback to default models if fetch fails
    return [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ]

def requires_max_tokens_local(provider: LLMProvider) -> bool:
    """Check if provider requires max_tokens parameter (has no server-side default)"""
    return provider == LLMProvider.ANTHROPIC


async def call_llm_with_retry(client, system_prompt: str, user_content: str, model: str,
                              temperature: Optional[float], max_tokens: Optional[int], json_mode: bool,
                              run_id: str, row_index: int, max_retries: int = 3,
                              provider: LLMProvider = None):
    """Call LLM with error handling and auto-retry for empty results.

    Args:
        temperature: None = use provider default, value = user override
        max_tokens: None = use provider default (unless provider requires it), value = user override
        provider: LLMProvider enum for checking provider requirements
    """
    attempt = 0
    last_error = None

    while attempt <= max_retries:
        start = time.time()
        try:
            # Some models (gpt-5, o1, o3) require max_completion_tokens instead of max_tokens
            uses_completion_tokens = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
            is_reasoning = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])

            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
            }

            # Temperature: only if user set it AND model supports it
            if temperature is not None and not is_reasoning:
                kwargs["temperature"] = temperature

            # Max tokens: only if user set it OR provider requires it
            if max_tokens is not None:
                if uses_completion_tokens:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
            elif provider and requires_max_tokens_local(provider):
                # Anthropic requires max_tokens - use sensible default
                kwargs["max_tokens"] = 4096

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content
            duration = time.time() - start

            # Check if result is empty/NA - auto-retry if so
            # Skip retry for reasoning models (gpt-5, o1, o3) as they may return empty during reasoning
            if not is_reasoning and ErrorClassifier.is_empty_result(output) and attempt < max_retries:
                db.log(LogLevel.WARNING, f"Empty result for row {row_index}, retrying...",
                       {"attempt": attempt, "output": output[:100] if output else None}, run_id=run_id)
                attempt += 1
                await asyncio.sleep(1)  # Brief delay before retry
                continue

            return output, duration, None, attempt

        except Exception as e:
            duration = time.time() - start
            error_info = ErrorClassifier.classify(e)

            # Log the error
            db.log(LogLevel.ERROR, f"Error on row {row_index}: {error_info.message}",
                   {"error_type": error_info.error_type.value, "original": error_info.original_error,
                    "attempt": attempt}, run_id=run_id)

            # Check if we should retry
            should_retry, delay = ErrorClassifier.should_auto_retry(error_info, attempt, max_retries)

            if should_retry:
                db.log(LogLevel.INFO, f"Retrying row {row_index} after {delay}s delay",
                       {"attempt": attempt + 1}, run_id=run_id)
                await asyncio.sleep(delay)
                attempt += 1
                last_error = error_info
                continue
            else:
                return None, duration, error_info, attempt

    # Max retries exceeded
    return None, 0, last_error, attempt

def get_current_settings(include_sensitive: bool = False) -> dict:
    """Get current settings from session state for saving.

    Args:
        include_sensitive: If False, excludes API keys (for session storage)
    """
    # Settings to save with sessions (NO API keys)
    session_keys = [
        "selected_provider", "base_url", "model_name", "custom_model",
        "temperature", "max_tokens", "json_mode", "max_concurrency", "test_batch_size",
        "realtime_progress", "save_path", "auto_retry", "max_retries",
        "dataset_mode", "schema_mode"
    ]

    # Only include API key for app_settings persistence (encrypted storage would be better)
    if include_sensitive:
        session_keys.append("api_key")

    return {k: st.session_state.get(k) for k in session_keys if k in st.session_state}

# ==========================================
# 5. SIDEBAR SETTINGS
# ==========================================

with st.sidebar:
    st.title("Settings")

    # Session info
    if "current_session_id" in st.session_state:
        session = db.get_session(st.session_state.current_session_id)
        if session:
            st.caption(f"📁 Session: {session.name}")

    st.divider()

    # ==========================================
    # PROVIDER SELECTION
    # ==========================================
    st.header("🤖 AI Provider")

    provider_names = [p.value for p in LLMProvider]

    # Get saved provider index or default to 0
    saved_provider = st.session_state.get("selected_provider", provider_names[0])
    default_provider_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

    selected_provider_name = st.selectbox(
        "Select Provider",
        provider_names,
        index=default_provider_idx,
        key="selected_provider",
        on_change=lambda: save_setting("selected_provider")
    )
    selected_provider = LLMProvider(selected_provider_name)
    provider_config = PROVIDER_CONFIGS[selected_provider]

    st.caption(f"ℹ️ {provider_config.description}")

    # API Key
    if provider_config.requires_api_key:
        api_key_help = None
        if selected_provider == LLMProvider.OPENROUTER:
            api_key_help = "Get your API key at openrouter.ai/keys"
        api_key = st.text_input(
            "API Key",
            type="password",
            key="api_key",
            help=api_key_help,
            on_change=lambda: save_setting("api_key")
        )
        if selected_provider == LLMProvider.OPENROUTER:
            st.caption("🔑 Get your key at [openrouter.ai/keys](https://openrouter.ai/keys)")
    else:
        api_key = "dummy"
        st.info("No API key required for local models")

    # Base URL
    if selected_provider in [LLMProvider.AZURE, LLMProvider.CUSTOM, LLMProvider.LM_STUDIO, LLMProvider.OLLAMA]:
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.get("base_url", provider_config.base_url or ""),
            key="base_url",
            help="Full URL to the API endpoint",
            on_change=lambda: save_setting("base_url")
        )
    else:
        base_url = provider_config.base_url

    # Model Selection
    st.subheader("Model")

    # Determine available models for this provider
    if selected_provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA]:
        # Dynamic model fetching for local providers
        local_models_key = f"local_models_{selected_provider.value}"
        effective_url = base_url or provider_config.base_url

        # Initialize or fetch models for this provider
        if local_models_key not in st.session_state:
            st.session_state[local_models_key] = fetch_local_models(effective_url)

        available_models = st.session_state.get(local_models_key, [])

        # Refresh button for local providers
        col_model, col_refresh = st.columns([3, 1])
        with col_refresh:
            if st.button("🔄", key="refresh_models", help="Refresh model list"):
                st.session_state[local_models_key] = fetch_local_models(effective_url)
                st.rerun()
    elif selected_provider == LLMProvider.OPENROUTER:
        # Dynamic model fetching for OpenRouter
        openrouter_models_key = "openrouter_models"

        # Initialize or fetch models
        if openrouter_models_key not in st.session_state:
            st.session_state[openrouter_models_key] = fetch_openrouter_models()

        available_models = st.session_state.get(openrouter_models_key, [])

        # Refresh button for OpenRouter
        col_model, col_refresh = st.columns([3, 1])
        with col_refresh:
            if st.button("🔄", key="refresh_openrouter_models", help="Refresh model list from OpenRouter"):
                st.session_state[openrouter_models_key] = fetch_openrouter_models()
                st.rerun()
    else:
        available_models = provider_config.models
        col_model = st.container()

    # Always allow custom model input
    with col_model if selected_provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.OPENROUTER] else st.container():
        use_custom = st.checkbox("Use custom model name", key="use_custom_model", value=st.session_state.get("use_custom_model", False))

        if use_custom or not available_models:
            # Direct text input for custom model
            model_name = st.text_input(
                "Model Name",
                value=st.session_state.get("custom_model", ""),
                key="custom_model",
                placeholder="Enter model name (e.g., gpt-4o, meta-llama-3.1-8b-instruct)",
                on_change=lambda: save_setting("custom_model")
            )
        else:
            # Dropdown with available models
            saved_model = st.session_state.get("model_name", available_models[0] if available_models else "")
            default_idx = available_models.index(saved_model) if saved_model in available_models else 0

            model_name = st.selectbox(
                "Select Model",
                available_models,
                index=default_idx,
                key="model_name",
                on_change=lambda: save_setting("model_name")
            )

    st.divider()

    # ==========================================
    # LLM CONTROL OPTIONS
    # ==========================================
    st.header("🎛️ LLM Controls")

    def on_temperature_change():
        save_setting("temperature")
        st.session_state["temperature_modified"] = True

    def on_max_tokens_change():
        save_setting("max_tokens")
        st.session_state["max_tokens_modified"] = True

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get("temperature", 0.0),
        step=0.1,
        key="temperature",
        help="0 = deterministic, 2 = very creative. Leave at default to use provider's default.",
        on_change=on_temperature_change
    )

    max_tokens = st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=128000,
        value=st.session_state.get("max_tokens", 2048),
        step=256,
        key="max_tokens",
        help="Maximum response length. Leave at default to use provider's default.",
        on_change=on_max_tokens_change
    )

    json_mode = st.checkbox(
        "JSON Mode",
        value=st.session_state.get("json_mode", False),
        key="json_mode",
        help="Force structured JSON output (supported by most providers)",
        on_change=lambda: save_setting("json_mode")
    )

    if json_mode and selected_provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.CUSTOM]:
        st.warning("⚠️ JSON mode may not work with local models. Will be skipped automatically.")

    st.divider()

    # ==========================================
    # PERFORMANCE SETTINGS
    # ==========================================
    st.header("🚀 Performance")

    max_concurrency = st.slider(
        "Max Concurrent Requests",
        1, 50,
        st.session_state.get("max_concurrency", 5),
        key="max_concurrency",
        on_change=lambda: save_setting("max_concurrency")
    )
    test_batch_size = st.number_input(
        "Test Batch Size",
        1, 1000,
        st.session_state.get("test_batch_size", 10),
        key="test_batch_size",
        on_change=lambda: save_setting("test_batch_size")
    )

    auto_retry = st.checkbox(
        "Auto-retry empty/failed",
        value=st.session_state.get("auto_retry", True),
        key="auto_retry",
        help="Automatically retry rows that return empty or error",
        on_change=lambda: save_setting("auto_retry")
    )

    max_retries = st.number_input(
        "Max Retries",
        1, 5,
        st.session_state.get("max_retries", 3),
        key="max_retries",
        on_change=lambda: save_setting("max_retries")
    ) if auto_retry else 0

    realtime_progress = st.checkbox(
        "Real-Time Progress",
        value=st.session_state.get("realtime_progress", True),
        key="realtime_progress",
        help="Update UI every row (slightly slower)",
        on_change=lambda: save_setting("realtime_progress")
    )

    save_path = st.text_input(
        "Auto-Save Path",
        placeholder="/path/to/save/results",
        key="save_path",
        on_change=lambda: save_setting("save_path")
    )

# ==========================================
# 6. MAIN INTERFACE
# ==========================================

st.title("🔄 Handai: AI Data Transformer")

# Mode Selection
st.header("1. Choose Mode")
dataset_mode = st.radio(
    "What would you like to do?",
    ["Transform Existing Dataset", "Generate New Dataset"],
    horizontal=True,
    key="dataset_mode",
    on_change=lambda: save_setting("dataset_mode")
)

# ==========================================
# SAMPLE TEST DATA
# ==========================================
SAMPLE_DATA = pd.DataFrame({
    "text": [
        "I absolutely love this product! Best purchase ever.",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
        "Amazing quality and fast shipping. Very happy!",
        "Broke after one week. Complete waste of money.",
        "Decent value for the price. Could be better.",
        "This exceeded all my expectations! Fantastic!",
        "Not worth it. Save your money.",
        "Pretty good overall. Minor issues but satisfied.",
        "Outstanding service and product. Will buy again!"
    ],
    "category": ["Electronics", "Clothing", "Home", "Electronics", "Toys",
                 "Home", "Beauty", "Electronics", "Clothing", "Food"],
    "price": [299.99, 45.00, 89.50, 199.99, 24.99,
              67.00, 35.50, 599.00, 55.00, 28.99]
})

# ==========================================
# MODE: TRANSFORM EXISTING DATASET
# ==========================================
if dataset_mode == "Transform Existing Dataset":
    st.header("2. Upload Data")

    # Option to use sample data
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls", "json"])
    with col_sample:
        st.write("")  # Spacing
        use_sample = st.button("📋 Use Sample Data", help="Load sample data for testing")

    if use_sample:
        st.session_state["use_sample_data"] = True

    # Determine data source
    df = None
    data_source = None

    if uploaded_file:
        # Load from uploaded file
        file_ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_ext == "json":
                df = pd.read_json(uploaded_file)
            data_source = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

        if df is None or df.empty:
            st.warning("The uploaded file is empty or could not be parsed.")
            st.stop()

    elif st.session_state.get("use_sample_data"):
        # Use sample data
        df = SAMPLE_DATA.copy()
        data_source = "sample_data.csv"
        st.success("Using sample data (10 product reviews)")

    if df is not None:
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect("Active Columns (sent to AI)", all_cols, default=all_cols)

        st.dataframe(df.head(), height=200)

        st.header("3. Define Transformation")
        system_prompt = st.text_area(
            "AI Instructions",
            height=250,
            placeholder="Example: Extract the main topic from each text and classify sentiment as positive/negative/neutral...",
            key="system_prompt",
            value=st.session_state.get("system_prompt", "")
        )

        if json_mode:
            st.info("💡 JSON Mode enabled - instruct the AI to return valid JSON in your prompt")

        st.header("4. Execute")

        c1, c2 = st.columns(2)
        run_test = c1.button(f"🧪 Test ({test_batch_size} rows)", type="primary")
        run_full = c2.button("🚀 Full Run", type="secondary")

        if run_test or run_full:
            # Validate inputs
            if not system_prompt or not system_prompt.strip():
                st.error("Please enter AI instructions before running.")
                st.stop()

            if not api_key and provider_config.requires_api_key:
                st.error("Please enter an API key for the selected provider.")
                st.stop()

            target_df = df.head(test_batch_size).copy() if run_test else df.copy()
            target_df = target_df.reset_index(drop=True)  # Ensure sequential 0-based index
            use_realtime = True if run_test else realtime_progress
            run_type = "test" if run_test else "full"

            # Create or get session
            if "current_session_id" not in st.session_state:
                session = db.create_session("transform", get_current_settings())
                st.session_state.current_session_id = session.session_id
                db.log(LogLevel.INFO, f"Created new session: {session.name}",
                       session_id=session.session_id)

            # Create run
            # Create run with full settings snapshot
            run = db.create_run(
                session_id=st.session_state.current_session_id,
                run_type=run_type,
                provider=selected_provider.value,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                schema={},
                variables={},
                input_file=data_source,
                input_rows=len(target_df),
                json_mode=json_mode,
                max_concurrency=max_concurrency,
                auto_retry=auto_retry,
                max_retry_attempts=max_retries,
                run_settings=get_current_settings()
            )

            db.log(LogLevel.INFO, f"Started {run_type} run with {len(target_df)} rows",
                   {"provider": selected_provider.value, "model": model_name},
                   run_id=run.run_id, session_id=st.session_state.current_session_id)

            # Effective JSON mode (skip for local providers and reasoning models)
            is_reasoning_model = any(x in model_name.lower() for x in ["gpt-5", "o1", "o3"])
            effective_json_mode = json_mode and selected_provider not in [
                LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.CUSTOM
            ] and not is_reasoning_model

            # Determine effective temperature and max_tokens (None if not modified by user)
            effective_temperature = temperature if st.session_state.get("temperature_modified") else None
            effective_max_tokens = max_tokens if st.session_state.get("max_tokens_modified") else None

            async def run_transformation():
                http_client = httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    timeout=httpx.Timeout(120.0, connect=10.0)
                )

                client = get_client(selected_provider, api_key, base_url, http_client)
                semaphore = asyncio.Semaphore(max_concurrency)

                async def process_row(row_idx, row):
                    async with semaphore:
                        if selected_cols:
                            row_data = row[selected_cols].to_json()
                        else:
                            row_data = row.to_json()

                        user_content = f"Data: {row_data}"

                        output, duration, error_info, attempts = await call_llm_with_retry(
                            client, system_prompt, user_content, model_name,
                            effective_temperature, effective_max_tokens, effective_json_mode,
                            run.run_id, row_idx, max_retries if auto_retry else 0,
                            selected_provider
                        )

                        if error_info:
                            result = RunResult.create(
                                run_id=run.run_id,
                                row_index=row_idx,
                                input_data=row_data,
                                output=f"Error: {error_info.message}",
                                status=ResultStatus.ERROR,
                                latency=duration,
                                error_type=error_info.error_type.value,
                                error_message=error_info.original_error,
                                retry_attempt=attempts
                            )
                            return row_idx, {"output": f"Error: {error_info.message}",
                                           "latency": round(duration, 3),
                                           "error": error_info}, False, result
                        else:
                            result = RunResult.create(
                                run_id=run.run_id,
                                row_index=row_idx,
                                input_data=row_data,
                                output=output,
                                status=ResultStatus.SUCCESS,
                                latency=duration,
                                retry_attempt=attempts
                            )
                            return row_idx, {"output": output, "latency": round(duration, 3)}, True, result

                tasks = [process_row(i, row) for i, row in target_df.iterrows()]

                # Progress UI
                st.subheader("Processing Status")
                status_metrics = st.empty()
                progress_bar = st.progress(0)
                log_placeholder = st.empty()
                error_expander = st.expander("🔴 Errors", expanded=False)
                log_text = ""
                errors_text = ""

                results_map = {}
                all_results = []
                total = len(tasks)
                completed = 0
                success_count = 0
                error_count = 0
                retry_count = 0

                for future in asyncio.as_completed(tasks):
                    idx, res_dict, success, db_result = await future
                    results_map[idx] = res_dict
                    all_results.append(db_result)
                    completed += 1

                    if db_result.retry_attempt > 0:
                        retry_count += db_result.retry_attempt

                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        error_info = res_dict.get("error")
                        if error_info:
                            errors_text = f"❌ Row {idx}: {error_info.message}\n   💡 {error_info.suggestion}\n\n" + errors_text

                    if use_realtime or completed % 10 == 0 or completed == total:
                        progress_bar.progress(completed / total)
                        status = "✅" if success else "❌"
                        log_line = f"{status} Row {idx}: {res_dict['latency']}s\n"
                        log_text = log_line + log_text
                        if log_text.count('\n') > 10:
                            log_text = '\n'.join(log_text.split('\n')[:10])
                        log_placeholder.code(log_text)
                        status_metrics.markdown(
                            f"**Progress:** {completed}/{total} | ✅ {success_count} | ❌ {error_count} | 🔄 {retry_count} retries"
                        )

                        if errors_text:
                            with error_expander:
                                st.markdown(errors_text[:2000])

                    # Auto-save
                    if save_path and (completed % 5 == 0 or completed == total):
                        try:
                            os.makedirs(save_path, exist_ok=True)
                            current_indices = sorted(results_map.keys())
                            temp_df = target_df.loc[current_indices].copy()
                            temp_df["ai_output"] = [results_map[i].get("output") for i in current_indices]
                            temp_df["latency_s"] = [results_map[i].get("latency", 0) for i in current_indices]
                            temp_df.to_csv(os.path.join(save_path, "partial_handai.csv"), index=False)
                        except:
                            pass

                # Save all results to database
                db.save_results_batch(all_results)

                await http_client.aclose()

                # Calculate stats
                latencies = [r.latency for r in all_results if r.status == ResultStatus.SUCCESS.value]
                avg_latency = sum(latencies) / len(latencies) if latencies else 0

                return [results_map[i] for i in sorted(results_map.keys())], success_count, error_count, retry_count, avg_latency

            # Run the processing
            start_time = time.time()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results, success_count, error_count, retry_count, avg_latency = loop.run_until_complete(run_transformation())
            total_duration = time.time() - start_time

            # Update run status
            final_status = RunStatus.COMPLETED if error_count == 0 else RunStatus.COMPLETED
            db.update_run_status(
                run.run_id, final_status,
                success_count=success_count,
                error_count=error_count,
                retry_count=retry_count,
                avg_latency=avg_latency,
                total_duration=total_duration
            )

            db.log(LogLevel.INFO, f"Run completed: {success_count} success, {error_count} errors",
                   {"duration": total_duration, "avg_latency": avg_latency},
                   run_id=run.run_id)

            # Add results to dataframe
            target_df["ai_output"] = [r.get("output", "N/A") for r in results]
            target_df["latency_s"] = [r.get("latency", 0) for r in results]

            st.success(f"Complete! ✅ {success_count} | ❌ {error_count} | 🔄 {retry_count} retries | Avg: {avg_latency:.2f}s | Total: {total_duration:.1f}s")
            st.dataframe(target_df, use_container_width=True)

            # Result Inspector
            st.divider()
            st.subheader("🔍 Result Inspector")

            c_insp1, c_insp2 = st.columns([1, 3])
            with c_insp1:
                max_row = max(0, len(target_df) - 1)  # Ensure non-negative
                row_to_inspect = st.number_input("Row", min_value=0, max_value=max_row, value=0)
            with c_insp2:
                inspect_mode = st.radio("View", ["AI Output", "Full Row"], horizontal=True)

            if not target_df.empty and row_to_inspect < len(target_df):
                sel_row = target_df.iloc[row_to_inspect]
                if inspect_mode == "AI Output":
                    st.code(sel_row.get("ai_output", ""), language="markdown")
                else:
                    st.code(sel_row.to_json(indent=2), language="json")

            # Download options
            st.subheader("📥 Download Results")
            dl_col1, dl_col2, dl_col3 = st.columns(3)

            with dl_col1:
                csv = target_df.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, "handai_results.csv", "text/csv", use_container_width=True)

            with dl_col2:
                excel_buffer = io.BytesIO()
                target_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button("Excel", excel_buffer, "handai_results.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

            with dl_col3:
                json_data = target_df.to_json(orient="records", indent=2)
                st.download_button("JSON", json_data, "handai_results.json", "application/json",
                                   use_container_width=True)

# ==========================================
# MODE: GENERATE NEW DATASET
# ==========================================
else:
    st.header("2. Describe Your Dataset")

    # Initialize session state for dynamic fields
    if "custom_fields" not in st.session_state:
        st.session_state.custom_fields = []

    # Free-form description
    generation_prompt = st.text_area(
        "What kind of data do you want to generate?",
        height=120,
        placeholder="Example: Generate realistic customer profiles with names, emails, purchase history...",
        key="generation_prompt",
        help="Just describe what you need - the AI will figure out the structure."
    )

    # Generation settings inline
    col1, col2, col3 = st.columns(3)
    with col1:
        num_rows = st.number_input("Rows to Generate", 1, 10000, 100, key="num_rows_to_generate")
    with col2:
        variation_level = st.select_slider(
            "Variation",
            options=["Low", "Medium", "High", "Maximum"],
            value="Medium"
        )
    with col3:
        output_format = st.selectbox("Output Format", ["Auto-detect", "Structured JSON", "Free text"])

    variation_temps = {"Low": 0.3, "Medium": 0.7, "High": 1.0, "Maximum": 1.5}
    gen_temperature = variation_temps[variation_level]

    # Schema mode selection
    schema_mode = st.radio(
        "Schema Definition",
        ["Free-form (AI decides structure)", "Custom Fields (I'll define)", "Use Template"],
        horizontal=True,
        key="schema_mode",
        on_change=lambda: save_setting("schema_mode")
    )

    schema = {}
    use_freeform = False

    if schema_mode == "Free-form (AI decides structure)":
        use_freeform = True
        st.info("💡 The AI will determine the best structure based on your description above.")

    elif schema_mode == "Custom Fields (I'll define)":
        with st.expander("🔧 Schema Builder - Add Fields", expanded=True):
            st.caption("Add fields one by one to define your dataset structure")

            with st.form("add_field_form", clear_on_submit=True):
                fc1, fc2, fc3, fc4 = st.columns([2, 1, 3, 1])
                with fc1:
                    new_field_name = st.text_input("Field Name", placeholder="e.g., customer_name")
                with fc2:
                    new_field_type = st.selectbox("Type", ["text", "number", "decimal", "boolean", "date", "list", "json"])
                with fc3:
                    new_field_desc = st.text_input("Description (optional)", placeholder="e.g., Full name")
                with fc4:
                    st.write("")
                    add_field = st.form_submit_button("➕ Add", use_container_width=True)

                if add_field and new_field_name:
                    st.session_state.custom_fields.append({
                        "name": new_field_name,
                        "type": new_field_type,
                        "description": new_field_desc
                    })
                    st.rerun()

            if st.session_state.custom_fields:
                st.write("**Current Fields:**")
                for idx, field in enumerate(st.session_state.custom_fields):
                    fc1, fc2, fc3, fc4 = st.columns([2, 1, 3, 1])
                    with fc1:
                        st.text(field["name"])
                    with fc2:
                        st.text(field["type"])
                    with fc3:
                        st.text(field.get("description", "-"))
                    with fc4:
                        if st.button("🗑️", key=f"del_{idx}", help="Remove field"):
                            st.session_state.custom_fields.pop(idx)
                            st.rerun()

                if st.button("Clear All Fields"):
                    st.session_state.custom_fields = []
                    st.rerun()

                type_map = {"text": "str", "number": "int", "decimal": "float", "boolean": "bool",
                           "date": "str", "list": "list", "json": "json"}
                schema = {f["name"]: type_map.get(f["type"], "str") for f in st.session_state.custom_fields}
                st.json(schema)
            else:
                st.warning("No fields added yet.")

    else:  # Use Template
        with st.expander("📋 Choose a Template", expanded=True):
            template_names = list(DATASET_TEMPLATES.keys())
            selected_template = st.selectbox("Template", template_names, key="selected_template")
            template = DATASET_TEMPLATES[selected_template]
            st.caption(f"📝 {template['description']}")

            if selected_template != "Custom (Define Your Own)":
                schema = template["schema"]
                st.json(schema)

    # Variables section
    with st.expander("🔄 Variables - Cycle Through Values"):
        st.caption("Define values to cycle through for each row")

        if "gen_variables" not in st.session_state:
            st.session_state.gen_variables = []

        with st.form("add_var_form", clear_on_submit=True):
            vc1, vc2, vc3 = st.columns([1, 3, 1])
            with vc1:
                new_var_name = st.text_input("Variable", placeholder="topic")
            with vc2:
                new_var_values = st.text_input("Values (comma-separated)", placeholder="sports, tech, health")
            with vc3:
                st.write("")
                add_var = st.form_submit_button("➕ Add", use_container_width=True)

            if add_var and new_var_name and new_var_values:
                st.session_state.gen_variables.append({
                    "name": new_var_name,
                    "values": [v.strip() for v in new_var_values.split(",")]
                })
                st.rerun()

        variables = {}
        if st.session_state.gen_variables:
            for idx, var in enumerate(st.session_state.gen_variables):
                vc1, vc2, vc3 = st.columns([1, 3, 1])
                with vc1:
                    st.text(f"{{{var['name']}}}")
                with vc2:
                    st.text(", ".join(var["values"]))
                with vc3:
                    if st.button("🗑️", key=f"del_var_{idx}"):
                        st.session_state.gen_variables.pop(idx)
                        st.rerun()
                variables[var["name"]] = var["values"]

    st.header("3. Generate")

    c1, c2 = st.columns(2)
    gen_test = c1.button(f"🧪 Test ({min(10, num_rows)} rows)", type="primary")
    gen_full = c2.button("🚀 Generate All", type="secondary")

    if gen_test or gen_full:
        # Validate inputs
        if not generation_prompt or not generation_prompt.strip():
            st.error("Please describe what data you want to generate.")
            st.stop()

        if not api_key and provider_config.requires_api_key:
            st.error("Please enter an API key for the selected provider.")
            st.stop()

        target_count = min(10, num_rows) if gen_test else num_rows
        run_type = "test" if gen_test else "full"

        # Create session if needed
        if "current_session_id" not in st.session_state:
            session = db.create_session("generate", get_current_settings())
            st.session_state.current_session_id = session.session_id

        # Create run
        # Create run with full settings snapshot
        run = db.create_run(
            session_id=st.session_state.current_session_id,
            run_type=run_type,
            provider=selected_provider.value,
            model=model_name,
            temperature=gen_temperature,
            max_tokens=max_tokens,
            system_prompt=generation_prompt,
            schema=schema,
            variables=variables,
            input_file="generated",
            input_rows=target_count,
            json_mode=True,  # Generation always uses JSON
            max_concurrency=max_concurrency,
            auto_retry=auto_retry,
            max_retry_attempts=max_retries,
            run_settings=get_current_settings()
        )

        db.log(LogLevel.INFO, f"Started generation run: {target_count} rows",
               {"provider": selected_provider.value, "model": model_name, "schema": schema},
               run_id=run.run_id)

        # Build system prompt
        if use_freeform:
            gen_system = """You are a synthetic data generator. Based on the user's description, generate realistic, diverse data.

CRITICAL RULES:
1. Return ONLY valid JSON - a single object with appropriate fields
2. Determine the best schema based on the user's description
3. Each response should be unique and realistic
4. Vary the content naturally
5. Do not include any explanation, just the JSON object
6. Use sensible field names in snake_case"""
        else:
            schema_str = json.dumps(schema, indent=2)
            gen_system = f"""You are a synthetic data generator. Generate realistic, diverse data following this exact schema:

{schema_str}

CRITICAL RULES:
1. Return ONLY valid JSON matching the schema exactly
2. Each response should be unique and realistic
3. Vary the content naturally
4. Do not include any explanation, just the JSON object"""

        is_reasoning_model = any(x in model_name.lower() for x in ["gpt-5", "o1", "o3"])
        supports_json = selected_provider not in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.CUSTOM] and not is_reasoning_model

        # For generation, temperature is always set via variation slider (explicit user choice)
        # max_tokens follows the minimal parameter principle
        effective_gen_max_tokens = max_tokens if st.session_state.get("max_tokens_modified") else None

        async def run_generation():
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                timeout=httpx.Timeout(120.0, connect=10.0)
            )

            client = get_client(selected_provider, api_key, base_url, http_client)
            semaphore = asyncio.Semaphore(max_concurrency)

            async def generate_row(row_idx):
                async with semaphore:
                    prompt = generation_prompt
                    for var_name, var_values in variables.items():
                        if f"{{{var_name}}}" in prompt:
                            value = var_values[row_idx % len(var_values)]
                            prompt = prompt.replace(f"{{{var_name}}}", value)

                    prompt = f"{prompt}\n\nGenerate row #{row_idx + 1}:"

                    output, duration, error_info, attempts = await call_llm_with_retry(
                        client, gen_system, prompt, model_name,
                        gen_temperature, effective_gen_max_tokens, supports_json,
                        run.run_id, row_idx, max_retries if auto_retry else 0,
                        selected_provider
                    )

                    if error_info:
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=prompt,
                            output=f"Error: {error_info.message}",
                            status=ResultStatus.ERROR,
                            latency=duration,
                            error_type=error_info.error_type.value,
                            error_message=error_info.original_error,
                            retry_attempt=attempts
                        )
                        return row_idx, {"error": error_info.message}, round(duration, 3), False, result

                    # Parse JSON
                    try:
                        parsed = json.loads(output)
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=prompt,
                            output=output,
                            status=ResultStatus.SUCCESS,
                            latency=duration,
                            retry_attempt=attempts
                        )
                        return row_idx, parsed, round(duration, 3), True, result
                    except json.JSONDecodeError:
                        import re
                        # Try to extract JSON - handles nested objects better
                        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', output, re.DOTALL)
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group())
                                result = RunResult.create(
                                    run_id=run.run_id,
                                    row_index=row_idx,
                                    input_data=prompt,
                                    output=json_match.group(),
                                    status=ResultStatus.SUCCESS,
                                    latency=duration,
                                    retry_attempt=attempts
                                )
                                return row_idx, parsed, round(duration, 3), True, result
                            except:
                                pass

                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=prompt,
                            output=output,
                            status=ResultStatus.ERROR,
                            latency=duration,
                            error_type="json_parse_error",
                            error_message="Could not parse JSON from response",
                            retry_attempt=attempts
                        )
                        return row_idx, {"raw_output": output}, round(duration, 3), False, result

            tasks = [generate_row(i) for i in range(target_count)]

            # Progress UI
            st.subheader("Generation Progress")
            status_metrics = st.empty()
            progress_bar = st.progress(0)
            log_placeholder = st.empty()
            error_expander = st.expander("🔴 Errors", expanded=False)
            log_text = ""
            errors_text = ""

            results = []
            all_db_results = []
            total = len(tasks)
            completed = 0
            success_count = 0
            error_count = 0
            retry_count = 0

            for future in asyncio.as_completed(tasks):
                idx, data, latency, success, db_result = await future
                results.append((idx, data, latency, success))
                all_db_results.append(db_result)
                completed += 1

                if db_result.retry_attempt > 0:
                    retry_count += db_result.retry_attempt

                if success:
                    success_count += 1
                else:
                    error_count += 1
                    errors_text = f"❌ Row {idx}: {data.get('error', 'Parse error')}\n" + errors_text

                progress_bar.progress(completed / total)
                status = "✅" if success else "❌"
                log_line = f"{status} Row {idx}: {latency}s\n"
                log_text = log_line + log_text
                if log_text.count('\n') > 10:
                    log_text = '\n'.join(log_text.split('\n')[:10])
                log_placeholder.code(log_text)
                status_metrics.markdown(
                    f"**Progress:** {completed}/{total} | ✅ {success_count} | ❌ {error_count} | 🔄 {retry_count} retries"
                )

                if errors_text:
                    with error_expander:
                        st.markdown(errors_text[:2000])

            # Save results
            db.save_results_batch(all_db_results)

            await http_client.aclose()

            results.sort(key=lambda x: x[0])
            return [(r[1], r[2], r[3]) for r in results], success_count, error_count, retry_count

        start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results, success_count, error_count, retry_count = loop.run_until_complete(run_generation())
        total_duration = time.time() - start_time

        # Update run
        latencies = [r[1] for r in results if r[2]]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        db.update_run_status(
            run.run_id, RunStatus.COMPLETED,
            success_count=success_count,
            error_count=error_count,
            retry_count=retry_count,
            avg_latency=avg_latency,
            total_duration=total_duration
        )

        # Build DataFrame
        rows = []
        latencies_list = []
        for data, latency, success in results:
            # Handle case where LLM returns a list instead of dict
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    # Take first item if it's a list of dicts
                    data = data[0]
                else:
                    # Wrap list in a dict
                    data = {"data": data}

            # Now data is guaranteed to be a dict
            if success and not data.get("error") and not data.get("raw_output"):
                rows.append(data)
            else:
                if schema:
                    error_row = {k: None for k in schema.keys()}
                else:
                    error_row = {}
                error_row["_error"] = data.get("error") or data.get("raw_output") or "Unknown error"
                rows.append(error_row)
            latencies_list.append(latency)

        generated_df = pd.DataFrame(rows)
        generated_df["_latency_s"] = latencies_list

        success_rate = success_count / len(results) * 100 if results else 0

        st.success(f"Generated {len(generated_df)} rows | ✅ {success_rate:.1f}% | ❌ {error_count} | 🔄 {retry_count} retries | Avg: {avg_latency:.2f}s")
        st.dataframe(generated_df, use_container_width=True)

        # Download options
        st.subheader("📥 Download Generated Dataset")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            csv = generated_df.to_csv(index=False).encode('utf-8')
            st.download_button("CSV", csv, "handai_generated.csv", "text/csv", use_container_width=True)

        with dl_col2:
            excel_buffer = io.BytesIO()
            generated_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button("Excel", excel_buffer, "handai_generated.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        with dl_col3:
            json_data = generated_df.to_json(orient="records", indent=2)
            st.download_button("JSON", json_data, "handai_generated.json", "application/json",
                               use_container_width=True)

# ==========================================
# 7. FOOTER
# ==========================================
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Handai v3.0 - Multi-Provider AI Data Transformer & Generator")
with col2:
    if st.button("📜 View History"):
        st.switch_page("pages/1_History.py")
