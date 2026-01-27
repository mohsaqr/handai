"""
Model Selector Component
UI for selecting and configuring LLM models.
When configured providers are active, uses their model lists.
"""

import streamlit as st
from typing import List, Optional
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.llm_client import fetch_local_models, fetch_openrouter_models
from ui.state import save_setting
from database import get_db


def _get_configured_provider_models(provider: LLMProvider) -> Optional[List[str]]:
    """
    Get model list for a configured provider.
    Returns seeded models if available, otherwise the configured default model
    combined with static PROVIDER_CONFIGS models.
    Returns None if no configured providers are active (legacy fallback).
    """
    db = get_db()
    providers = db.get_enabled_configured_providers()
    for p in providers:
        if p.provider_type == provider.value:
            # First check for seeded models in session state
            seeded = st.session_state.get(f"seeded_models_{p.id}")
            if seeded:
                return seeded
            # Otherwise build list from default model + static config
            config = PROVIDER_CONFIGS.get(provider)
            models = list(config.models) if config and config.models else []
            if p.default_model and p.default_model not in models:
                models.insert(0, p.default_model)
            return models if models else None
    return None


def render_model_selector(
    provider: LLMProvider,
    base_url: Optional[str] = None,
    key_prefix: str = "main"
) -> str:
    """
    Render model selection UI for a provider.

    Args:
        provider: The selected LLM provider
        base_url: Optional base URL for local providers
        key_prefix: Prefix for widget keys

    Returns:
        Selected model name
    """
    st.subheader("Model")

    provider_config = PROVIDER_CONFIGS[provider]

    # Check configured providers first (new system)
    configured_models = _get_configured_provider_models(provider)
    if configured_models:
        available_models = configured_models
        col_model = st.container()
    # Legacy: dynamic model fetching for local providers
    elif provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA]:
        local_models_key = f"local_models_{provider.value}"
        effective_url = base_url or provider_config.base_url

        if local_models_key not in st.session_state:
            st.session_state[local_models_key] = fetch_local_models(effective_url)

        available_models = st.session_state.get(local_models_key, [])

        col_model, col_refresh = st.columns([3, 1])
        with col_refresh:
            if st.button("", key=f"{key_prefix}_refresh_models", help="Refresh model list"):
                st.session_state[local_models_key] = fetch_local_models(effective_url)
                st.rerun()

    elif provider == LLMProvider.OPENROUTER:
        openrouter_models_key = "openrouter_models"

        if openrouter_models_key not in st.session_state:
            st.session_state[openrouter_models_key] = fetch_openrouter_models()

        available_models = st.session_state.get(openrouter_models_key, [])

        col_model, col_refresh = st.columns([3, 1])
        with col_refresh:
            if st.button("", key=f"{key_prefix}_refresh_openrouter", help="Refresh model list from OpenRouter"):
                st.session_state[openrouter_models_key] = fetch_openrouter_models()
                st.rerun()
    else:
        available_models = provider_config.models
        col_model = st.container()

    # Custom model checkbox
    with col_model if provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.OPENROUTER] and not configured_models else st.container():
        use_custom = st.checkbox(
            "Use custom model name",
            key=f"{key_prefix}_use_custom_model",
            value=st.session_state.get("use_custom_model", False)
        )
        st.session_state.use_custom_model = use_custom

        if use_custom or not available_models:
            model_name = st.text_input(
                "Model Name",
                value=st.session_state.get("custom_model", ""),
                key=f"{key_prefix}_custom_model",
                placeholder="Enter model name (e.g., gpt-4o, meta-llama-3.1-8b-instruct)",
                on_change=lambda: save_setting("custom_model")
            )
            st.session_state.custom_model = model_name
        else:
            saved_model = st.session_state.get("model_name", available_models[0] if available_models else "")
            default_idx = available_models.index(saved_model) if saved_model in available_models else 0

            model_name = st.selectbox(
                "Select Model",
                available_models,
                index=default_idx,
                key=f"{key_prefix}_model_name",
                on_change=lambda: save_setting("model_name")
            )
            st.session_state.model_name = model_name

    return model_name


def render_model_selector_compact(
    provider: LLMProvider,
    available_models: List[str],
    key_prefix: str = "compact"
) -> str:
    """
    Render a compact model selector (single dropdown).

    Args:
        provider: The selected LLM provider
        available_models: List of available model names
        key_prefix: Prefix for widget keys

    Returns:
        Selected model name
    """
    if not available_models:
        return st.text_input(
            "Model",
            value=st.session_state.get("model_name", ""),
            key=f"{key_prefix}_model",
            placeholder="Enter model name"
        )

    saved_model = st.session_state.get("model_name", available_models[0])
    default_idx = available_models.index(saved_model) if saved_model in available_models else 0

    return st.selectbox(
        "Model",
        available_models,
        index=default_idx,
        key=f"{key_prefix}_model"
    )
