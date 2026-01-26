"""
Provider Selector Component
UI for selecting LLM provider and managing API keys
"""

import streamlit as st
from typing import Tuple, Optional
from core.providers import (
    LLMProvider, PROVIDER_CONFIGS, get_provider_names
)
from ui.state import (
    save_setting, get_api_key_for_provider, set_api_key_for_provider
)


def render_provider_selector(key_prefix: str = "main") -> Tuple[LLMProvider, str, Optional[str]]:
    """
    Render provider selection UI with API key input.

    Args:
        key_prefix: Prefix for widget keys to avoid conflicts

    Returns:
        Tuple of (selected_provider, api_key, base_url)
    """
    st.header("AI Provider")

    provider_names = get_provider_names()

    # Get saved provider or default
    saved_provider = st.session_state.get("selected_provider", provider_names[0])
    default_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

    selected_provider_name = st.selectbox(
        "Select Provider",
        provider_names,
        index=default_idx,
        key=f"{key_prefix}_selected_provider",
        on_change=lambda: save_setting("selected_provider")
    )

    # Store in main session state
    st.session_state.selected_provider = selected_provider_name

    selected_provider = LLMProvider(selected_provider_name)
    provider_config = PROVIDER_CONFIGS[selected_provider]

    st.caption(f"{provider_config.description}")

    # API Key management
    api_key = ""
    if provider_config.requires_api_key:
        # Get provider-specific key
        saved_key = get_api_key_for_provider(selected_provider_name)

        api_key_help = None
        if selected_provider == LLMProvider.OPENROUTER:
            api_key_help = "Get your API key at openrouter.ai/keys"

        # Use a unique key for each provider's API key input
        api_key = st.text_input(
            "API Key",
            value=saved_key,
            type="password",
            key=f"{key_prefix}_api_key_{selected_provider_name}",
            help=api_key_help
        )

        # Save button or auto-save
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Save", key=f"{key_prefix}_save_key", use_container_width=True):
                set_api_key_for_provider(selected_provider_name, api_key)
                st.toast(f"API key saved for {selected_provider_name}")

        # Show status
        if saved_key:
            st.success("API key configured", icon=":material/check:")
        else:
            st.warning("No API key set", icon=":material/warning:")

        # Provider-specific hints
        if selected_provider == LLMProvider.OPENROUTER:
            st.caption("[Get your key at openrouter.ai/keys](https://openrouter.ai/keys)")
    else:
        api_key = "dummy"
        st.info("No API key required for local models")

    # Base URL for providers that need it
    base_url = None
    if selected_provider in [LLMProvider.AZURE, LLMProvider.CUSTOM, LLMProvider.LM_STUDIO, LLMProvider.OLLAMA]:
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.get("base_url", provider_config.base_url or ""),
            key=f"{key_prefix}_base_url",
            help="Full URL to the API endpoint",
            on_change=lambda: save_setting("base_url")
        )
        st.session_state.base_url = base_url
    else:
        base_url = provider_config.base_url

    return selected_provider, api_key, base_url


def render_provider_selector_compact(key_prefix: str = "compact") -> Tuple[LLMProvider, str, Optional[str]]:
    """
    Render a compact provider selector (single row).

    Returns:
        Tuple of (selected_provider, api_key, base_url)
    """
    provider_names = get_provider_names()
    saved_provider = st.session_state.get("selected_provider", provider_names[0])
    default_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_provider_name = st.selectbox(
            "Provider",
            provider_names,
            index=default_idx,
            key=f"{key_prefix}_provider",
            label_visibility="collapsed"
        )
        st.session_state.selected_provider = selected_provider_name

    selected_provider = LLMProvider(selected_provider_name)
    provider_config = PROVIDER_CONFIGS[selected_provider]

    with col2:
        if provider_config.requires_api_key:
            saved_key = get_api_key_for_provider(selected_provider_name)
            api_key = st.text_input(
                "API Key",
                value=saved_key,
                type="password",
                key=f"{key_prefix}_api_key",
                label_visibility="collapsed",
                placeholder="Enter API key..."
            )
        else:
            api_key = "dummy"
            st.info("No key needed")

    base_url = provider_config.base_url

    return selected_provider, api_key, base_url
