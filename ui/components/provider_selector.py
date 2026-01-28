"""
Provider Selector Component
UI for selecting LLM provider — reads from configured_providers table
"""

import streamlit as st
from typing import Tuple, Optional
from core.providers import (
    LLMProvider, PROVIDER_CONFIGS, get_provider_names, is_local_provider
)
from ui.state import (
    save_setting, get_api_key_for_provider, set_api_key_for_provider
)
from database import get_db


def _get_enabled_provider_options():
    """Return list of (display_name, provider_type, api_key, base_url, default_model) for enabled providers."""
    db = get_db()
    providers = db.get_enabled_configured_providers()
    if providers:
        return [
            (p.display_name, p.provider_type, p.api_key, p.base_url, p.default_model)
            for p in providers
        ]
    # Fallback: if no configured providers, use legacy provider list
    return None


def render_provider_selector(key_prefix: str = "main") -> Tuple[LLMProvider, str, Optional[str]]:
    """
    Render provider selection UI.
    Prefers configured_providers table; falls back to legacy enum list.

    Returns:
        Tuple of (selected_provider, api_key, base_url)
    """
    st.header("AI Provider")

    enabled = _get_enabled_provider_options()

    if enabled:
        names = [e[0] for e in enabled]
        if not names:
            st.warning("No providers configured. Please configure providers in the LLM Providers page.")
            return LLMProvider.OPENAI, "", None

        saved = st.session_state.get("selected_provider", names[0])
        default_idx = names.index(saved) if saved in names else 0

        selected_name = st.selectbox(
            "Select Provider", names, index=default_idx,
            key=f"{key_prefix}_selected_provider",
        )
        st.session_state.selected_provider = selected_name
        entry = enabled[names.index(selected_name)]
        _, ptype, api_key, base_url, default_model = entry

        try:
            provider_enum = LLMProvider(ptype)
        except ValueError:
            provider_enum = LLMProvider.CUSTOM

        st.session_state.model_name = default_model

        # Show active model as read-only caption
        st.caption(f"Model: **{default_model}** · [Configure in LLM Providers](/llm-providers)")

        # Check if this provider requires an API key
        requires_key = not is_local_provider(provider_enum)
        has_key = api_key and api_key.strip()

        if requires_key and not has_key:
            st.warning("No API key configured for this provider. Set it in **LLM Providers** page.")
        else:
            st.success("Provider configured", icon=":material/check:")

        # Use dummy key for local providers that don't need one
        if not has_key:
            api_key = "dummy"

        return provider_enum, api_key, base_url

    # Fallback to legacy behaviour
    provider_names = get_provider_names()
    saved_provider = st.session_state.get("selected_provider", provider_names[0])
    default_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

    selected_provider_name = st.selectbox(
        "Select Provider", provider_names, index=default_idx,
        key=f"{key_prefix}_selected_provider",
        on_change=lambda: save_setting("selected_provider")
    )
    st.session_state.selected_provider = selected_provider_name
    selected_provider = LLMProvider(selected_provider_name)
    provider_config = PROVIDER_CONFIGS[selected_provider]
    st.caption(f"{provider_config.description}")

    api_key = ""
    if provider_config.requires_api_key:
        saved_key = get_api_key_for_provider(selected_provider_name)
        api_key = st.text_input(
            "API Key", value=saved_key, type="password",
            key=f"{key_prefix}_api_key_{selected_provider_name}",
        )
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Save", key=f"{key_prefix}_save_key", use_container_width=True):
                set_api_key_for_provider(selected_provider_name, api_key)
                st.toast(f"API key saved for {selected_provider_name}")
        if saved_key:
            st.success("API key configured", icon=":material/check:")
        else:
            st.warning("No API key set", icon=":material/warning:")
    else:
        api_key = "dummy"
        st.info("No API key required for local models")

    base_url = None
    if selected_provider in [LLMProvider.AZURE, LLMProvider.CUSTOM, LLMProvider.LM_STUDIO, LLMProvider.OLLAMA]:
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.get("base_url", provider_config.base_url or ""),
            key=f"{key_prefix}_base_url",
            on_change=lambda: save_setting("base_url")
        )
        st.session_state.base_url = base_url
    else:
        base_url = provider_config.base_url

    return selected_provider, api_key, base_url


def render_provider_selector_compact(key_prefix: str = "compact") -> Tuple[LLMProvider, str, Optional[str]]:
    """
    Render a compact provider selector (single row).
    Prefers configured_providers; falls back to legacy.

    Returns:
        Tuple of (selected_provider, api_key, base_url)
    """
    enabled = _get_enabled_provider_options()

    if enabled:
        names = [e[0] for e in enabled]
        if not names:
            return LLMProvider.OPENAI, "dummy", None

        saved = st.session_state.get("selected_provider", names[0])
        default_idx = names.index(saved) if saved in names else 0

        selected_name = st.selectbox(
            "Provider", names, index=default_idx,
            key=f"{key_prefix}_provider", label_visibility="collapsed"
        )
        st.session_state.selected_provider = selected_name
        entry = enabled[names.index(selected_name)]
        _, ptype, api_key, base_url, default_model = entry

        try:
            provider_enum = LLMProvider(ptype)
        except ValueError:
            provider_enum = LLMProvider.CUSTOM

        st.session_state.model_name = default_model
        return provider_enum, api_key or "dummy", base_url

    # Fallback
    provider_names = get_provider_names()
    saved_provider = st.session_state.get("selected_provider", provider_names[0])
    default_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_provider_name = st.selectbox(
            "Provider", provider_names, index=default_idx,
            key=f"{key_prefix}_provider", label_visibility="collapsed"
        )
        st.session_state.selected_provider = selected_provider_name

    selected_provider = LLMProvider(selected_provider_name)
    provider_config = PROVIDER_CONFIGS[selected_provider]

    with col2:
        if provider_config.requires_api_key:
            saved_key = get_api_key_for_provider(selected_provider_name)
            api_key = st.text_input(
                "API Key", value=saved_key, type="password",
                key=f"{key_prefix}_api_key", label_visibility="collapsed",
                placeholder="Enter API key..."
            )
        else:
            api_key = "dummy"
            st.info("No key needed")

    return selected_provider, api_key, provider_config.base_url
