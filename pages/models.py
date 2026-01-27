"""
Handai LLM Provider Settings Page
Configure AI providers including OpenAI, Gemini, Ollama, and LM Studio
"""

import streamlit as st
import json
import time
import asyncio
from datetime import datetime

from core.providers import LLMProvider, PROVIDER_CONFIGS, is_local_provider
from core.llm_client import (
    get_client, fetch_local_models, fetch_openrouter_models, call_llm_simple
)
from database import get_db, ConfiguredProvider
from ui.state import initialize_session_state


# Known provider capability tags
CAPABILITY_OPTIONS = ["Streaming", "Vision", "Function Calling", "JSON Mode"]

# Provider type labels
PROVIDER_TYPE_LABELS = {
    "OpenAI": "cloud",
    "Anthropic (Claude)": "cloud",
    "Google (Gemini)": "cloud",
    "Groq": "cloud",
    "Together AI": "cloud",
    "Azure OpenAI": "cloud",
    "OpenRouter": "cloud",
    "LM Studio (Local)": "local",
    "Ollama (Local)": "local",
    "Custom Endpoint": "custom",
}


def _get_provider_icon(provider_type: str) -> str:
    icons = {
        "OpenAI": ":material/smart_toy:",
        "Anthropic (Claude)": ":material/psychology:",
        "Google (Gemini)": ":material/diamond:",
        "Groq": ":material/bolt:",
        "Together AI": ":material/group:",
        "Azure OpenAI": ":material/cloud:",
        "OpenRouter": ":material/router:",
        "LM Studio (Local)": ":material/computer:",
        "Ollama (Local)": ":material/terminal:",
        "Custom Endpoint": ":material/settings:",
    }
    return icons.get(provider_type, ":material/smart_toy:")


@st.dialog("Add Provider")
def _add_provider_dialog():
    db = get_db()
    provider_types = [p.value for p in LLMProvider]
    provider_types.append("Custom")

    selected_type = st.selectbox("Provider Type", provider_types, key="add_prov_type")

    # Pre-fill from known configs
    config = None
    if selected_type != "Custom":
        try:
            enum_val = LLMProvider(selected_type)
            config = PROVIDER_CONFIGS[enum_val]
        except (ValueError, KeyError):
            pass

    col1, col2 = st.columns(2)
    with col1:
        display_name = st.text_input(
            "Display Name",
            value=config.name if config else selected_type,
            key="add_prov_name"
        )
    with col2:
        default_model = st.text_input(
            "Default Model",
            value=config.default_model if config else "",
            key="add_prov_model"
        )

    st.subheader("Connection")
    base_url = st.text_input(
        "Base URL",
        value=config.base_url or "" if config else "",
        key="add_prov_url"
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        key="add_prov_key",
        placeholder="Enter API key..."
    )

    st.subheader("Defaults")
    dc1, dc2 = st.columns(2)
    with dc1:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="add_prov_temp")
    with dc2:
        max_tokens = st.number_input("Max Tokens", 1, 128000, 2048, 256, key="add_prov_tokens")

    # Capabilities
    capabilities = []
    if config:
        # Auto-detect some capabilities
        if not is_local_provider(LLMProvider(selected_type)) if selected_type in [p.value for p in LLMProvider] else False:
            capabilities = ["Streaming", "JSON Mode"]

    st.divider()
    col_cancel, col_add = st.columns(2)
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col_add:
        if st.button("Add Provider", type="primary", use_container_width=True):
            if not display_name or not default_model:
                st.error("Display name and default model are required.")
                return
            provider = ConfiguredProvider.create(
                provider_type=selected_type,
                display_name=display_name,
                default_model=default_model,
                base_url=base_url or None,
                api_key=api_key or None,
                temperature=temperature,
                max_tokens=max_tokens,
                is_enabled=bool(api_key) or (config and not config.requires_api_key),
                capabilities=capabilities,
            )
            db.save_configured_provider(provider)
            st.toast(f"Added {display_name}")
            st.rerun()


@st.dialog("Edit Provider")
def _edit_provider_dialog(provider_id: str):
    db = get_db()
    prov = db.get_configured_provider(provider_id)
    if not prov:
        st.error("Provider not found")
        return

    col1, col2 = st.columns(2)
    with col1:
        display_name = st.text_input("Display Name", value=prov.display_name, key="edit_prov_name")
    with col2:
        default_model = st.text_input("Default Model", value=prov.default_model, key="edit_prov_model")

    st.subheader("Connection")
    base_url = st.text_input("Base URL", value=prov.base_url or "", key="edit_prov_url")
    api_key = st.text_input(
        "API Key", type="password", key="edit_prov_key",
        placeholder="Leave empty to keep existing"
    )

    st.subheader("Defaults")
    dc1, dc2 = st.columns(2)
    with dc1:
        temperature = st.slider("Temperature", 0.0, 2.0, prov.temperature, 0.1, key="edit_prov_temp")
    with dc2:
        max_tokens = st.number_input("Max Tokens", 1, 128000, prov.max_tokens, 256, key="edit_prov_tokens")

    st.divider()
    col_cancel, col_update = st.columns(2)
    with col_cancel:
        if st.button("Cancel", use_container_width=True, key="edit_cancel"):
            st.rerun()
    with col_update:
        if st.button("Update Provider", type="primary", use_container_width=True, key="edit_save"):
            updates = {
                "display_name": display_name,
                "default_model": default_model,
                "base_url": base_url or None,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if api_key:
                updates["api_key"] = api_key
            db.update_configured_provider(provider_id, **updates)
            st.toast(f"Updated {display_name}")
            st.rerun()


def _seed_defaults():
    """Populate all known providers with default configs"""
    db = get_db()
    existing = {p.provider_type for p in db.get_all_configured_providers()}
    count = 0
    for llm_prov in LLMProvider:
        if llm_prov.value in existing:
            continue
        config = PROVIDER_CONFIGS[llm_prov]
        caps = []
        if not is_local_provider(llm_prov):
            caps = ["Streaming", "JSON Mode"]
        provider = ConfiguredProvider.create(
            provider_type=llm_prov.value,
            display_name=config.name,
            default_model=config.default_model,
            base_url=config.base_url,
            temperature=0.7,
            max_tokens=2048,
            is_enabled=False,
            capabilities=caps,
        )
        db.save_configured_provider(provider)
        count += 1
    return count


def _seed_models(provider: ConfiguredProvider):
    """Fetch available models from provider API"""
    models = []
    ptype = provider.provider_type
    base = provider.base_url

    if ptype in ["LM Studio (Local)", "Ollama (Local)", "Custom Endpoint"]:
        if base:
            models = fetch_local_models(base)
    elif ptype == "OpenRouter":
        models = fetch_openrouter_models()
    else:
        # For cloud providers with known configs, use static list
        try:
            enum_val = LLMProvider(ptype)
            config = PROVIDER_CONFIGS[enum_val]
            models = config.models
        except (ValueError, KeyError):
            pass

    return models


def _test_provider(provider: ConfiguredProvider):
    """Send a test prompt to validate provider connectivity"""
    try:
        enum_val = LLMProvider(provider.provider_type)
    except ValueError:
        enum_val = LLMProvider.CUSTOM

    client = get_client(
        enum_val,
        provider.api_key or "dummy",
        provider.base_url
    )

    async def _run_test():
        start = time.time()
        output, error = await call_llm_simple(
            client,
            "You are a helpful assistant.",
            "Say 'Hello from Handai!' in exactly those words.",
            provider.default_model,
            temperature=0.0,
            max_tokens=50,
        )
        latency = round(time.time() - start, 2)
        return output, error, latency

    return asyncio.run(_run_test())


def _render_provider_card(provider: ConfiguredProvider):
    """Render a single provider card"""
    db = get_db()
    type_label = PROVIDER_TYPE_LABELS.get(provider.provider_type, "custom")

    with st.container(border=True):
        # Header row
        header_cols = st.columns([0.5, 2, 1.5, 1, 1, 0.8, 0.8])

        with header_cols[0]:
            st.markdown(_get_provider_icon(provider.provider_type))

        with header_cols[1]:
            st.markdown(f"**{provider.display_name}**")

        with header_cols[2]:
            if provider.is_enabled:
                st.markdown(":green[Enabled]")
            else:
                st.markdown(":gray[Disabled]")
            st.caption(f"{provider.default_model} · {type_label}")

        with header_cols[3]:
            st.caption(f"{provider.total_requests} req")

        with header_cols[4]:
            # Test button
            if st.button("Test", key=f"test_{provider.id}", help="Send test prompt"):
                with st.spinner("Testing..."):
                    output, error, latency = _test_provider(provider)
                    status = "pass" if output and not error else "fail"
                    db.update_configured_provider(
                        provider.id,
                        last_tested=datetime.now().isoformat(),
                        last_test_status=status,
                        last_test_latency=latency,
                    )
                    if status == "pass":
                        st.toast(f"Test passed ({latency}s)")
                    else:
                        st.toast(f"Test failed: {error}", icon=":material/error:")
                    st.rerun()

        with header_cols[5]:
            # Toggle
            new_enabled = st.toggle(
                "On",
                value=provider.is_enabled,
                key=f"toggle_{provider.id}",
                label_visibility="collapsed",
            )
            if new_enabled != provider.is_enabled:
                db.update_configured_provider(provider.id, is_enabled=new_enabled)
                st.rerun()

        with header_cols[6]:
            expanded_key = f"expand_{provider.id}"
            if st.button("Details", key=f"details_btn_{provider.id}",
                         help="Show details"):
                st.session_state[expanded_key] = not st.session_state.get(expanded_key, False)
                st.rerun()

        # Last test info
        if provider.last_tested:
            status_icon = ":green[pass]" if provider.last_test_status == "pass" else ":red[fail]"
            st.caption(f"Last test: {status_icon} · {provider.last_test_latency}s")

        # Expanded body
        if st.session_state.get(f"expand_{provider.id}", False):
            st.divider()
            col_conn, col_gen, col_caps = st.columns(3)

            with col_conn:
                st.markdown("**Connection**")
                st.text(f"Base URL: {provider.base_url or 'Default'}")
                key_display = "****" + provider.api_key[-4:] if provider.api_key and len(provider.api_key) > 4 else ("Set" if provider.api_key else "Not set")
                st.text(f"API Key: {key_display}")
                st.text(f"Timeout: {provider.request_timeout}ms")
                st.text(f"Max Retries: {provider.max_retries}")

            with col_gen:
                st.markdown("**Generation Defaults**")
                st.text(f"Temperature: {provider.temperature}")
                st.text(f"Max Tokens: {provider.max_tokens}")
                st.text(f"Top P: {provider.top_p}")
                st.text(f"Freq. Penalty: {provider.frequency_penalty}")

            with col_caps:
                st.markdown("**Capabilities**")
                caps = provider.get_capabilities()
                if caps:
                    for cap in caps:
                        st.markdown(f"`{cap}`")
                else:
                    st.caption("None configured")

            # Footer buttons
            st.divider()
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("Edit", key=f"edit_{provider.id}", use_container_width=True):
                    _edit_provider_dialog(provider.id)
            with btn_cols[1]:
                if st.button("Seed Models", key=f"seed_{provider.id}", use_container_width=True):
                    models = _seed_models(provider)
                    if models:
                        st.session_state[f"seeded_models_{provider.id}"] = models
                        st.toast(f"Found {len(models)} models")
                        st.rerun()
                    else:
                        st.toast("No models found", icon=":material/warning:")
            with btn_cols[2]:
                if st.button("Delete", key=f"del_{provider.id}", type="secondary",
                             use_container_width=True):
                    st.session_state[f"confirm_delete_{provider.id}"] = True
                    st.rerun()

            # Delete confirmation
            if st.session_state.get(f"confirm_delete_{provider.id}", False):
                st.warning(f"Delete **{provider.display_name}**? This cannot be undone.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("Cancel", key=f"cancel_del_{provider.id}", use_container_width=True):
                        st.session_state[f"confirm_delete_{provider.id}"] = False
                        st.rerun()
                with cc2:
                    if st.button("Confirm Delete", key=f"confirm_del_{provider.id}",
                                 type="primary", use_container_width=True):
                        db.delete_configured_provider(provider.id)
                        st.session_state[f"confirm_delete_{provider.id}"] = False
                        st.toast(f"Deleted {provider.display_name}")
                        st.rerun()

            # Show seeded models
            seeded = st.session_state.get(f"seeded_models_{provider.id}")
            if seeded:
                with st.expander(f"Available Models ({len(seeded)})"):
                    for m in seeded:
                        st.text(m)


def render():
    """Render the LLM Provider Settings page"""
    initialize_session_state()
    db = get_db()

    st.title("LLM Provider Settings")
    st.caption("Configure AI providers including OpenAI, Gemini, Ollama, and LM Studio")

    # Top action bar
    col_seed, col_add, col_spacer = st.columns([1, 1, 3])
    with col_seed:
        if st.button("Seed Defaults", use_container_width=True,
                      help="Populate all known providers with default configs"):
            count = _seed_defaults()
            if count > 0:
                st.toast(f"Added {count} provider(s)")
            else:
                st.toast("All providers already configured")
            st.rerun()
    with col_add:
        if st.button("+ Add Provider", use_container_width=True):
            _add_provider_dialog()

    st.divider()

    # Render provider cards
    providers = db.get_all_configured_providers()

    if not providers:
        st.info("No providers configured yet. Click **Seed Defaults** to add all known providers, "
                "or **+ Add Provider** to add one manually.")
        return

    # Show enabled first, then disabled
    enabled = [p for p in providers if p.is_enabled]
    disabled = [p for p in providers if not p.is_enabled]

    if enabled:
        st.subheader(f"Enabled ({len(enabled)})")
        for p in enabled:
            _render_provider_card(p)

    if disabled:
        st.subheader(f"Disabled ({len(disabled)})")
        for p in disabled:
            _render_provider_card(p)


if __name__ == "__main__":
    render()
