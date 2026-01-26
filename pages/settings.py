"""
Handai Settings Page
Centralized settings for API keys, providers, and preferences
"""

import streamlit as st
import json
from typing import Dict, Any

from core.providers import LLMProvider, PROVIDER_CONFIGS, get_provider_names
from ui.state import (
    initialize_session_state, save_setting, save_all_current_settings,
    get_api_key_for_provider, set_api_key_for_provider,
    get_default_model_for_provider, set_default_model_for_provider,
    get_providers_with_api_keys
)
from database import get_db
from config import (
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TEST_BATCH_SIZE, DEFAULT_MAX_RETRIES
)


def render():
    """Render the settings page"""
    initialize_session_state()
    db = get_db()

    st.title("Settings")

    # Tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs(["API Keys", "LLM Controls", "Performance", "Storage"])

    # ==========================================
    # TAB 1: API Keys
    # ==========================================
    with tab1:
        st.header("API Keys")
        st.caption("Configure API keys for each provider. Keys are stored securely in the local database.")

        # Get current status
        providers_with_keys = get_providers_with_api_keys()

        # Create a table-like layout
        for provider in LLMProvider:
            config = PROVIDER_CONFIGS[provider]

            if not config.requires_api_key:
                continue

            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 3, 1])

                with col1:
                    st.write(f"**{provider.value}**")
                    st.caption(config.description[:50] + "..." if len(config.description) > 50 else config.description)

                with col2:
                    current_key = get_api_key_for_provider(provider.value)
                    new_key = st.text_input(
                        "API Key",
                        value=current_key,
                        type="password",
                        key=f"api_key_{provider.value}",
                        label_visibility="collapsed",
                        placeholder="Enter API key..."
                    )

                with col3:
                    if current_key:
                        st.success("Saved")
                    else:
                        st.caption("Not set")

                    if new_key != current_key:
                        if st.button("Save", key=f"save_key_{provider.value}"):
                            set_api_key_for_provider(provider.value, new_key)
                            st.toast(f"API key saved for {provider.value}")
                            st.rerun()

        # Local providers note
        st.divider()
        st.subheader("Local Providers")
        st.info("LM Studio, Ollama, and Custom endpoints don't require API keys.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**LM Studio**")
            lm_url = st.text_input(
                "Base URL",
                value=st.session_state.get("lm_studio_url", "http://localhost:1234/v1"),
                key="lm_studio_url_input"
            )
            if lm_url != st.session_state.get("lm_studio_url"):
                db.save_provider_setting("LM Studio (Local)", "base_url", lm_url)
                st.session_state.lm_studio_url = lm_url

        with col2:
            st.write("**Ollama**")
            ollama_url = st.text_input(
                "Base URL",
                value=st.session_state.get("ollama_url", "http://localhost:11434/v1"),
                key="ollama_url_input"
            )
            if ollama_url != st.session_state.get("ollama_url"):
                db.save_provider_setting("Ollama (Local)", "base_url", ollama_url)
                st.session_state.ollama_url = ollama_url

    # ==========================================
    # TAB 2: LLM Controls
    # ==========================================
    with tab2:
        st.header("Default LLM Settings")
        st.caption("These settings are used as defaults for new processing runs.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Default Provider")
            provider_names = get_provider_names()
            saved_provider = st.session_state.get("selected_provider", "OpenAI")
            default_idx = provider_names.index(saved_provider) if saved_provider in provider_names else 0

            selected_provider = st.selectbox(
                "Provider",
                provider_names,
                index=default_idx,
                key="settings_default_provider"
            )

            if selected_provider != st.session_state.get("selected_provider"):
                st.session_state.selected_provider = selected_provider
                save_setting("selected_provider")

            # Default model for selected provider
            provider_enum = LLMProvider(selected_provider)
            config = PROVIDER_CONFIGS[provider_enum]
            available_models = config.models

            if available_models:
                saved_model = st.session_state.get("model_name", config.default_model)
                model_idx = available_models.index(saved_model) if saved_model in available_models else 0

                selected_model = st.selectbox(
                    "Default Model",
                    available_models,
                    index=model_idx,
                    key="settings_default_model"
                )

                if selected_model != st.session_state.get("model_name"):
                    st.session_state.model_name = selected_model
                    save_setting("model_name")

        with col2:
            st.subheader("Generation Parameters")

            temperature = st.slider(
                "Default Temperature",
                0.0, 2.0,
                st.session_state.get("temperature", DEFAULT_TEMPERATURE),
                0.1,
                key="settings_temperature",
                help="0 = deterministic, 2 = very creative"
            )
            if temperature != st.session_state.get("temperature"):
                st.session_state.temperature = temperature
                save_setting("temperature")

            max_tokens = st.number_input(
                "Default Max Tokens",
                1, 128000,
                st.session_state.get("max_tokens", DEFAULT_MAX_TOKENS),
                256,
                key="settings_max_tokens"
            )
            if max_tokens != st.session_state.get("max_tokens"):
                st.session_state.max_tokens = max_tokens
                save_setting("max_tokens")

            json_mode = st.checkbox(
                "JSON Mode by Default",
                st.session_state.get("json_mode", False),
                key="settings_json_mode",
                help="Force structured JSON output"
            )
            if json_mode != st.session_state.get("json_mode"):
                st.session_state.json_mode = json_mode
                save_setting("json_mode")

    # ==========================================
    # TAB 3: Performance
    # ==========================================
    with tab3:
        st.header("Performance Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Concurrency")

            max_concurrency = st.slider(
                "Max Concurrent Requests",
                1, 50,
                st.session_state.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
                key="settings_concurrency",
                help="Higher values = faster but more API load"
            )
            if max_concurrency != st.session_state.get("max_concurrency"):
                st.session_state.max_concurrency = max_concurrency
                save_setting("max_concurrency")

            test_batch_size = st.number_input(
                "Test Batch Size",
                1, 1000,
                st.session_state.get("test_batch_size", DEFAULT_TEST_BATCH_SIZE),
                key="settings_test_batch"
            )
            if test_batch_size != st.session_state.get("test_batch_size"):
                st.session_state.test_batch_size = test_batch_size
                save_setting("test_batch_size")

        with col2:
            st.subheader("Retry Settings")

            auto_retry = st.checkbox(
                "Auto-retry Failed Requests",
                st.session_state.get("auto_retry", True),
                key="settings_auto_retry",
                help="Automatically retry on errors"
            )
            if auto_retry != st.session_state.get("auto_retry"):
                st.session_state.auto_retry = auto_retry
                save_setting("auto_retry")

            if auto_retry:
                max_retries = st.number_input(
                    "Max Retries",
                    1, 5,
                    st.session_state.get("max_retries", DEFAULT_MAX_RETRIES),
                    key="settings_max_retries"
                )
                if max_retries != st.session_state.get("max_retries"):
                    st.session_state.max_retries = max_retries
                    save_setting("max_retries")

            realtime_progress = st.checkbox(
                "Real-Time Progress Updates",
                st.session_state.get("realtime_progress", True),
                key="settings_realtime",
                help="Update UI every row (slightly slower)"
            )
            if realtime_progress != st.session_state.get("realtime_progress"):
                st.session_state.realtime_progress = realtime_progress
                save_setting("realtime_progress")

    # ==========================================
    # TAB 4: Storage
    # ==========================================
    with tab4:
        st.header("Storage Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Auto-Save")

            save_path = st.text_input(
                "Auto-Save Path",
                st.session_state.get("save_path", ""),
                key="settings_save_path",
                placeholder="/path/to/save/results"
            )
            if save_path != st.session_state.get("save_path"):
                st.session_state.save_path = save_path
                save_setting("save_path")

            st.caption("Results will be auto-saved to this location during processing")

        with col2:
            st.subheader("Export/Import Settings")

            if st.button("Export All Settings", use_container_width=True):
                settings = db.get_all_settings()
                # Remove sensitive data
                settings_export = {k: v for k, v in settings.items() if "api_key" not in k.lower()}
                st.download_button(
                    "Download Settings JSON",
                    json.dumps(settings_export, indent=2),
                    "handai_settings.json",
                    "application/json"
                )

            uploaded_settings = st.file_uploader(
                "Import Settings",
                type=["json"],
                key="import_settings"
            )
            if uploaded_settings:
                try:
                    imported = json.load(uploaded_settings)
                    if st.button("Apply Imported Settings"):
                        db.save_all_settings(imported)
                        st.success("Settings imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error importing settings: {e}")

        st.divider()

        st.subheader("Database Info")
        stats = db.get_global_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sessions", stats.get("total_sessions", 0))
        with col2:
            st.metric("Total Runs", stats.get("total_runs", 0))
        with col3:
            st.metric("Rows Processed", stats.get("total_rows_processed", 0))

        if st.button("Clear All History", type="secondary"):
            if st.button("Confirm Clear All Data", type="primary"):
                # This would need a confirmation dialog in practice
                st.warning("This action cannot be undone!")


if __name__ == "__main__":
    render()
