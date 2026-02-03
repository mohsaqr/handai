"""
Handai Settings Page
Performance, storage, and general preferences
"""

import streamlit as st
import json

from ui.state import (
    initialize_session_state, save_setting,
)
from database import get_db
from config import (
    DEFAULT_MAX_CONCURRENCY, DEFAULT_TEST_BATCH_SIZE, DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
)


def render():
    """Render the settings page"""
    initialize_session_state()
    db = get_db()

    st.title("Settings")
    st.caption("Configure model defaults, performance, and storage options.")

    # Tabs for settings categories
    tab1, tab2, tab3 = st.tabs([
        ":material/tune: Model Defaults",
        ":material/speed: Performance",
        ":material/folder: Storage"
    ])

    # ==========================================
    # TAB: Model Defaults
    # ==========================================
    with tab1:
        st.header("Model Defaults")
        st.caption("Default settings used when running AI tools")

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                st.session_state.get("temperature", DEFAULT_TEMPERATURE),
                0.1,
                key="settings_temperature",
                help="0 = deterministic, 2 = creative"
            )
            if temperature != st.session_state.get("temperature"):
                st.session_state.temperature = temperature
                save_setting("temperature")

        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                1, 128000,
                st.session_state.get("max_tokens", DEFAULT_MAX_TOKENS),
                256,
                key="settings_max_tokens",
                help="Maximum response length"
            )
            if max_tokens != st.session_state.get("max_tokens"):
                st.session_state.max_tokens = max_tokens
                save_setting("max_tokens")

        json_mode = st.checkbox(
            "JSON Mode",
            st.session_state.get("json_mode", False),
            key="settings_json_mode",
            help="Force structured JSON output (when supported by provider)"
        )
        if json_mode != st.session_state.get("json_mode"):
            st.session_state.json_mode = json_mode
            save_setting("json_mode")

    # ==========================================
    # TAB: Performance
    # ==========================================
    with tab2:
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
    # TAB: Storage
    # ==========================================
    with tab3:
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
