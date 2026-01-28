"""
Handai Generate Page
Generate new synthetic datasets with AI
"""

import streamlit as st
import asyncio
import pandas as pd

from tools.generate import GenerateTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state
from ui.components.provider_selector import render_provider_selector
from ui.components.llm_controls import render_llm_controls, render_performance_controls
from ui.components.progress_display import ProgressDisplay
from ui.components.download_buttons import render_download_buttons


def render():
    """Render the generate page"""
    initialize_session_state()
    register_default_tools()

    st.title("Generate Data")
    st.markdown("Create synthetic datasets with AI-powered generation. Describe what you need and let AI build it for you.")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Provider selection
        provider, api_key, base_url = render_provider_selector()

        st.divider()

        # LLM controls (without JSON mode - generation always uses JSON)
        st.header("LLM Controls")
        max_tokens = st.number_input(
            "Max Tokens",
            1, 128000,
            st.session_state.get("max_tokens", 2048),
            256,
            key="gen_max_tokens",
            help="Maximum number of tokens in each AI response"
        )
        st.session_state.max_tokens = max_tokens

        st.divider()

        # Performance controls
        st.header("Performance")
        max_concurrency = st.slider(
            "Max Concurrent Requests",
            1, 50,
            st.session_state.get("max_concurrency", 5),
            key="gen_concurrency",
            help="Higher values = faster but more API load"
        )
        st.session_state.max_concurrency = max_concurrency

        auto_retry = st.checkbox(
            "Auto-retry Failed",
            st.session_state.get("auto_retry", True),
            key="gen_auto_retry",
            help="Automatically retry rows that fail or return empty"
        )
        st.session_state.auto_retry = auto_retry

        if auto_retry:
            max_retries = st.number_input(
                "Max Retries",
                1, 5,
                st.session_state.get("max_retries", 3),
                key="gen_max_retries",
                help="Number of retry attempts per failed row"
            )
            st.session_state.max_retries = max_retries

    # Main content - use the tool
    tool = GenerateTool()

    # Render configuration UI
    config = tool.render_config()

    # Show error if config is invalid
    if not config.is_valid and config.error_message:
        if "Please describe" in config.error_message:
            st.info("Describe the data you want to generate to get started.")
        elif "Please define column" in config.error_message:
            st.info("Next: Define column names for your tabular output.")
        else:
            st.error(config.error_message)

    # Execute buttons
    st.divider()

    num_rows = config.config_data.get("num_rows", 100) if config.config_data else 100

    col1, col2 = st.columns(2)

    with col1:
        run_test = st.button(
            f"Test ({min(10, num_rows)} rows)",
            type="primary",
            use_container_width=True,
            disabled=not config.is_valid
        )
    with col2:
        run_full = st.button(
            "Generate All",
            type="secondary",
            use_container_width=True,
            disabled=not config.is_valid
        )

    if run_test or run_full:
        if not config.is_valid:
            st.error(config.error_message or "Configuration is not valid")
            st.stop()

        # Prepare config for execution
        exec_config = config.config_data.copy()
        exec_config["is_test"] = run_test

        # Create progress display
        progress = ProgressDisplay(
            title="Generation Progress",
            max_log_lines=10
        )

        # Progress callback
        def progress_callback(completed, total, success, errors, retries, log_entry, is_error):
            progress.update(completed, total, success, errors, retries, log_entry, is_error)

        # Run the tool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            tool.execute(exec_config, progress_callback)
        )

        # Render results
        st.divider()
        tool.render_results(result)


if __name__ == "__main__":
    render()
