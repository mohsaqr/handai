"""
Handai Transform Page
Transform existing datasets with AI
"""

import streamlit as st
import asyncio
import pandas as pd

from tools.transform import TransformTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state
from ui.components.provider_selector import render_provider_selector
from ui.components.model_selector import render_model_selector
from ui.components.llm_controls import render_llm_controls, render_performance_controls
from ui.components.progress_display import ProgressDisplay, render_completion_summary
from ui.components.result_inspector import render_result_inspector
from ui.components.download_buttons import render_download_buttons


def render():
    """Render the transform page"""
    initialize_session_state()
    register_default_tools()

    st.title("Transform Data")
    st.caption("Upload a dataset and transform each row using AI")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Provider selection
        provider, api_key, base_url = render_provider_selector()

        st.divider()

        # Model selection
        model = render_model_selector(provider, base_url)

        st.divider()

        # LLM controls
        temperature, max_tokens, json_mode = render_llm_controls(provider, show_header=True)

        st.divider()

        # Performance controls
        max_concurrency, test_batch_size, auto_retry, max_retries, realtime_progress, save_path = render_performance_controls()

    # Main content - use the tool
    tool = TransformTool()

    # Render configuration UI
    config = tool.render_config()

    # Show error if config is invalid
    if not config.is_valid and config.error_message:
        if "Please upload" in config.error_message or "Please enter AI instructions" in config.error_message:
            # These are expected states, not errors
            pass
        else:
            st.error(config.error_message)

    # Execute buttons
    if config.config_data and config.config_data.get("df") is not None:
        st.header("3. Execute")

        col1, col2 = st.columns(2)
        test_batch_size = st.session_state.get("test_batch_size", 10)

        with col1:
            run_test = st.button(
                f"Test ({test_batch_size} rows)",
                type="primary",
                use_container_width=True,
                disabled=not config.is_valid
            )
        with col2:
            run_full = st.button(
                "Full Run",
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
                title="Processing Status",
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
