"""
Handai Qualitative Coder Page
Code qualitative data with AI
"""

import streamlit as st
import asyncio

from tools.qualitative import QualitativeTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state
from ui.components.provider_selector import render_provider_selector
from ui.components.progress_display import ProgressDisplay


def render():
    """Render the qualitative coder page"""
    initialize_session_state()
    register_default_tools()

    st.title("Qualitative Coder")
    st.caption("Code qualitative data like interviews, observations, and surveys with AI")

    # Sidebar settings
    with st.sidebar:
        st.header(":material/smart_toy: AI Provider")

        provider, api_key, base_url = render_provider_selector(key_prefix="qualitative")

        st.divider()
        st.caption(":material/settings: [Settings](/settings) for Temperature, Max Tokens, and Performance")

    # Main content
    tool = QualitativeTool()

    config = tool.render_config()

    if not config.is_valid and config.error_message:
        if "Please upload" not in config.error_message and "Please enter" not in config.error_message:
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
                disabled=not config.is_valid,
                key="qualitative_test_btn"
            )
        with col2:
            run_full = st.button(
                "Full Run",
                type="secondary",
                use_container_width=True,
                disabled=not config.is_valid,
                key="qualitative_full_btn"
            )

        if run_test or run_full:
            if not config.is_valid:
                st.error(config.error_message or "Configuration is not valid")
                st.stop()

            exec_config = config.config_data.copy()
            exec_config["is_test"] = run_test

            progress = ProgressDisplay(title="Processing Status", max_log_lines=10)

            def progress_callback(completed, total, success, errors, retries, log_entry, is_error):
                progress.update(completed, total, success, errors, retries, log_entry, is_error)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    tool.execute(exec_config, progress_callback)
                )
            except Exception as e:
                st.error(f"Qualitative coding failed: {str(e)}")
                return
            finally:
                loop.close()

            st.divider()
            tool.render_results(result)


if __name__ == "__main__":
    render()
