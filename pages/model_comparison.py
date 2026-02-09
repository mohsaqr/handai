"""
Handai Model Comparison Page
Compare outputs across multiple LLM models side-by-side
"""

import streamlit as st
import asyncio

from tools.model_comparison import ModelComparisonTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state
from ui.components.progress_display import ProgressDisplay


def render():
    """Render the model comparison page"""
    initialize_session_state()
    register_default_tools()

    st.title("Model Comparison")
    st.markdown("Run the same task across multiple models and compare their outputs side-by-side.")

    # Sidebar
    with st.sidebar:
        st.header(":material/compare: Comparison")
        st.caption("Compare outputs from multiple LLM models on the same data.")

        st.divider()
        st.caption(":material/settings: [Settings](/settings) for Temperature, Max Tokens, and Performance")

    # Main content
    tool = ModelComparisonTool()

    config = tool.render_config()

    if not config.is_valid and config.error_message:
        # Only show errors that aren't initial state messages
        skip_messages = ["Please upload", "Please enter", "No providers", "Select at least"]
        if not any(msg in config.error_message for msg in skip_messages):
            st.error(config.error_message)

    # Execute buttons
    if config.config_data and config.config_data.get("df") is not None and config.is_valid:
        st.header("5. Execute")

        col1, col2 = st.columns(2)
        test_batch_size = st.session_state.get("test_batch_size", 10)

        with col1:
            run_test = st.button(
                f"Test ({test_batch_size} rows)",
                type="primary",
                use_container_width=True,
                key="comparison_test_btn"
            )
        with col2:
            run_full = st.button(
                "Full Run",
                type="secondary",
                use_container_width=True,
                key="comparison_full_btn"
            )

        if run_test or run_full:
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
                st.error(f"Model comparison failed: {str(e)}")
                return
            finally:
                loop.close()

            st.divider()
            tool.render_results(result)


if __name__ == "__main__":
    render()
