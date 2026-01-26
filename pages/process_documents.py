"""
Handai Process Documents Page
Extract structured data from documents using AI
"""

import streamlit as st
import asyncio

from tools.process_documents import ProcessDocumentsTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state
from ui.components.provider_selector import render_provider_selector
from ui.components.model_selector import render_model_selector
from ui.components.llm_controls import render_llm_controls, render_performance_controls
from ui.components.progress_display import ProgressDisplay


def render():
    """Render the process documents page"""
    initialize_session_state()
    register_default_tools()

    st.title("Process Documents")
    st.caption("Extract structured data from documents using AI")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Provider selection
        provider, api_key, base_url = render_provider_selector()

        st.divider()

        # Model selection
        model = render_model_selector(provider, base_url)

        st.divider()

        # LLM controls (without JSON mode since we use CSV for document processing)
        temperature, max_tokens, json_mode = render_llm_controls(provider, show_header=True)

        st.divider()

        # Performance controls
        max_concurrency, test_batch_size, auto_retry, max_retries, realtime_progress, save_path = render_performance_controls()

    # Main content - use the tool
    tool = ProcessDocumentsTool()

    # Render configuration UI
    config = tool.render_config()

    # Show error if config is invalid
    if not config.is_valid and config.error_message:
        if "Please select documents" in config.error_message:
            # Guide user to select documents
            st.info("Select a folder path or upload files to get started.")
        elif "Please enter" in config.error_message:
            # Guide user to complete configuration
            st.info(config.error_message.replace("Please enter", "Next: Enter"))
        else:
            st.error(config.error_message)

    # Execute buttons
    has_documents = config.config_data and (
        config.config_data.get("files") or
        config.config_data.get("uploaded_files")
    )

    if has_documents:
        st.header("3. Process Documents")

        col1, col2, col3 = st.columns([1, 1, 2])
        test_batch_size = st.session_state.get("test_batch_size", 5)

        with col1:
            run_test = st.button(
                f"Test ({test_batch_size})",
                type="primary",
                use_container_width=True,
                disabled=not config.is_valid
            )
        with col2:
            run_full = st.button(
                "Process All",
                type="secondary",
                use_container_width=True,
                disabled=not config.is_valid
            )
        with col3:
            output_path = st.text_input(
                "Auto-save to:",
                placeholder="Optional: /path/to/output.csv",
                key="doc_output_path"
            )

        if run_test or run_full:
            if not config.is_valid:
                st.error(config.error_message or "Configuration is not valid")
                st.stop()

            # Prepare config for execution
            exec_config = config.config_data.copy()
            exec_config["is_test"] = run_test
            if output_path:
                exec_config["save_path"] = output_path

            # Create progress display
            progress = ProgressDisplay(
                title="Processing Status",
                max_log_lines=15
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
