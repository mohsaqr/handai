"""
Handai AI Coder Page
AI-assisted manual coding with inter-rater reliability
"""

import streamlit as st
import asyncio

from tools.ai_coder import AICoderTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state


def render():
    """Render the AI coder page"""
    initialize_session_state()
    register_default_tools()

    st.title("AI Coder")
    st.caption("AI-assisted manual coding with inter-rater reliability analytics")

    # Main content
    tool = AICoderTool()

    config = tool.render_config()

    if not config.is_valid and config.error_message:
        if "Please upload" not in config.error_message and "Please enter" not in config.error_message:
            st.error(config.error_message)

    # Start coding button
    if config.is_valid:
        st.header("5. Code Data")

        if st.button("Start Coding", type="primary", use_container_width=True, key="aic_start_btn"):
            st.session_state["aic_coding_started"] = True

        if st.session_state.get("aic_coding_started"):
            st.divider()

            # Execute to get result object (just passes through data)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(tool.execute(config.config_data, None))
            finally:
                loop.close()

            tool.render_results(result)


if __name__ == "__main__":
    render()
