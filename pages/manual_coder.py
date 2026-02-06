"""
Handai Manual Coder Page
Code qualitative data manually with clickable codes
"""

import streamlit as st
import asyncio

from tools.manual_coder import ManualCoderTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state


def render():
    """Render the manual coder page"""
    initialize_session_state()
    register_default_tools()

    st.title("Manual Coder")
    st.caption("Code qualitative data manually by clicking predefined codes")

    # Main content
    tool = ManualCoderTool()

    config = tool.render_config()

    if not config.is_valid and config.error_message:
        if "Please upload" not in config.error_message and "Please enter" not in config.error_message:
            st.error(config.error_message)

    # Start coding button
    if config.is_valid:
        st.header("4. Code Data")

        if st.button("Start Coding", type="primary", use_container_width=True, key="mc_start_btn"):
            st.session_state["mc_coding_started"] = True

        if st.session_state.get("mc_coding_started"):
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
