"""
Handai Home Page
Landing page with tool cards and navigation
"""

import streamlit as st
from tools.registry import ToolRegistry, register_default_tools
from ui.state import initialize_session_state, get_providers_with_api_keys
from database import get_db


def render():
    """Render the home page"""
    # Initialize
    initialize_session_state()
    register_default_tools()
    db = get_db()

    # Header
    st.title("Handai")
    st.markdown("""
    **AI-Powered Data Transformation & Generation**

    Upload datasets for AI-powered transformation, or generate entirely new synthetic datasets.
    """)

    # Quick stats
    stats = db.get_global_stats()
    if stats.get("total_runs", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sessions", stats.get("total_sessions", 0))
        with col2:
            st.metric("Total Runs", stats.get("total_runs", 0))
        with col3:
            st.metric("Rows Processed", stats.get("total_rows_processed", 0))
        with col4:
            total = (stats.get("total_success", 0) or 0) + (stats.get("total_errors", 0) or 0)
            success_rate = (stats.get("total_success", 0) or 0) / total * 100 if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        st.divider()

    # Tool cards
    st.header("Available Tools")
    st.caption("Select a tool from the sidebar to get started")

    tools = ToolRegistry.list_tools()

    # Create columns for tool cards
    cols = st.columns(len(tools))

    for col, tool in zip(cols, tools):
        with col:
            with st.container(border=True):
                st.subheader(f"{tool.icon} {tool.name}")
                st.write(tool.description)

    st.divider()

    # Quick setup check
    st.header("Setup Status")

    providers_with_keys = get_providers_with_api_keys()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Keys")
        if providers_with_keys:
            for provider in providers_with_keys:
                st.success(f"{provider}: Configured")
        else:
            st.warning("No API keys configured")
            st.caption("Go to **Settings** in the sidebar to add your API keys")

    with col2:
        st.subheader("Navigation")
        st.info("Use the **sidebar** to navigate between pages")

    # Recent sessions
    sessions = db.get_all_sessions(limit=5)
    if sessions:
        st.divider()
        st.header("Recent Sessions")

        for session in sessions:
            runs = db.get_session_runs(session.session_id)
            run_count = len(runs)

            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{session.name}** ({session.mode})")
                st.caption(f"Created: {session.created_at[:16]} | Runs: {run_count}")
            with col2:
                if run_count > 0:
                    last_run = runs[0]
                    status_icon = {
                        "completed": "✓",
                        "running": "⏳",
                        "failed": "✗",
                        "cancelled": "⊘"
                    }.get(last_run.status, "")
                    st.caption(f"Last: {status_icon} {last_run.status}")

    # Footer
    st.divider()
    st.caption("Handai v4.0 - Multi-Provider AI Data Transformer & Generator")


if __name__ == "__main__":
    render()
