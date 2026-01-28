"""
Handai Home Page
Landing page with tool cards and navigation
"""

import streamlit as st
from tools.registry import register_default_tools
from ui.state import initialize_session_state
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
            st.metric("Sessions", stats.get("total_sessions", 0), help="Total saved sessions")
        with col2:
            st.metric("Total Runs", stats.get("total_runs", 0), help="Total tool executions across all sessions")
        with col3:
            st.metric("Rows Processed", stats.get("total_rows_processed", 0), help="Total data rows processed by AI")
        with col4:
            total = (stats.get("total_success", 0) or 0) + (stats.get("total_errors", 0) or 0)
            success_rate = (stats.get("total_success", 0) or 0) / total * 100 if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%", help="Percentage of rows processed without errors")

        st.divider()

    # Tool cards
    st.header("Available Tools")

    tool_cards = [
        {
            "url": "/transform",
            "icon": ":material/transform:",
            "title": "Transform Data",
            "description": "Upload a CSV and use AI to transform, enrich, or classify each row",
        },
        {
            "url": "/generate",
            "icon": ":material/auto_awesome:",
            "title": "Generate Data",
            "description": "Describe what you need and let AI generate synthetic rows from scratch",
        },
        {
            "url": "/process-documents",
            "icon": ":material/description:",
            "title": "Process Documents",
            "description": "Extract structured data from PDFs, text files, and other documents",
        },
    ]

    cols = st.columns(len(tool_cards))
    for col, card in zip(cols, tool_cards):
        with col:
            with st.container(border=True):
                st.page_link(card["url"], label=card["title"], icon=card["icon"])
                st.caption(card["description"])

    st.divider()

    # System page links
    system_cards = [
        {
            "url": "/llm-providers",
            "icon": ":material/smart_toy:",
            "title": "LLM Providers",
            "description": "Configure AI providers, API keys, and default models",
        },
        {
            "url": "/history",
            "icon": ":material/history:",
            "title": "History",
            "description": "View past sessions and runs",
        },
        {
            "url": "/settings",
            "icon": ":material/settings:",
            "title": "Settings",
            "description": "App preferences and configuration",
        },
    ]

    sys_cols = st.columns(len(system_cards))
    for col, card in zip(sys_cols, system_cards):
        with col:
            with st.container(border=True):
                st.page_link(card["url"], label=card["title"], icon=card["icon"])
                st.caption(card["description"])

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
