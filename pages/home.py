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

    nav_pages = st.session_state.get("_pages", {})

    tool_cards = [
        ("transform", ":material/transform:", "Transform Data",
         "Upload a CSV and use AI to transform, enrich, or classify each row.",
         "- Enrich rows with extracted entities, topics, or classifications\n"
         "- Add sentiment scores, summaries, or translations to every row\n"
         "- Clean, normalize, or reformat messy data at scale\n"
         "- Concurrent processing with auto-retry and real-time progress\n\n"
         "*Best for: data enrichment, classification, cleaning, and bulk text analysis*"),
        ("generate", ":material/auto_awesome:", "Generate Data",
         "Describe what you need and let AI generate synthetic rows from scratch.",
         "- Generate realistic tabular data in CSV, JSON, or free-text format\n"
         "- Define custom schemas with typed fields or let AI decide the structure\n"
         "- Use variable cycling to produce diverse, controlled variations\n"
         "- Ideal for prototyping, testing, and training data creation\n\n"
         "*Best for: synthetic datasets, test fixtures, mock data, and augmentation*"),
        ("process-documents", ":material/description:", "Process Documents",
         "Extract structured data from PDFs, text files, and other documents.",
         "- Upload PDFs, DOCX, or TXT files for AI-powered extraction\n"
         "- Define output columns and let AI parse each document into rows\n"
         "- Batch-process entire folders of documents in parallel\n"
         "- Built-in CSV output formatting with master prompt enforcement\n\n"
         "*Best for: invoice parsing, resume extraction, report digitization*"),
        ("qualitative", ":material/psychology:", "Qualitative Coder",
         "Code qualitative data like interviews, observations, and surveys with AI.",
         "- Pre-configured prompt template optimized for qualitative coding\n"
         "- Enforces strict CSV output format — no prose, just codes and values\n"
         "- OpenAI cost estimation before running\n"
         "- Same powerful processing engine as Transform with coding-specific defaults\n\n"
         "*Best for: thematic analysis, interview coding, open-ended survey responses*"),
        ("consensus", ":material/groups:", "Consensus Coder",
         "Run multiple AI models in parallel and synthesize the best answer.",
         "- Configure 2-3 independent worker models plus a judge model\n"
         "- Workers process each row in parallel; judge synthesizes the best answer\n"
         "- Inter-rater reliability analytics: Cohen's Kappa, pairwise agreement, Jaccard index\n"
         "- Visualizations: judge alignment bar chart, consensus distribution pie chart\n\n"
         "*Best for: high-stakes coding, reducing AI bias, validation through model triangulation*"),
    ]

    # Row 1: 3 cards, Row 2: 2 cards
    row1 = st.columns(3)
    for col, (key, icon, title, tagline, details) in zip(row1, tool_cards[:3]):
        with col:
            with st.container(border=True):
                page = nav_pages.get(key)
                if page:
                    st.page_link(page, label=title, icon=icon)
                st.write(tagline)
                with st.expander("Learn more"):
                    st.markdown(details)

    row2 = st.columns(3)
    for col, (key, icon, title, tagline, details) in zip(row2, tool_cards[3:]):
        with col:
            with st.container(border=True):
                page = nav_pages.get(key)
                if page:
                    st.page_link(page, label=title, icon=icon)
                st.write(tagline)
                with st.expander("Learn more"):
                    st.markdown(details)

    st.divider()

    # System page links
    system_cards = [
        ("llm-providers", ":material/smart_toy:", "LLM Providers",
         "Configure AI providers, API keys, and default models. "
         "Supports OpenAI, Anthropic, Google Gemini, and more."),
        ("history", ":material/history:", "History",
         "View past sessions and runs. "
         "Review outputs, re-download results, and track usage."),
        ("settings", ":material/settings:", "Settings",
         "App preferences and configuration. "
         "Adjust defaults, concurrency limits, and display options."),
    ]

    sys_cols = st.columns(len(system_cards))
    for col, (key, icon, title, description) in zip(sys_cols, system_cards):
        with col:
            with st.container(border=True):
                page = nav_pages.get(key)
                if page:
                    st.page_link(page, label=title, icon=icon)
                st.caption(description)

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
