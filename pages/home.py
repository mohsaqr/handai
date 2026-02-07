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

    # Larger font styling for home page
    st.markdown("""
    <style>
        /* Larger title */
        .main h1 { font-size: 3rem !important; }
        /* Larger card text */
        .main .stMarkdown p { font-size: 1.25rem !important; line-height: 1.6 !important; }
        /* Larger headers */
        .main h2 { font-size: 2.2rem !important; }
        .main h3 { font-size: 1.6rem !important; }
        /* Bigger, bolder page links */
        .main .stPageLink span {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
        }
        /* Bigger icons in page links */
        .main .stPageLink [data-testid="stIcon"] {
            width: 2rem !important;
            height: 2rem !important;
        }
        /* Make the entire link area more prominent */
        .main .stPageLink {
            padding: 0.5rem 0 !important;
        }
        /* Larger metric values */
        .main [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
        /* Larger metric labels */
        .main [data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
        }
        /* Larger expander text */
        .main .streamlit-expanderHeader {
            font-size: 1.15rem !important;
        }
        /* Caption text */
        .main .stCaption {
            font-size: 1.05rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Handai")
    st.markdown("""
    **AI-Powered Data Transformation & Generation**

    Upload datasets for AI-powered transformation, or generate entirely new synthetic datasets.
    """)

    nav_pages = st.session_state.get("_pages", {})

    # Tool cards
    st.header("Available Tools")

    # Data Processing Tools
    st.subheader("Data Processing")
    data_tools = [
        ("transform", ":material/sync:", "Transform Data",
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
        ("automator", ":material/precision_manufacturing:", "Automator",
         "Build multi-step AI pipelines with branching logic and automation.",
         "- Chain multiple AI operations together in sequence\n"
         "- Conditional branching based on previous step outputs\n"
         "- Save and reuse automation templates\n"
         "- Schedule recurring data processing jobs\n\n"
         "*Best for: complex workflows, ETL pipelines, recurring batch jobs*"),
    ]

    row1 = st.columns(4)
    for col, (key, icon, title, tagline, details) in zip(row1, data_tools):
        with col:
            with st.container(border=True):
                page = nav_pages.get(key)
                if page:
                    st.page_link(page, label=title, icon=icon)
                st.write(tagline)
                with st.expander("Learn more"):
                    st.markdown(details)

    # Qualitative Analysis Tools
    st.subheader("Qualitative Analysis")
    qual_tools = [
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
        ("codebook-generator", ":material/book:", "Codebook Generator",
         "Generate structured codebooks from qualitative data using AI.",
         "- Automatically identify themes and patterns in your data\n"
         "- Generate code definitions, examples, and inclusion criteria\n"
         "- Export codebooks in various formats (CSV, JSON, Word)\n"
         "- Iteratively refine codes with AI assistance\n\n"
         "*Best for: grounded theory, codebook development, thematic framework creation*"),
        ("manual-coder", ":material/touch_app:", "Manual Coder",
         "Code qualitative data manually with clickable codes and keyboard shortcuts.",
         "- Fast, distraction-free immersive coding mode\n"
         "- Clickable code buttons with customizable highlights\n"
         "- Session save/load with auto-restore on refresh\n"
         "- Export coded data with one-hot encoding or text format\n"
         "- Sample datasets included for practice\n\n"
         "*Best for: human coding, inter-rater reliability studies, training coders*"),
        ("ai-coder", ":material/smart_toy:", "AI Coder",
         "AI-assisted manual coding with inter-rater reliability analytics.",
         "- All features of Manual Coder plus AI suggestions\n"
         "- Per-row or batch AI processing modes\n"
         "- Three display modes: AI First, Side-by-side, Inline Badges\n"
         "- Inter-rater reliability: Cohen's Kappa, Jaccard, Precision/Recall\n"
         "- Confusion matrix and disagreement analysis\n"
         "- Export with AI codes and confidence scores\n\n"
         "*Best for: AI-assisted coding, human-AI comparison, reliability studies*"),
    ]

    row2 = st.columns(5)
    for col, (key, icon, title, tagline, details) in zip(row2, qual_tools):
        with col:
            with st.container(border=True):
                page = nav_pages.get(key)
                if page:
                    st.page_link(page, label=title, icon=icon)
                st.write(tagline)
                with st.expander("Learn more"):
                    st.markdown(details)

    st.divider()

    # Quick stats (below cards)
    stats = db.get_global_stats()
    if stats.get("total_runs", 0) > 0:
        st.header("Statistics")
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
