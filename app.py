"""
Handai v4.0 - AI Data Transformer & Generator
Main entry point with multi-page navigation
"""

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Handai",
    layout="wide",
    page_icon=":material/sync:",
    initial_sidebar_state="expanded"
)

# Enlarge sidebar navigation text
st.markdown("""
<style>
    /* Sidebar nav link labels - bigger */
    [data-testid="stSidebarNav"] span {
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    /* Sidebar section headers - bigger */
    [data-testid="stSidebarNav"] h2,
    [data-testid="stSidebarNav"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    /* Material icons in sidebar - bigger */
    [data-testid="stSidebarNav"] [data-testid="stIcon"] {
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    /* Sidebar header styling */
    .sidebar .stMarkdown h1, .sidebar .stMarkdown h2, .sidebar .stMarkdown h3 {
        font-size: 1.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Import pages
from pages import home, transform, generate, process_documents, history, settings, models, qualitative, consensus, codebook_generator, automator, manual_coder

# Define pages for navigation with explicit URL paths
page_home = st.Page(home.render, title="Home", icon=":material/home:", url_path="home")
page_transform = st.Page(transform.render, title="Transform Data", icon=":material/sync:", url_path="transform")
page_generate = st.Page(generate.render, title="Generate Data", icon=":material/auto_awesome:", url_path="generate")
page_process_docs = st.Page(process_documents.render, title="Process Documents", icon=":material/description:", url_path="process-documents")
page_automator = st.Page(automator.render, title="Automator", icon=":material/precision_manufacturing:", url_path="automator")
page_qualitative = st.Page(qualitative.render, title="Qualitative Coder", icon=":material/psychology:", url_path="qualitative")
page_consensus = st.Page(consensus.render, title="Consensus Coder", icon=":material/groups:", url_path="consensus")
page_codebook = st.Page(codebook_generator.render, title="Codebook Generator", icon=":material/book:", url_path="codebook-generator")
page_manual_coder = st.Page(manual_coder.render, title="Manual Coder", icon=":material/touch_app:", url_path="manual-coder")
page_models = st.Page(models.render, title="LLM Providers", icon=":material/model_training:", url_path="llm-providers")
page_history = st.Page(history.render, title="History", icon=":material/history:", url_path="history")
page_settings = st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings")

pages = {
    "Main": [page_home, page_transform, page_generate, page_process_docs, page_automator, page_qualitative, page_consensus, page_codebook, page_manual_coder],
    "System": [page_models, page_history, page_settings],
}

# Store page objects so other pages can reference them for navigation
st.session_state["_pages"] = {
    "transform": page_transform,
    "generate": page_generate,
    "process-documents": page_process_docs,
    "automator": page_automator,
    "qualitative": page_qualitative,
    "consensus": page_consensus,
    "codebook-generator": page_codebook,
    "manual-coder": page_manual_coder,
    "llm-providers": page_models,
    "history": page_history,
    "settings": page_settings,
}

# Create navigation
nav = st.navigation(pages)

# Run the selected page
nav.run()
