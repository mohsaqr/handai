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

# Import pages
from pages import home, transform, generate, process_documents, history, settings, models

# Define pages for navigation with explicit URL paths
page_home = st.Page(home.render, title="Home", icon=":material/home:", url_path="home")
page_transform = st.Page(transform.render, title="Transform Data", icon=":material/sync:", url_path="transform")
page_generate = st.Page(generate.render, title="Generate Data", icon=":material/auto_awesome:", url_path="generate")
page_process_docs = st.Page(process_documents.render, title="Process Documents", icon=":material/description:", url_path="process-documents")
page_models = st.Page(models.render, title="LLM Providers", icon=":material/model_training:", url_path="llm-providers")
page_history = st.Page(history.render, title="History", icon=":material/history:", url_path="history")
page_settings = st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings")

pages = {
    "Main": [page_home, page_transform, page_generate, page_process_docs],
    "System": [page_models, page_history, page_settings],
}

# Store page objects so other pages can reference them for navigation
st.session_state["_pages"] = {
    "transform": page_transform,
    "generate": page_generate,
    "process-documents": page_process_docs,
    "llm-providers": page_models,
    "history": page_history,
    "settings": page_settings,
}

# Create navigation
nav = st.navigation(pages)

# Run the selected page
nav.run()
