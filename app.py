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
pages = {
    "Main": [
        st.Page(home.render, title="Home", icon=":material/home:", url_path="home"),
        st.Page(transform.render, title="Transform Data", icon=":material/sync:", url_path="transform"),
        st.Page(generate.render, title="Generate Data", icon=":material/auto_awesome:", url_path="generate"),
        st.Page(process_documents.render, title="Process Documents", icon=":material/description:", url_path="process-documents"),
    ],
    "System": [
        st.Page(models.render, title="LLM Providers", icon=":material/model_training:", url_path="llm-providers"),
        st.Page(history.render, title="History", icon=":material/history:", url_path="history"),
        st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
    ]
}

# Create navigation
nav = st.navigation(pages)

# Run the selected page
nav.run()
