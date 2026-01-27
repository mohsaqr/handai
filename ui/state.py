"""
Handai Session State Management
Utilities for managing Streamlit session state and persistent settings
"""

import streamlit as st
from typing import Any, Dict, Optional, List
from database import get_db
from core.providers import LLMProvider, get_provider_names
from config import (
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TEST_BATCH_SIZE, DEFAULT_MAX_RETRIES
)


def get_db_instance():
    """Get database instance, cached for session"""
    if "db" not in st.session_state:
        st.session_state.db = get_db()
    return st.session_state.db


def initialize_session_state():
    """Initialize all session state defaults and load from database"""
    if "state_initialized" not in st.session_state:
        db = get_db_instance()

        # Load persistent settings from database
        saved_settings = db.get_all_settings()

        # Define defaults
        defaults = {
            "selected_provider": "OpenAI",
            "base_url": "",
            "model_name": "gpt-4o",
            "custom_model": "",
            "use_custom_model": False,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "json_mode": False,
            "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            "test_batch_size": DEFAULT_TEST_BATCH_SIZE,
            "realtime_progress": True,
            "save_path": "",
            "auto_retry": True,
            "max_retries": DEFAULT_MAX_RETRIES,
            "dataset_mode": "Transform Existing Dataset",
            "schema_mode": "Free-form (AI decides structure)",
            "system_prompt": "",
            "generation_prompt": "",
            "custom_fields": [],
            "gen_variables": [],
        }

        # Apply saved settings over defaults
        for key, default_value in defaults.items():
            if key in saved_settings:
                st.session_state[key] = saved_settings[key]
            elif key not in st.session_state:
                st.session_state[key] = default_value

        st.session_state.state_initialized = True


def save_setting(key: str):
    """Save a single setting to the database when it changes"""
    if key in st.session_state:
        db = get_db_instance()
        db.save_setting(key, st.session_state[key])


def save_all_current_settings():
    """Save all current settings to database"""
    settings_keys = [
        "selected_provider", "base_url", "model_name", "custom_model",
        "temperature", "max_tokens", "json_mode", "max_concurrency", "test_batch_size",
        "realtime_progress", "save_path", "auto_retry", "max_retries",
        "dataset_mode", "schema_mode", "generation_prompt", "system_prompt"
    ]
    db = get_db_instance()
    settings = {k: st.session_state.get(k) for k in settings_keys if k in st.session_state}
    db.save_all_settings(settings)


def load_persistent_settings():
    """Load settings from database into session state"""
    db = get_db_instance()
    saved = db.get_all_settings()
    if saved:
        for key, value in saved.items():
            if key not in st.session_state:
                st.session_state[key] = value


def get_current_settings(include_sensitive: bool = False) -> dict:
    """
    Get current settings from session state.

    Args:
        include_sensitive: If True, include API keys

    Returns:
        Dictionary of current settings
    """
    # Settings to include (NO API keys by default)
    session_keys = [
        "selected_provider", "base_url", "model_name", "custom_model",
        "temperature", "max_tokens", "json_mode", "max_concurrency", "test_batch_size",
        "realtime_progress", "save_path", "auto_retry", "max_retries",
        "dataset_mode", "schema_mode"
    ]

    # Only include API key for full persistence
    if include_sensitive:
        session_keys.append("api_key")

    return {k: st.session_state.get(k) for k in session_keys if k in st.session_state}


# ==========================================
# PROVIDER-SPECIFIC API KEY MANAGEMENT
# ==========================================

def get_api_key_for_provider(provider_name: str) -> str:
    """
    Get API key for a specific provider from database.

    Args:
        provider_name: Display name of the provider (e.g., "OpenAI")

    Returns:
        API key string or empty string if not set
    """
    db = get_db_instance()
    key = db.get_provider_setting(provider_name, "api_key", "")
    return key if key else ""


def set_api_key_for_provider(provider_name: str, api_key: str):
    """
    Save API key for a specific provider to database.

    Args:
        provider_name: Display name of the provider
        api_key: The API key to save
    """
    db = get_db_instance()
    if api_key:
        db.save_provider_setting(provider_name, "api_key", api_key)
    else:
        db.delete_provider_setting(provider_name, "api_key")


def get_default_model_for_provider(provider_name: str) -> Optional[str]:
    """Get default model for a provider"""
    db = get_db_instance()
    return db.get_provider_setting(provider_name, "default_model")


def set_default_model_for_provider(provider_name: str, model: str):
    """Set default model for a provider"""
    db = get_db_instance()
    db.save_provider_setting(provider_name, "default_model", model)


def get_providers_with_api_keys() -> Dict[str, bool]:
    """Get dict of providers and whether they have API keys configured"""
    db = get_db_instance()
    return db.get_all_providers_with_api_keys()


def get_all_provider_settings(provider_name: str) -> Dict[str, Any]:
    """Get all settings for a specific provider"""
    db = get_db_instance()
    return db.get_all_provider_settings(provider_name)


# ==========================================
# SESSION MANAGEMENT
# ==========================================

def get_current_session_id() -> Optional[str]:
    """Get current session ID from state"""
    return st.session_state.get("current_session_id")


def set_current_session_id(session_id: str):
    """Set current session ID in state"""
    st.session_state.current_session_id = session_id


def clear_current_session():
    """Clear current session from state"""
    if "current_session_id" in st.session_state:
        del st.session_state.current_session_id


# ==========================================
# PROVIDER STATE HELPERS
# ==========================================

def get_selected_provider() -> LLMProvider:
    """Get currently selected provider as enum"""
    provider_name = st.session_state.get("selected_provider", "OpenAI")
    try:
        return LLMProvider(provider_name)
    except ValueError:
        # Check if it's a configured provider display name
        from database import get_db
        db = get_db()
        for p in db.get_enabled_configured_providers():
            if p.display_name == provider_name:
                try:
                    return LLMProvider(p.provider_type)
                except ValueError:
                    return LLMProvider.CUSTOM
        return LLMProvider.OPENAI


def get_selected_model() -> str:
    """Get currently selected model name"""
    if st.session_state.get("use_custom_model"):
        return st.session_state.get("custom_model", "")
    return st.session_state.get("model_name", "gpt-4o")


def get_effective_api_key() -> str:
    """Get effective API key for current provider"""
    provider_name = st.session_state.get("selected_provider", "OpenAI")
    # First check configured_providers table
    from database import get_db
    db = get_db()
    for p in db.get_enabled_configured_providers():
        if p.display_name == provider_name or p.provider_type == provider_name:
            if p.api_key:
                return p.api_key
    # Then check legacy provider_settings
    key = get_api_key_for_provider(provider_name)
    if key:
        return key
    # Fall back to general api_key in session state (for backward compatibility)
    return st.session_state.get("api_key", "")
