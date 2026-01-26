"""
LLM Controls Component
UI for temperature, max tokens, JSON mode, and other LLM settings
"""

import streamlit as st
from typing import Tuple
from core.providers import LLMProvider
from ui.state import save_setting
from config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


def render_llm_controls(
    provider: LLMProvider = None,
    key_prefix: str = "main",
    show_header: bool = True
) -> Tuple[float, int, bool]:
    """
    Render LLM control settings.

    Args:
        provider: Current provider (for JSON mode warning)
        key_prefix: Prefix for widget keys
        show_header: Whether to show section header

    Returns:
        Tuple of (temperature, max_tokens, json_mode)
    """
    if show_header:
        st.header("LLM Controls")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get("temperature", DEFAULT_TEMPERATURE),
        step=0.1,
        key=f"{key_prefix}_temperature",
        help="0 = deterministic, 2 = very creative",
        on_change=lambda: save_setting("temperature")
    )
    st.session_state.temperature = temperature

    max_tokens = st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=128000,
        value=st.session_state.get("max_tokens", DEFAULT_MAX_TOKENS),
        step=256,
        key=f"{key_prefix}_max_tokens",
        help="Maximum response length",
        on_change=lambda: save_setting("max_tokens")
    )
    st.session_state.max_tokens = max_tokens

    json_mode = st.checkbox(
        "JSON Mode",
        value=st.session_state.get("json_mode", False),
        key=f"{key_prefix}_json_mode",
        help="Force structured JSON output (supported by most providers)",
        on_change=lambda: save_setting("json_mode")
    )
    st.session_state.json_mode = json_mode

    # Warning for local providers
    if json_mode and provider in [LLMProvider.LM_STUDIO, LLMProvider.OLLAMA, LLMProvider.CUSTOM]:
        st.warning("JSON mode may not work with local models. Will be skipped automatically.")

    return temperature, max_tokens, json_mode


def render_performance_controls(
    key_prefix: str = "main",
    show_header: bool = True
) -> Tuple[int, int, bool, int, bool, str]:
    """
    Render performance and execution settings.

    Args:
        key_prefix: Prefix for widget keys
        show_header: Whether to show section header

    Returns:
        Tuple of (max_concurrency, test_batch_size, auto_retry, max_retries, realtime_progress, save_path)
    """
    if show_header:
        st.header("Performance")

    max_concurrency = st.slider(
        "Max Concurrent Requests",
        1, 50,
        st.session_state.get("max_concurrency", 5),
        key=f"{key_prefix}_max_concurrency",
        on_change=lambda: save_setting("max_concurrency")
    )
    st.session_state.max_concurrency = max_concurrency

    test_batch_size = st.number_input(
        "Test Batch Size",
        1, 1000,
        st.session_state.get("test_batch_size", 10),
        key=f"{key_prefix}_test_batch_size",
        on_change=lambda: save_setting("test_batch_size")
    )
    st.session_state.test_batch_size = test_batch_size

    auto_retry = st.checkbox(
        "Auto-retry empty/failed",
        value=st.session_state.get("auto_retry", True),
        key=f"{key_prefix}_auto_retry",
        help="Automatically retry rows that return empty or error",
        on_change=lambda: save_setting("auto_retry")
    )
    st.session_state.auto_retry = auto_retry

    max_retries = 0
    if auto_retry:
        max_retries = st.number_input(
            "Max Retries",
            1, 5,
            st.session_state.get("max_retries", 3),
            key=f"{key_prefix}_max_retries",
            on_change=lambda: save_setting("max_retries")
        )
        st.session_state.max_retries = max_retries

    realtime_progress = st.checkbox(
        "Real-Time Progress",
        value=st.session_state.get("realtime_progress", True),
        key=f"{key_prefix}_realtime_progress",
        help="Update UI every row (slightly slower)",
        on_change=lambda: save_setting("realtime_progress")
    )
    st.session_state.realtime_progress = realtime_progress

    save_path = st.text_input(
        "Auto-Save Path",
        value=st.session_state.get("save_path", ""),
        placeholder="/path/to/save/results",
        key=f"{key_prefix}_save_path",
        on_change=lambda: save_setting("save_path")
    )
    st.session_state.save_path = save_path

    return max_concurrency, test_batch_size, auto_retry, max_retries, realtime_progress, save_path


def render_llm_controls_compact(key_prefix: str = "compact") -> Tuple[float, int, bool]:
    """
    Render compact LLM controls in columns.

    Returns:
        Tuple of (temperature, max_tokens, json_mode)
    """
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        temperature = st.slider(
            "Temp",
            0.0, 2.0,
            st.session_state.get("temperature", 0.0),
            0.1,
            key=f"{key_prefix}_temp"
        )

    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            1, 128000,
            st.session_state.get("max_tokens", 2048),
            256,
            key=f"{key_prefix}_tokens"
        )

    with col3:
        json_mode = st.checkbox(
            "JSON",
            st.session_state.get("json_mode", False),
            key=f"{key_prefix}_json"
        )

    return temperature, max_tokens, json_mode
