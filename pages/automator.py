"""
Handai General Purpose Automator Page
Apply any AI task across your dataset row by row
"""

import streamlit as st
import asyncio

from tools.automator import AutomatorTool
from tools.registry import register_default_tools
from ui.state import initialize_session_state
from ui.components.progress_display import ProgressDisplay
from database import get_db
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.llm_client import fetch_local_models


def _get_active_provider():
    """Get the active provider from configured providers, auto-detecting LM Studio if available."""
    db = get_db()

    # First, try to auto-detect and enable LM Studio if running
    lm_studio_url = "http://localhost:1234/v1"
    try:
        models = fetch_local_models(lm_studio_url)
        if models:
            # Filter out embedding models - we need chat/instruct models
            chat_models = [m for m in models if "embed" not in m.lower()]
            if not chat_models:
                chat_models = models  # Fallback if all are filtered

            selected_model = chat_models[0]

            # LM Studio is running - ensure it's configured and enabled
            providers = db.get_all_configured_providers()
            lm_studio = next((p for p in providers if p.provider_type == "LM Studio (Local)"), None)

            if lm_studio:
                if not lm_studio.is_enabled or not lm_studio.default_model:
                    # Update with detected model and enable
                    db.update_configured_provider(
                        lm_studio.id,
                        is_enabled=True,
                        default_model=selected_model
                    )
                return {
                    "provider": LLMProvider.LM_STUDIO,
                    "api_key": "dummy",
                    "base_url": lm_studio_url,
                    "model": selected_model,
                    "name": "LM Studio (Auto-detected)"
                }
    except Exception:
        pass  # LM Studio not running

    # Get enabled configured providers
    enabled = db.get_enabled_configured_providers()
    if enabled:
        p = enabled[0]  # Use first enabled provider
        try:
            provider_enum = LLMProvider(p.provider_type)
        except ValueError:
            provider_enum = LLMProvider.CUSTOM

        return {
            "provider": provider_enum,
            "api_key": p.api_key or "dummy",
            "base_url": p.base_url,
            "model": p.default_model,
            "name": p.display_name
        }

    return None


def render():
    """Render the automator page"""
    initialize_session_state()
    register_default_tools()

    st.title("General Purpose Automator")
    st.caption("Apply any AI task across your dataset - classify, extract, summarize, translate, score, tag, and more")

    # Sidebar - show active provider
    with st.sidebar:
        st.header(":material/smart_toy: AI Provider")

        active = _get_active_provider()

        if active:
            st.success(f"**{active['name']}**", icon=":material/check:")
            st.caption(f"Model: {active['model']}")

            # Store for the tool to use
            st.session_state["_automator_provider"] = active["provider"]
            st.session_state["_automator_api_key"] = active["api_key"]
            st.session_state["_automator_base_url"] = active["base_url"]
            st.session_state["model_name"] = active["model"]
        else:
            st.warning("No provider configured")
            st.info("Go to **[LLM Providers](/llm-providers)** to set up a provider, or start **LM Studio** for local inference.")
            st.session_state["_automator_provider"] = None

        st.divider()
        st.caption(":material/tune: [LLM Providers](/llm-providers) to configure models")
        st.caption(":material/settings: [Settings](/settings) for performance options")

    # Main content
    tool = AutomatorTool()

    config = tool.render_config()

    if not config.is_valid and config.error_message:
        if "Please upload" not in config.error_message and "No AI provider" not in config.error_message:
            st.error(config.error_message)

    # Execute buttons
    if config.config_data and config.config_data.get("df") is not None:
        st.header("7. Execute")

        col1, col2 = st.columns(2)
        test_batch_size = st.session_state.get("test_batch_size", 10)

        with col1:
            run_test = st.button(
                f"Test Run ({test_batch_size} rows)",
                type="primary",
                use_container_width=True,
                disabled=not config.is_valid,
                key="automator_run_test"
            )
        with col2:
            run_full = st.button(
                "Full Run",
                type="secondary",
                use_container_width=True,
                disabled=not config.is_valid,
                key="automator_run_full"
            )

        if run_test or run_full:
            if not config.is_valid:
                st.error(config.error_message or "Configuration is not valid")
                st.stop()

            # Prepare execution config
            exec_config = config.config_data.copy()
            exec_config["is_test"] = run_test

            # Show generated system prompt in expander
            from tools.automator import build_system_prompt
            with st.expander("Generated System Prompt", expanded=False):
                system_prompt = build_system_prompt(exec_config["automator_config"])
                st.code(system_prompt, language="text")

            # Create progress display
            progress = ProgressDisplay(title="Processing Status", max_log_lines=10)

            def progress_callback(completed, total, success, errors, retries, log_entry, is_error):
                progress.update(completed, total, success, errors, retries, log_entry, is_error)

            # Run async execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    tool.execute(exec_config, progress_callback)
                )
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                return
            finally:
                loop.close()

            # Display results
            st.divider()
            tool.render_results(result)


if __name__ == "__main__":
    render()
