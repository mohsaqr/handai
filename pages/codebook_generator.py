"""
Handai Codebook Generator Page
Generate qualitative codebooks with AI
"""

import streamlit as st
import asyncio

from tools.codebook_generator import CodebookGeneratorTool
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
    """Render the codebook generator page"""
    initialize_session_state()
    register_default_tools()

    st.title("Codebook Generator")
    st.caption("Generate qualitative codebooks with AI-powered theme discovery and code definitions")

    # Sidebar - show active provider (configured in LLM Providers page)
    with st.sidebar:
        st.header(":material/smart_toy: AI Provider")

        active = _get_active_provider()

        if active:
            st.success(f"**{active['name']}**", icon=":material/check:")
            st.caption(f"Model: {active['model']}")

            # Store for the tool to use
            st.session_state["_codebook_provider"] = active["provider"]
            st.session_state["_codebook_api_key"] = active["api_key"]
            st.session_state["_codebook_base_url"] = active["base_url"]
            st.session_state["model_name"] = active["model"]
        else:
            st.warning("No provider configured")
            st.info("Go to **[LLM Providers](/llm-providers)** to set up a provider, or start **LM Studio** for local inference.")
            st.session_state["_codebook_provider"] = None

        st.divider()
        st.caption(":material/tune: [LLM Providers](/llm-providers) to configure models")

    # Main content
    tool = CodebookGeneratorTool()

    config = tool.render_config()

    if not config.is_valid and config.error_message:
        if "Please upload" not in config.error_message and "Please configure" not in config.error_message:
            st.error(config.error_message)

    # Execute button
    if config.config_data and config.config_data.get("df") is not None:
        st.header("5. Generate Codebook")

        run_generate = st.button(
            "Generate Codebook",
            type="primary",
            use_container_width=True,
            disabled=not config.is_valid,
            key="codebook_generate_btn"
        )

        if run_generate:
            if not config.is_valid:
                st.error(config.error_message or "Configuration is not valid")
                st.stop()

            progress = ProgressDisplay(title="Generation Progress", max_log_lines=5)

            def progress_callback(completed, total, success, errors, retries, log_entry, is_error):
                progress.update(completed, total, success, errors, retries, log_entry, is_error)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    tool.execute(config.config_data, progress_callback)
                )
            except Exception as e:
                st.error(f"Codebook generation failed: {str(e)}")
                return
            finally:
                loop.close()

            st.divider()
            tool.render_results(result)


if __name__ == "__main__":
    render()
