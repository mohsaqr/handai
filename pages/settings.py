"""
Handai Settings Page
Performance, storage, and general preferences
"""

import streamlit as st
import json

from ui.state import (
    initialize_session_state, save_setting,
)
from database import get_db
from config import (
    DEFAULT_MAX_CONCURRENCY, DEFAULT_TEST_BATCH_SIZE, DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
)
from core.prompt_registry import (
    PromptRegistry, ensure_prompts_registered, get_effective_prompt,
    get_prompt_status, set_temporary_override, set_permanent_override,
    reset_to_default, get_default_prompt
)


def render():
    """Render the settings page"""
    initialize_session_state()
    db = get_db()

    st.title("Settings")
    st.caption("Configure model defaults, performance, and storage options.")

    # Ensure prompts are registered
    ensure_prompts_registered()

    # Tabs for settings categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/tune: Model Defaults",
        ":material/speed: Performance",
        ":material/folder: Storage",
        ":material/code: Coding Tools",
        ":material/edit_note: System Prompts"
    ])

    # ==========================================
    # TAB: Model Defaults
    # ==========================================
    with tab1:
        st.header("Model Defaults")
        st.caption("Default settings used when running AI tools")

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature",
                0.0, 2.0,
                st.session_state.get("temperature", DEFAULT_TEMPERATURE),
                0.1,
                key="settings_temperature",
                help="0 = deterministic, 2 = creative"
            )
            if temperature != st.session_state.get("temperature"):
                st.session_state.temperature = temperature
                save_setting("temperature")

        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                1, 128000,
                st.session_state.get("max_tokens", DEFAULT_MAX_TOKENS),
                256,
                key="settings_max_tokens",
                help="Maximum response length"
            )
            if max_tokens != st.session_state.get("max_tokens"):
                st.session_state.max_tokens = max_tokens
                save_setting("max_tokens")

        json_mode = st.checkbox(
            "JSON Mode",
            st.session_state.get("json_mode", False),
            key="settings_json_mode",
            help="Force structured JSON output (when supported by provider)"
        )
        if json_mode != st.session_state.get("json_mode"):
            st.session_state.json_mode = json_mode
            save_setting("json_mode")

    # ==========================================
    # TAB: Performance
    # ==========================================
    with tab2:
        st.header("Performance Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Concurrency")

            max_concurrency = st.slider(
                "Max Concurrent Requests",
                1, 50,
                st.session_state.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
                key="settings_concurrency",
                help="Higher values = faster but more API load"
            )
            if max_concurrency != st.session_state.get("max_concurrency"):
                st.session_state.max_concurrency = max_concurrency
                save_setting("max_concurrency")

            test_batch_size = st.number_input(
                "Test Batch Size",
                1, 1000,
                st.session_state.get("test_batch_size", DEFAULT_TEST_BATCH_SIZE),
                key="settings_test_batch"
            )
            if test_batch_size != st.session_state.get("test_batch_size"):
                st.session_state.test_batch_size = test_batch_size
                save_setting("test_batch_size")

        with col2:
            st.subheader("Retry Settings")

            auto_retry = st.checkbox(
                "Auto-retry Failed Requests",
                st.session_state.get("auto_retry", True),
                key="settings_auto_retry",
                help="Automatically retry on errors"
            )
            if auto_retry != st.session_state.get("auto_retry"):
                st.session_state.auto_retry = auto_retry
                save_setting("auto_retry")

            if auto_retry:
                max_retries = st.number_input(
                    "Max Retries",
                    1, 5,
                    st.session_state.get("max_retries", DEFAULT_MAX_RETRIES),
                    key="settings_max_retries"
                )
                if max_retries != st.session_state.get("max_retries"):
                    st.session_state.max_retries = max_retries
                    save_setting("max_retries")

            realtime_progress = st.checkbox(
                "Real-Time Progress Updates",
                st.session_state.get("realtime_progress", True),
                key="settings_realtime",
                help="Update UI every row (slightly slower)"
            )
            if realtime_progress != st.session_state.get("realtime_progress"):
                st.session_state.realtime_progress = realtime_progress
                save_setting("realtime_progress")

    # ==========================================
    # TAB: Storage
    # ==========================================
    with tab3:
        st.header("Storage Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Auto-Save")

            save_path = st.text_input(
                "Auto-Save Path",
                st.session_state.get("save_path", ""),
                key="settings_save_path",
                placeholder="/path/to/save/results"
            )
            if save_path != st.session_state.get("save_path"):
                st.session_state.save_path = save_path
                save_setting("save_path")

            st.caption("Results will be auto-saved to this location during processing")

        with col2:
            st.subheader("Export/Import Settings")

            if st.button("Export All Settings", use_container_width=True):
                settings = db.get_all_settings()
                # Remove sensitive data
                settings_export = {k: v for k, v in settings.items() if "api_key" not in k.lower()}
                st.download_button(
                    "Download Settings JSON",
                    json.dumps(settings_export, indent=2),
                    "handai_settings.json",
                    "application/json"
                )

            uploaded_settings = st.file_uploader(
                "Import Settings",
                type=["json"],
                key="import_settings"
            )
            if uploaded_settings:
                try:
                    imported = json.load(uploaded_settings)
                    if st.button("Apply Imported Settings"):
                        db.save_all_settings(imported)
                        st.success("Settings imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error importing settings: {e}")

        st.divider()

        st.subheader("Database Info")
        stats = db.get_global_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sessions", stats.get("total_sessions", 0))
        with col2:
            st.metric("Total Runs", stats.get("total_runs", 0))
        with col3:
            st.metric("Rows Processed", stats.get("total_rows_processed", 0))

        if st.button("Clear All History", type="secondary"):
            if st.button("Confirm Clear All Data", type="primary"):
                # This would need a confirmation dialog in practice
                st.warning("This action cannot be undone!")

    # ==========================================
    # TAB: Coding Tools
    # ==========================================
    with tab4:
        st.header("Coding Tools Settings")
        st.caption("Display and behavior settings for AI Coder and Manual Coder")

        # === DISPLAY SETTINGS ===
        st.subheader("Display")
        col1, col2 = st.columns(2)

        with col1:
            light_mode = st.toggle(
                "Light mode",
                value=st.session_state.get("coding_light_mode", True),
                key="settings_coding_light_mode",
                help="Use light theme for coding interface"
            )
            st.session_state["coding_light_mode"] = light_mode

            context_rows = st.slider(
                "Context rows",
                min_value=0, max_value=5,
                value=st.session_state.get("coding_context_rows", 2),
                key="settings_coding_context_rows",
                help="Number of surrounding rows to show"
            )
            st.session_state["coding_context_rows"] = context_rows

        with col2:
            auto_advance = st.toggle(
                "Auto-advance after coding",
                value=st.session_state.get("coding_auto_advance", False),
                key="settings_coding_auto_advance",
                help="Move to next row after applying a code"
            )
            st.session_state["coding_auto_advance"] = auto_advance

        st.divider()

        # === LAYOUT SETTINGS ===
        st.subheader("Layout")
        col1, col2 = st.columns(2)

        with col1:
            horizontal = st.toggle(
                "Horizontal code buttons",
                value=st.session_state.get("coding_horizontal_codes", False),
                key="settings_coding_horizontal",
                help="Show codes in a horizontal row instead of sidebar"
            )
            st.session_state["coding_horizontal_codes"] = horizontal

        with col2:
            buttons_above = st.toggle(
                "Buttons above text",
                value=st.session_state.get("coding_buttons_above", False),
                key="settings_coding_buttons_above",
                help="Place code buttons above the text display"
            )
            st.session_state["coding_buttons_above"] = buttons_above

        st.divider()

        # === AI CODER SPECIFIC ===
        st.subheader("AI Coder")
        col1, col2 = st.columns(2)

        with col1:
            ai_mode = st.selectbox(
                "Default processing mode",
                options=["per_row", "batch"],
                format_func=lambda x: "Per-row" if x == "per_row" else "Batch",
                index=0 if st.session_state.get("aic_default_mode", "per_row") == "per_row" else 1,
                key="settings_aic_default_mode",
                help="Per-row: AI suggests as you navigate. Batch: Pre-process all rows."
            )
            st.session_state["aic_default_mode"] = ai_mode

        with col2:
            ai_display = st.selectbox(
                "Default display mode",
                options=["ai_first", "inline_badges"],
                format_func=lambda x: "AI First" if x == "ai_first" else "Inline Badges",
                index=0 if st.session_state.get("aic_default_display", "ai_first") == "ai_first" else 1,
                key="settings_aic_default_display",
                help="AI First: Show suggestions above text. Inline: Show confidence on buttons."
            )
            st.session_state["aic_default_display"] = ai_display

        aic_autosave = st.toggle(
            "Autosave enabled",
            value=st.session_state.get("aic_autosave_enabled", True),
            key="settings_aic_autosave",
            help="Auto-save AI Coder sessions on navigation"
        )
        st.session_state["aic_autosave_enabled"] = aic_autosave

        st.divider()

        # === MANUAL CODER SPECIFIC ===
        st.subheader("Manual Coder")
        mc_autosave = st.toggle(
            "Autosave enabled",
            value=st.session_state.get("mc_autosave_enabled", True),
            key="settings_mc_autosave",
            help="Auto-save Manual Coder sessions on navigation"
        )
        st.session_state["mc_autosave_enabled"] = mc_autosave

    # ==========================================
    # TAB: System Prompts
    # ==========================================
    with tab5:
        st.header("System Prompts")
        st.caption("Customize AI system prompts used by various tools. Changes can be temporary (session only) or permanent (saved to database).")

        # Quick actions for rigorous prompts
        st.subheader("Quick Actions")
        col_rig1, col_rig2, col_rig3 = st.columns([2, 2, 3])

        with col_rig1:
            if st.button("Use Rigorous Prompts", use_container_width=True, type="primary",
                        help="Replace all standard prompts with their rigorous research-grade variants"):
                # Find all rigorous prompts and set them as permanent overrides
                rigorous_applied = 0
                all_registered = PromptRegistry.get_all()
                for prompt_id, prompt_def in all_registered.items():
                    if prompt_id.endswith(".rigorous"):
                        # Get the base prompt ID (without .rigorous)
                        base_id = prompt_id.rsplit(".rigorous", 1)[0]
                        if base_id in all_registered:
                            set_permanent_override(base_id, prompt_def.default_value)
                            rigorous_applied += 1
                if rigorous_applied > 0:
                    st.success(f"Applied {rigorous_applied} rigorous prompts!")
                    st.rerun()
                else:
                    st.warning("No rigorous prompts found to apply.")

        with col_rig2:
            if st.button("Reset All to Default", use_container_width=True,
                        help="Remove all overrides and restore original prompts"):
                reset_count = 0
                all_registered = PromptRegistry.get_all()
                for prompt_id in all_registered.keys():
                    if not prompt_id.endswith(".rigorous"):
                        status = get_prompt_status(prompt_id)
                        if status != "default":
                            reset_to_default(prompt_id)
                            reset_count += 1
                if reset_count > 0:
                    st.success(f"Reset {reset_count} prompts to default!")
                    st.rerun()
                else:
                    st.info("All prompts are already using defaults.")

        with col_rig3:
            st.caption("Rigorous prompts are designed for academic research with detailed methodological guidelines.")

        st.divider()

        # Get all registered prompts
        all_prompts = PromptRegistry.get_all()
        categories = PromptRegistry.get_categories()

        if not all_prompts:
            st.info("No system prompts registered.")
        else:
            # Category filter
            selected_category = st.selectbox(
                "Filter by Category",
                ["All Categories"] + categories,
                key="prompt_category_filter"
            )

            # Filter prompts by category (exclude .rigorous variants from main list)
            if selected_category == "All Categories":
                filtered_prompts = {k: v for k, v in all_prompts.items() if not k.endswith(".rigorous")}
            else:
                category_prompts = PromptRegistry.get_by_category(selected_category)
                filtered_prompts = {k: v for k, v in category_prompts.items() if not k.endswith(".rigorous")}

            st.divider()

            # Render each prompt
            for prompt_id, prompt_def in sorted(filtered_prompts.items(), key=lambda x: (x[1].category, x[1].name)):
                status = get_prompt_status(prompt_id)

                # Status indicator
                if status == "session":
                    status_badge = ":orange[Session Override]"
                elif status == "permanent":
                    status_badge = ":blue[Permanent Override]"
                else:
                    status_badge = ":green[Using Default]"

                # Create expander for each prompt
                with st.expander(f"**{prompt_def.name}** - {status_badge}", expanded=False):
                    st.caption(f"**Module:** {prompt_def.module} | **ID:** `{prompt_id}`")
                    st.markdown(f"*{prompt_def.description}*")

                    # Get current effective value
                    current_value = get_effective_prompt(prompt_id)
                    default_value = get_default_prompt(prompt_id)

                    # Text area for editing
                    edited_value = st.text_area(
                        "Prompt Content",
                        value=current_value,
                        height=200,
                        key=f"prompt_edit_{prompt_id}",
                        label_visibility="collapsed"
                    )

                    # Check if value has been modified
                    is_modified = edited_value != current_value

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        if st.button(
                            "Save Temporary",
                            key=f"save_temp_{prompt_id}",
                            use_container_width=True,
                            disabled=not is_modified,
                            help="Save for this session only (will be lost on restart)"
                        ):
                            set_temporary_override(prompt_id, edited_value)
                            st.success("Saved as temporary override!")
                            st.rerun()

                    with col2:
                        if st.button(
                            "Save Permanent",
                            key=f"save_perm_{prompt_id}",
                            use_container_width=True,
                            disabled=not is_modified,
                            help="Save to database (persists across sessions)"
                        ):
                            set_permanent_override(prompt_id, edited_value)
                            st.success("Saved as permanent override!")
                            st.rerun()

                    with col3:
                        if st.button(
                            "Reset to Default",
                            key=f"reset_{prompt_id}",
                            use_container_width=True,
                            disabled=status == "default",
                            help="Remove all overrides and use the default value"
                        ):
                            reset_to_default(prompt_id)
                            st.success("Reset to default!")
                            st.rerun()

                    with col4:
                        if st.button(
                            "View Default",
                            key=f"view_default_{prompt_id}",
                            use_container_width=True,
                            help="Show the original default value"
                        ):
                            st.session_state[f"show_default_{prompt_id}"] = not st.session_state.get(f"show_default_{prompt_id}", False)
                            st.rerun()

                    # Show default value if requested
                    if st.session_state.get(f"show_default_{prompt_id}", False):
                        st.markdown("---")
                        st.markdown("**Default Value:**")
                        st.code(default_value, language="text")


if __name__ == "__main__":
    render()
