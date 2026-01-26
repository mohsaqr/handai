"""
Generate Tool
Tool for generating new synthetic datasets with AI
"""

import streamlit as st
import pandas as pd
import json
import asyncio
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.templates import DATASET_TEMPLATES, get_template_names
from core.processing import GenerateProcessor, ProcessingConfig
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model,
    get_current_settings, set_current_session_id, get_current_session_id
)
from config import VARIATION_TEMPS, TYPE_MAP


class GenerateTool(BaseTool):
    """Tool for generating new synthetic datasets with AI"""

    id = "generate"
    name = "Generate Data"
    description = "Generate new synthetic datasets using AI"
    icon = ""
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render generate tool configuration UI"""

        st.header("1. Describe Your Dataset")

        # Initialize session state for dynamic fields
        if "custom_fields" not in st.session_state:
            st.session_state.custom_fields = []
        if "gen_variables" not in st.session_state:
            st.session_state.gen_variables = []

        # Free-form description
        generation_prompt = st.text_area(
            "What kind of data do you want to generate?",
            height=120,
            placeholder="Example: Generate realistic customer profiles with names, emails, purchase history...",
            key="generate_prompt",
            help="Just describe what you need - the AI will figure out the structure."
        )

        # Generation settings inline
        col1, col2, col3 = st.columns(3)
        with col1:
            num_rows = st.number_input("Rows to Generate", 1, 10000, 100, key="gen_num_rows")
        with col2:
            variation_level = st.select_slider(
                "Variation",
                options=["Low", "Medium", "High", "Maximum"],
                value="Medium",
                key="gen_variation"
            )
        with col3:
            output_format = st.selectbox(
                "Output Format",
                ["Auto-detect", "Structured JSON", "Free text"],
                key="gen_output_format"
            )

        gen_temperature = VARIATION_TEMPS[variation_level]

        # Schema mode selection
        schema_mode = st.radio(
            "Schema Definition",
            ["Free-form (AI decides structure)", "Custom Fields (I'll define)", "Use Template"],
            horizontal=True,
            key="gen_schema_mode"
        )

        schema = {}
        use_freeform = False

        if schema_mode == "Free-form (AI decides structure)":
            use_freeform = True
            st.info("The AI will determine the best structure based on your description above.")

        elif schema_mode == "Custom Fields (I'll define)":
            schema = self._render_schema_builder()

        else:  # Use Template
            schema = self._render_template_selector()

        # Variables section
        variables = self._render_variables_section()

        # Validate
        if not generation_prompt or not generation_prompt.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please describe what data you want to generate",
                config_data={
                    "num_rows": num_rows,
                    "schema": schema,
                    "use_freeform": use_freeform,
                    "variables": variables
                }
            )

        # Get settings from session state
        provider = get_selected_provider()
        api_key = get_effective_api_key()
        model = get_selected_model()

        if PROVIDER_CONFIGS[provider].requires_api_key and not api_key:
            return ToolConfig(
                is_valid=False,
                error_message="Please enter an API key for the selected provider",
                config_data={
                    "generation_prompt": generation_prompt,
                    "num_rows": num_rows,
                    "schema": schema,
                    "use_freeform": use_freeform,
                    "variables": variables
                }
            )

        # Build config
        config_data = {
            "generation_prompt": generation_prompt,
            "num_rows": num_rows,
            "schema": schema,
            "use_freeform": use_freeform,
            "variables": variables,
            "gen_temperature": gen_temperature,
            "provider": provider,
            "api_key": api_key,
            "base_url": st.session_state.get("base_url"),
            "model": model,
            "max_tokens": st.session_state.get("max_tokens", 2048),
            "max_concurrency": st.session_state.get("max_concurrency", 5),
            "auto_retry": st.session_state.get("auto_retry", True),
            "max_retries": st.session_state.get("max_retries", 3),
        }

        return ToolConfig(is_valid=True, config_data=config_data)

    def _render_schema_builder(self) -> Dict[str, str]:
        """Render the custom schema builder UI"""
        schema = {}

        with st.expander("Schema Builder - Add Fields", expanded=True):
            st.caption("Add fields one by one to define your dataset structure")

            with st.form("add_field_form", clear_on_submit=True):
                fc1, fc2, fc3, fc4 = st.columns([2, 1, 3, 1])
                with fc1:
                    new_field_name = st.text_input("Field Name", placeholder="e.g., customer_name")
                with fc2:
                    new_field_type = st.selectbox("Type", list(TYPE_MAP.keys()))
                with fc3:
                    new_field_desc = st.text_input("Description (optional)", placeholder="e.g., Full name")
                with fc4:
                    st.write("")
                    add_field = st.form_submit_button("Add", use_container_width=True)

                if add_field and new_field_name:
                    st.session_state.custom_fields.append({
                        "name": new_field_name,
                        "type": new_field_type,
                        "description": new_field_desc
                    })
                    st.rerun()

            if st.session_state.custom_fields:
                st.write("**Current Fields:**")
                for idx, field in enumerate(st.session_state.custom_fields):
                    fc1, fc2, fc3, fc4 = st.columns([2, 1, 3, 1])
                    with fc1:
                        st.text(field["name"])
                    with fc2:
                        st.text(field["type"])
                    with fc3:
                        st.text(field.get("description", "-"))
                    with fc4:
                        if st.button("Remove", key=f"del_field_{idx}"):
                            st.session_state.custom_fields.pop(idx)
                            st.rerun()

                if st.button("Clear All Fields"):
                    st.session_state.custom_fields = []
                    st.rerun()

                schema = {f["name"]: TYPE_MAP.get(f["type"], "str") for f in st.session_state.custom_fields}
                st.json(schema)
            else:
                st.warning("No fields added yet.")

        return schema

    def _render_template_selector(self) -> Dict[str, str]:
        """Render the template selector UI"""
        schema = {}

        with st.expander("Choose a Template", expanded=True):
            template_names = get_template_names()
            selected_template = st.selectbox("Template", template_names, key="selected_gen_template")
            template = DATASET_TEMPLATES[selected_template]
            st.caption(f"{template['description']}")

            if selected_template != "Custom (Define Your Own)":
                schema = template["schema"]
                st.json(schema)

        return schema

    def _render_variables_section(self) -> Dict[str, List[str]]:
        """Render the variables section UI"""
        variables = {}

        with st.expander("Variables - Cycle Through Values"):
            st.caption("Define values to cycle through for each row")

            with st.form("add_var_form", clear_on_submit=True):
                vc1, vc2, vc3 = st.columns([1, 3, 1])
                with vc1:
                    new_var_name = st.text_input("Variable", placeholder="topic")
                with vc2:
                    new_var_values = st.text_input("Values (comma-separated)", placeholder="sports, tech, health")
                with vc3:
                    st.write("")
                    add_var = st.form_submit_button("Add", use_container_width=True)

                if add_var and new_var_name and new_var_values:
                    st.session_state.gen_variables.append({
                        "name": new_var_name,
                        "values": [v.strip() for v in new_var_values.split(",")]
                    })
                    st.rerun()

            if st.session_state.gen_variables:
                for idx, var in enumerate(st.session_state.gen_variables):
                    vc1, vc2, vc3 = st.columns([1, 3, 1])
                    with vc1:
                        st.text(f"{{{var['name']}}}")
                    with vc2:
                        st.text(", ".join(var["values"]))
                    with vc3:
                        if st.button("Remove", key=f"del_var_{idx}"):
                            st.session_state.gen_variables.pop(idx)
                            st.rerun()
                    variables[var["name"]] = var["values"]

        return variables

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute the generate operation"""

        is_test = config.get("is_test", False)
        num_rows = config["num_rows"]
        target_count = min(10, num_rows) if is_test else num_rows
        run_type = "test" if is_test else "full"

        # Create or get session
        session_id = get_current_session_id()
        if not session_id:
            session = self.db.create_session("generate", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)

        # Create run
        run = self.db.create_run(
            session_id=session_id,
            run_type=run_type,
            provider=config["provider"].value,
            model=config["model"],
            temperature=config["gen_temperature"],
            max_tokens=config["max_tokens"],
            system_prompt=config["generation_prompt"],
            schema=config["schema"],
            variables=config["variables"],
            input_file="generated",
            input_rows=target_count,
            json_mode=True,
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retry_attempts=config["max_retries"],
            run_settings=get_current_settings()
        )

        self.db.log(LogLevel.INFO, f"Started generation run: {target_count} rows",
                   {"provider": config["provider"].value, "model": config["model"], "schema": config["schema"]},
                   run_id=run.run_id)

        # Create processor
        processing_config = ProcessingConfig(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=config["gen_temperature"],
            max_tokens=config["max_tokens"],
            json_mode=True,  # Generation always uses JSON
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retries=config["max_retries"],
            save_path=None,
            realtime_progress=True
        )

        processor = GenerateProcessor(processing_config, run.run_id, session_id)

        try:
            result = await processor.generate(
                target_count,
                config["generation_prompt"],
                config["schema"],
                config["variables"],
                config["use_freeform"],
                progress_callback
            )

            # Update run status
            self.db.update_run_status(
                run.run_id, RunStatus.COMPLETED,
                success_count=result.success_count,
                error_count=result.error_count,
                retry_count=result.retry_count,
                avg_latency=result.avg_latency,
                total_duration=result.total_duration
            )

            # Build DataFrame from results
            rows = []
            latencies = []
            schema = config["schema"]

            for item in result.results:
                data = item.get("data", {})
                success = item.get("success", False)
                latency = item.get("latency", 0)

                if success and not data.get("error") and not data.get("raw_output"):
                    rows.append(data)
                else:
                    if schema:
                        error_row = {k: None for k in schema.keys()}
                    else:
                        error_row = {}
                    error_row["_error"] = data.get("error") or data.get("raw_output") or "Unknown error"
                    rows.append(error_row)
                latencies.append(latency)

            generated_df = pd.DataFrame(rows)
            generated_df["_latency_s"] = latencies

            return ToolResult(
                success=True,
                data=generated_df,
                stats={
                    "generated": len(generated_df),
                    "success": result.success_count,
                    "errors": result.error_count,
                    "retries": result.retry_count,
                    "avg_latency": f"{result.avg_latency:.2f}s"
                }
            )

        except Exception as e:
            self.db.update_run_status(run.run_id, RunStatus.FAILED)
            self.db.log(LogLevel.ERROR, f"Generation failed: {str(e)}", run_id=run.run_id)
            return ToolResult(success=False, error_message=str(e))

    def render_results(self, result: ToolResult):
        """Render generation results"""
        import streamlit as st
        from ui.components.result_inspector import render_result_inspector, render_error_summary
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Generation failed: {result.error_message}")
            return

        # Success message with stats
        stats = result.stats
        success_rate = stats.get("success", 0) / stats.get("generated", 1) * 100

        st.success(
            f"Generated {stats.get('generated', 0)} rows | "
            f" {success_rate:.1f}% | "
            f" {stats.get('errors', 0)} | "
            f" {stats.get('retries', 0)} retries | "
            f"Avg: {stats.get('avg_latency', '0s')}"
        )

        # Data preview
        st.dataframe(result.data, use_container_width=True)

        # Error summary
        render_error_summary(result.data)

        # Download buttons
        st.divider()
        render_download_buttons(result.data, filename_prefix="handai_generated")
