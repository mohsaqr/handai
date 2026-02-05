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
from core.prompt_registry import get_effective_prompt, ensure_prompts_registered
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model,
    get_current_settings, set_current_session_id, get_current_session_id
)
from config import VARIATION_TEMPS, TYPE_MAP


# Output format descriptions
OUTPUT_FORMATS = {
    "Tabular (CSV)": {
        "description": "Structured rows and columns - best for spreadsheets and data analysis",
        "recommended": True
    },
    "JSON": {
        "description": "Structured key-value pairs - best for APIs and programming",
        "recommended": False
    },
    "Free Text": {
        "description": "Unstructured text - best for creative content and narratives",
        "recommended": False
    }
}


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

        # Initialize session state for dynamic fields
        if "custom_fields" not in st.session_state:
            st.session_state.custom_fields = []
        if "column_suggestions" not in st.session_state:
            st.session_state.column_suggestions = ""

        # Section 1: Prompt (Primary Input - First)
        st.subheader("What would you like to generate?")
        generation_prompt = st.text_area(
            "Describe your data",
            height=120,
            placeholder="Example: Generate realistic customer profiles including full names, email addresses, and purchase history...",
            key="generate_prompt",
            help="Describe the data you need. Be specific about the content and any constraints.",
            label_visibility="collapsed"
        )

        st.divider()

        # Section 2: Output Format + Structure (Side by Side)
        col_format, col_structure = st.columns(2)

        with col_format:
            st.markdown("**Output Format**")
            output_format = st.radio(
                "Output Format",
                list(OUTPUT_FORMATS.keys()),
                index=0,  # Tabular is default
                key="gen_output_format",
                help="Choose how your generated data should be structured",
                label_visibility="collapsed"
            )
            # Show format description
            format_info = OUTPUT_FORMATS[output_format]
            st.caption(format_info['description'])

        with col_structure:
            st.markdown("**Structure**")
            if output_format == "Free Text":
                st.info("Free text mode - AI generates unstructured content.")
                schema_mode = "Let AI decide"
            else:
                schema_mode = st.radio(
                    "Structure",
                    ["Let AI decide", "Define columns", "Use Template"],
                    key="gen_schema_mode",
                    label_visibility="collapsed"
                )

        # Structure details (conditional on schema_mode)
        schema = {}
        use_freeform = False
        csv_columns = ""

        if output_format == "Free Text" or schema_mode == "Let AI decide":
            use_freeform = True
        elif schema_mode == "Define columns":
            if output_format == "Tabular (CSV)":
                # Render column suggestions above the column input
                self._render_column_suggestions(generation_prompt)
                csv_columns = self._render_column_input()
                if csv_columns:
                    schema = {col.strip(): "string" for col in csv_columns.split(",")}
            else:
                schema = self._render_schema_builder()
        elif schema_mode == "Use Template":
            schema = self._render_template_selector()
            if schema and output_format == "Tabular (CSV)":
                csv_columns = ",".join(schema.keys())
                # Auto-fill suggestions from template schema keys
                st.session_state.column_suggestions = csv_columns

        st.divider()

        # Section 3: Generation Settings
        st.subheader("Generation Settings")
        col_rows, col_variation = st.columns(2)

        with col_rows:
            num_rows = st.number_input("Rows to Generate", 1, 10000, 100, key="gen_num_rows")

        with col_variation:
            variation_level = st.select_slider(
                "Variation Level",
                options=["Low", "Medium", "High", "Maximum"],
                value="Medium",
                key="gen_variation",
                help="Higher variation produces more diverse outputs"
            )

        gen_temperature = VARIATION_TEMPS[variation_level]

        # Variables disabled - pass empty dict
        variables = {}

        # Validate
        if not generation_prompt or not generation_prompt.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please describe what data you want to generate",
                config_data={
                    "num_rows": num_rows,
                    "output_format": output_format,
                    "schema": schema,
                    "use_freeform": use_freeform,
                    "csv_columns": csv_columns,
                    "variables": variables
                }
            )

        # For tabular format with defined columns, validate columns exist
        if output_format == "Tabular (CSV)" and not use_freeform and not csv_columns:
            return ToolConfig(
                is_valid=False,
                error_message="Please define column names for tabular output",
                config_data={
                    "generation_prompt": generation_prompt,
                    "num_rows": num_rows,
                    "output_format": output_format,
                    "schema": schema,
                    "use_freeform": use_freeform,
                    "csv_columns": csv_columns,
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
                    "output_format": output_format,
                    "schema": schema,
                    "use_freeform": use_freeform,
                    "csv_columns": csv_columns,
                    "variables": variables
                }
            )

        # Build config
        config_data = {
            "generation_prompt": generation_prompt,
            "num_rows": num_rows,
            "output_format": output_format,
            "schema": schema,
            "use_freeform": use_freeform,
            "csv_columns": csv_columns,
            "suggested_columns": st.session_state.get("column_suggestions", ""),
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

    def _render_column_suggestions(self, generation_prompt: str) -> None:
        """Render column suggestions UI with AI suggestion capability"""
        st.write("**Suggested Columns**")

        # Editable text area for suggestions
        suggestions = st.text_area(
            "Column suggestions (comma-separated)",
            value=st.session_state.column_suggestions,
            placeholder="name, email, age, city, purchase_amount",
            key="gen_column_suggestions_input",
            height=68,
            help="Edit suggestions or use the buttons below to auto-generate",
            label_visibility="collapsed"
        )

        # Update session state when user edits
        if suggestions != st.session_state.column_suggestions:
            st.session_state.column_suggestions = suggestions

        # Buttons row
        col_suggest, col_apply = st.columns(2)

        with col_suggest:
            if st.button("Suggest from Prompt", key="btn_suggest_columns", use_container_width=True):
                if generation_prompt and generation_prompt.strip():
                    with st.spinner("Analyzing prompt..."):
                        suggested = self._get_ai_column_suggestions(generation_prompt)
                        if suggested:
                            st.session_state.column_suggestions = suggested
                            st.rerun()
                else:
                    st.warning("Please enter a generation prompt first")

        with col_apply:
            if st.button("Apply to Columns", key="btn_apply_suggestions", use_container_width=True):
                if st.session_state.column_suggestions:
                    st.session_state.gen_csv_columns = st.session_state.column_suggestions
                    st.rerun()

    def _get_ai_column_suggestions(self, generation_prompt: str) -> str:
        """Get AI-suggested column names based on the generation prompt"""
        from core.llm_client import get_client, create_http_client
        from core.providers import is_local_provider
        from database import get_db
        import asyncio

        provider = get_selected_provider()
        model = get_selected_model()
        base_url = st.session_state.get("base_url")

        # Get API key and base_url - try configured providers first
        api_key = None
        provider_name = st.session_state.get("selected_provider", "OpenAI")
        db = get_db()
        configured_providers = db.get_enabled_configured_providers()

        for p in configured_providers:
            if p.display_name == provider_name or p.provider_type == provider_name:
                if p.api_key:
                    api_key = p.api_key
                if p.base_url:
                    base_url = p.base_url
                if p.default_model:
                    model = p.default_model
                break

        # Fallback to get_effective_api_key
        if not api_key:
            api_key = get_effective_api_key()

        # Local providers (LM Studio, Ollama) don't need real API keys
        is_local = is_local_provider(provider)
        if is_local and (not api_key or api_key == "dummy"):
            api_key = "dummy"  # Use dummy key for local providers

        if not api_key:
            st.error(f"API key required for AI suggestions. Please configure in LLM Providers page.")
            return ""

        # Get effective system prompt (respects overrides from Settings > System Prompts)
        ensure_prompts_registered()
        system_prompt = get_effective_prompt("generate.column_suggestions")

        user_prompt = f"Suggest column names for this data: {generation_prompt}"

        try:
            http_client = create_http_client()
            client = get_client(provider, api_key, base_url, http_client)

            # Run async call using a new event loop to avoid conflicts
            async def get_suggestions():
                try:
                    if provider == LLMProvider.OPENAI:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=100,
                            temperature=0
                        )
                        return response.choices[0].message.content.strip()
                    elif provider == LLMProvider.ANTHROPIC:
                        response = await client.messages.create(
                            model=model,
                            max_tokens=100,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_prompt}]
                        )
                        return response.content[0].text.strip()
                    elif provider == LLMProvider.GOOGLE:
                        response = await client.models.generate_content_async(
                            model=model,
                            contents=f"{system_prompt}\n\n{user_prompt}"
                        )
                        return response.text.strip()
                    else:
                        # OpenAI-compatible providers
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=100,
                            temperature=0
                        )
                        return response.choices[0].message.content.strip()
                finally:
                    await http_client.aclose()

            # Use new event loop to avoid conflicts with existing loops
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(get_suggestions())
            finally:
                loop.close()
            return result
        except Exception as e:
            st.error(f"Failed to get suggestions: {str(e)}")
            return ""

    def _render_column_input(self) -> str:
        """Render simple column input for tabular format"""
        st.write("**Define Columns:**")
        csv_columns = st.text_input(
            "Column Names (comma-separated)",
            placeholder="name, email, age, city, purchase_amount",
            key="gen_csv_columns",
            help="Enter the column headers for your tabular data"
        )
        if csv_columns:
            cols = [c.strip() for c in csv_columns.split(",") if c.strip()]
            st.info(f"Columns: `{', '.join(cols)}`")
        return csv_columns

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
                progress_callback,
                output_format=config.get("output_format", "JSON"),
                csv_columns=config.get("csv_columns", ""),
                suggested_columns=config.get("suggested_columns", "")
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
