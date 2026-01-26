"""
Transform Tool
Tool for transforming existing datasets with AI
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.processing import TransformProcessor, ProcessingConfig
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model,
    get_current_settings, set_current_session_id, get_current_session_id
)
from config import SAMPLE_DATA_COLUMNS


class TransformTool(BaseTool):
    """Tool for transforming existing datasets with AI"""

    id = "transform"
    name = "Transform Data"
    description = "Upload a dataset and transform each row using AI"
    icon = ""
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render transform tool configuration UI"""

        # File upload section
        st.header("1. Upload Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls", "json"],
                key="transform_upload"
            )
        with col_sample:
            st.write("")
            use_sample = st.button("Use Sample Data", help="Load sample data for testing")

        if use_sample:
            st.session_state["use_sample_data"] = True

        # Load data
        df = None
        data_source = None

        if uploaded_file:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            try:
                if file_ext == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_ext in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file)
                elif file_ext == "json":
                    df = pd.read_json(uploaded_file)
                data_source = uploaded_file.name
            except Exception as e:
                return ToolConfig(is_valid=False, error_message=f"Error loading file: {str(e)}")

        elif st.session_state.get("use_sample_data"):
            df = pd.DataFrame(SAMPLE_DATA_COLUMNS)
            data_source = "sample_data.csv"
            st.success("Using sample data (10 product reviews)")

        if df is None:
            return ToolConfig(is_valid=False, error_message="Please upload a file or use sample data")

        if df.empty:
            return ToolConfig(is_valid=False, error_message="The uploaded file is empty")

        # Column selection
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Active Columns (sent to AI)",
            all_cols,
            default=all_cols,
            key="transform_selected_cols"
        )

        st.dataframe(df.head(), height=200)

        # Prompt section
        st.header("2. Define Transformation")
        system_prompt = st.text_area(
            "AI Instructions",
            height=250,
            placeholder="Example: Extract the main topic from each text and classify sentiment as positive/negative/neutral...",
            key="transform_system_prompt",
            value=st.session_state.get("system_prompt", "")
        )

        if st.session_state.get("json_mode", False):
            st.info("JSON Mode enabled - instruct the AI to return valid JSON in your prompt")

        # Validate
        if not system_prompt or not system_prompt.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please enter AI instructions",
                config_data={
                    "df": df,
                    "data_source": data_source,
                    "selected_cols": selected_cols
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
                    "df": df,
                    "data_source": data_source,
                    "selected_cols": selected_cols,
                    "system_prompt": system_prompt
                }
            )

        # Build config
        config_data = {
            "df": df,
            "data_source": data_source,
            "selected_cols": selected_cols,
            "system_prompt": system_prompt,
            "provider": provider,
            "api_key": api_key,
            "base_url": st.session_state.get("base_url"),
            "model": model,
            "temperature": st.session_state.get("temperature", 0.0),
            "max_tokens": st.session_state.get("max_tokens", 2048),
            "json_mode": st.session_state.get("json_mode", False),
            "max_concurrency": st.session_state.get("max_concurrency", 5),
            "auto_retry": st.session_state.get("auto_retry", True),
            "max_retries": st.session_state.get("max_retries", 3),
            "realtime_progress": st.session_state.get("realtime_progress", True),
            "save_path": st.session_state.get("save_path", ""),
            "test_batch_size": st.session_state.get("test_batch_size", 10),
        }

        return ToolConfig(is_valid=True, config_data=config_data)

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute the transform operation"""

        df = config["df"]
        is_test = config.get("is_test", False)
        test_batch_size = config.get("test_batch_size", 10)

        # Prepare target dataframe
        if is_test:
            target_df = df.head(test_batch_size).copy()
        else:
            target_df = df.copy()
        target_df = target_df.reset_index(drop=True)

        run_type = "test" if is_test else "full"

        # Create or get session
        session_id = get_current_session_id()
        if not session_id:
            session = self.db.create_session("transform", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)
            self.db.log(LogLevel.INFO, f"Created new session: {session.name}",
                       session_id=session_id)

        # Create run
        run = self.db.create_run(
            session_id=session_id,
            run_type=run_type,
            provider=config["provider"].value,
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            system_prompt=config["system_prompt"],
            schema={},
            variables={},
            input_file=config["data_source"],
            input_rows=len(target_df),
            json_mode=config["json_mode"],
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retry_attempts=config["max_retries"],
            run_settings=get_current_settings()
        )

        self.db.log(LogLevel.INFO, f"Started {run_type} run with {len(target_df)} rows",
                   {"provider": config["provider"].value, "model": config["model"]},
                   run_id=run.run_id, session_id=session_id)

        # Create processor
        processing_config = ProcessingConfig(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            json_mode=config["json_mode"],
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retries=config["max_retries"],
            save_path=config["save_path"],
            realtime_progress=config.get("realtime_progress", True) if not is_test else True
        )

        processor = TransformProcessor(processing_config, run.run_id, session_id)

        # Run processing
        try:
            result = await processor.process(
                target_df,
                config["system_prompt"],
                config["selected_cols"],
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

            self.db.log(LogLevel.INFO, f"Run completed: {result.success_count} success, {result.error_count} errors",
                       {"duration": result.total_duration, "avg_latency": result.avg_latency},
                       run_id=run.run_id)

            # Add results to dataframe
            target_df["ai_output"] = [r.get("output", "N/A") for r in result.results]
            target_df["latency_s"] = [r.get("latency", 0) for r in result.results]

            return ToolResult(
                success=True,
                data=target_df,
                stats={
                    "success": result.success_count,
                    "errors": result.error_count,
                    "retries": result.retry_count,
                    "avg_latency": f"{result.avg_latency:.2f}s",
                    "duration": f"{result.total_duration:.1f}s"
                }
            )

        except Exception as e:
            self.db.update_run_status(run.run_id, RunStatus.FAILED)
            self.db.log(LogLevel.ERROR, f"Run failed: {str(e)}", run_id=run.run_id)
            return ToolResult(success=False, error_message=str(e))

    def render_results(self, result: ToolResult):
        """Render transform results"""
        import streamlit as st
        from ui.components.result_inspector import render_result_inspector
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Transformation failed: {result.error_message}")
            return

        # Success message with stats
        stats = result.stats
        st.success(
            f"Complete! "
            f" {stats.get('success', 0)} | "
            f" {stats.get('errors', 0)} | "
            f" {stats.get('retries', 0)} retries | "
            f"Avg: {stats.get('avg_latency', '0s')} | "
            f"Total: {stats.get('duration', '0s')}"
        )

        # Data preview
        st.dataframe(result.data, use_container_width=True)

        # Result inspector
        st.divider()
        render_result_inspector(result.data)

        # Download buttons
        st.divider()
        render_download_buttons(result.data)
