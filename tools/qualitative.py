"""
Qualitative Coder Tool
Code qualitative data (interviews, surveys, observations) with AI
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Callable

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.processing import TransformProcessor, ProcessingConfig
from core.prompt_registry import get_effective_prompt, ensure_prompts_registered
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model,
    get_current_settings, set_current_session_id, get_current_session_id
)
from config import SAMPLE_DATA_COLUMNS

DEFAULT_QUALITATIVE_PROMPT = """Analyze the provided data and respond with ONLY the requested output values.

CRITICAL FORMAT REQUIREMENTS:
- Output MUST be in strict CSV format (comma-separated values)
- NO explanations, NO prose, NO markdown, NO code blocks
- NO headers or labels - just the raw values

Respond with ONLY the CSV-formatted data values. Nothing else."""


def estimate_openai_cost(df, system_prompt, model, selected_cols):
    """Estimate OpenAI API cost for processing a dataframe."""
    try:
        import tiktoken
    except ImportError:
        return 0, 0.0

    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    prices = pricing.get(model, None)
    if not prices:
        return 0, 0.0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    subset = df[selected_cols] if selected_cols else df
    prompt_tokens = len(encoding.encode(system_prompt))
    sample = subset.head(100)
    data_tokens_sum = sum(len(encoding.encode(row.to_json())) for _, row in sample.iterrows())
    avg_row_tokens = data_tokens_sum / len(sample) if len(sample) > 0 else 0
    total_tokens = (prompt_tokens + avg_row_tokens) * len(df)
    cost = (total_tokens / 1_000_000) * prices["input"]
    return int(total_tokens), cost


class QualitativeTool(BaseTool):
    """Tool for qualitative coding of data with AI"""

    id = "qualitative"
    name = "Qualitative Coder"
    description = "Code qualitative data with AI"
    icon = ":material/psychology:"
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render qualitative coder configuration UI"""

        # File upload section
        st.header("1. Upload Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls", "json"],
                key="qualitative_upload"
            )
        with col_sample:
            st.write("")
            use_sample = st.button("Use Sample Data", help="Load sample data for testing",
                                   key="qualitative_sample_btn")

        if use_sample:
            st.session_state["qualitative_use_sample"] = True

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

        elif st.session_state.get("qualitative_use_sample"):
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
            key="qualitative_selected_cols"
        )

        st.dataframe(df.head(), height=200)

        # Prompt section
        st.header("2. Coding Instructions")

        # Get the effective prompt (respects overrides from Settings > System Prompts)
        ensure_prompts_registered()
        default_prompt = get_effective_prompt("qualitative.default_prompt")

        system_prompt = st.text_area(
            "AI Instructions",
            height=250,
            key="qualitative_system_prompt",
            value=st.session_state.get("qualitative_prompt_value", default_prompt),
            help="Pre-filled with a qualitative coding prompt. Edit to match your analysis needs."
        )

        # Cost estimation for OpenAI
        provider = get_selected_provider()
        api_key = get_effective_api_key()
        model = get_selected_model()

        if provider in (LLMProvider.OPENAI,) and system_prompt and selected_cols:
            tokens, cost = estimate_openai_cost(df, system_prompt, model, selected_cols)
            if cost > 0:
                st.caption(f"Estimated cost: ${cost:.4f} (~{tokens:,} tokens)")

        if not system_prompt or not system_prompt.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please enter coding instructions",
                config_data={
                    "df": df,
                    "data_source": data_source,
                    "selected_cols": selected_cols
                }
            )

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
            "json_mode": False,
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
        """Execute the qualitative coding operation"""

        df = config["df"]
        is_test = config.get("is_test", False)
        test_batch_size = config.get("test_batch_size", 10)

        if is_test:
            target_df = df.head(test_batch_size).copy()
        else:
            target_df = df.copy()
        target_df = target_df.reset_index(drop=True)

        run_type = "test" if is_test else "full"

        session_id = get_current_session_id()
        if not session_id:
            session = self.db.create_session("qualitative", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)
            self.db.log(LogLevel.INFO, f"Created new session: {session.name}",
                       session_id=session_id)

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
            json_mode=False,
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retry_attempts=config["max_retries"],
            run_settings=get_current_settings()
        )

        self.db.log(LogLevel.INFO, f"Started {run_type} run with {len(target_df)} rows",
                   {"provider": config["provider"].value, "model": config["model"]},
                   run_id=run.run_id, session_id=session_id)

        processing_config = ProcessingConfig(
            provider=config["provider"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            json_mode=False,
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retries=config["max_retries"],
            save_path=config["save_path"],
            realtime_progress=config.get("realtime_progress", True) if not is_test else True
        )

        processor = TransformProcessor(processing_config, run.run_id, session_id)

        try:
            result = await processor.process(
                target_df,
                config["system_prompt"],
                config["selected_cols"],
                progress_callback
            )

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
        """Render qualitative coding results"""
        from ui.components.result_inspector import render_result_inspector
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Qualitative coding failed: {result.error_message}")
            return

        stats = result.stats
        st.success(
            f"Complete! "
            f" {stats.get('success', 0)} | "
            f" {stats.get('errors', 0)} | "
            f" {stats.get('retries', 0)} retries | "
            f"Avg: {stats.get('avg_latency', '0s')} | "
            f"Total: {stats.get('duration', '0s')}"
        )

        st.dataframe(result.data, use_container_width=True)

        st.divider()
        render_result_inspector(result.data)

        st.divider()
        render_download_buttons(result.data, filename_prefix="qualitative_results")
