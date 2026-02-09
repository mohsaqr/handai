"""
Model Comparison Tool
Run same task across multiple LLM models and compare outputs
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, Any, Optional, Callable, List

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider
from core.comparison_processor import ComparisonProcessor, ComparisonConfig, ModelConfig
from core.comparison_analytics import calculate_all_agreement_metrics, render_agreement_metrics
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_current_settings, set_current_session_id, get_current_session_id
)
from core.sample_data import get_sample_data, get_dataset_info


def _get_enabled_providers() -> List[Dict[str, Any]]:
    """Return list of enabled configured providers."""
    db = get_db()
    providers = db.get_enabled_configured_providers()
    if providers:
        return [
            {
                "display_name": p.display_name,
                "provider_type": p.provider_type,
                "api_key": p.api_key,
                "base_url": p.base_url,
                "default_model": p.default_model,
                "temperature": p.temperature,
                "max_tokens": p.max_tokens,
            }
            for p in providers
        ]
    return []


class ModelComparisonTool(BaseTool):
    """Tool for comparing outputs across multiple LLM models"""

    id = "model_comparison"
    name = "Model Comparison"
    description = "Run same task across multiple models and compare outputs"
    icon = ":material/compare:"
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render model comparison configuration UI"""

        providers = _get_enabled_providers()

        # Step 1: Upload Data
        st.header("1. Upload Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls", "json"],
                key="comparison_upload"
            )
        with col_sample:
            st.write("")
            use_sample = st.button(
                "Use Sample Data",
                help="Load sample data for testing",
                key="comparison_sample_btn"
            )

        if use_sample:
            st.session_state["comparison_use_sample"] = True

        # Sample data selector
        if st.session_state.get("comparison_use_sample"):
            sample_options = {
                "product_reviews": "Product Reviews (20 reviews with sentiment)",
                "research_abstracts": "Research Abstracts (15 papers for classification)",
                "support_tickets": "Support Tickets (20 customer issues)",
            }
            selected_sample = st.selectbox(
                "Choose sample dataset",
                options=list(sample_options.keys()),
                format_func=lambda x: sample_options[x],
                key="comparison_sample_choice"
            )

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

        elif st.session_state.get("comparison_use_sample"):
            selected = st.session_state.get("comparison_sample_choice", "product_reviews")
            df = pd.DataFrame(get_sample_data(selected))
            info = get_dataset_info()[selected]
            data_source = f"{selected}.csv"
            st.success(f"Using sample data: {info['name']} ({info['rows']} rows)")

        if df is None:
            return ToolConfig(is_valid=False, error_message="Please upload a file or use sample data")

        if df.empty:
            return ToolConfig(is_valid=False, error_message="The uploaded file is empty")

        all_cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Active Columns (sent to AI)",
            all_cols,
            default=all_cols,
            key="comparison_selected_cols"
        )

        st.dataframe(df.head(), height=200)

        # Step 2: Select Models
        st.header("2. Select Models")

        if not providers:
            st.warning("No providers configured. Please set up providers in the **LLM Providers** page.")
            return ToolConfig(
                is_valid=False,
                error_message="No providers configured",
                config_data={"df": df, "data_source": data_source, "selected_cols": selected_cols}
            )

        # Quick-select buttons
        col_quick1, col_quick2 = st.columns(2)
        with col_quick1:
            if st.button("Select All", key="comparison_select_all"):
                st.session_state["comparison_selected_providers"] = [p["display_name"] for p in providers]
        with col_quick2:
            if st.button("Clear Selection", key="comparison_clear_selection"):
                st.session_state["comparison_selected_providers"] = []

        # Multi-select for providers
        provider_names = [p["display_name"] for p in providers]
        default_selection = st.session_state.get("comparison_selected_providers", provider_names[:2] if len(provider_names) >= 2 else provider_names)

        # Filter default selection to only include valid options
        default_selection = [name for name in default_selection if name in provider_names]

        selected_provider_names = st.multiselect(
            "Select models to compare",
            options=provider_names,
            default=default_selection,
            key="comparison_provider_multiselect",
            help="Select 2 or more models to compare their outputs"
        )

        # Update session state
        st.session_state["comparison_selected_providers"] = selected_provider_names

        if len(selected_provider_names) < 2:
            st.warning("Please select at least 2 models for comparison")
            return ToolConfig(
                is_valid=False,
                error_message="Select at least 2 models",
                config_data={"df": df, "data_source": data_source, "selected_cols": selected_cols}
            )

        # Show selected models with their details
        st.markdown("**Selected Models:**")
        selected_providers = [p for p in providers if p["display_name"] in selected_provider_names]

        cols = st.columns(min(len(selected_providers), 4))
        for i, provider in enumerate(selected_providers):
            with cols[i % 4]:
                st.markdown(f"**{provider['display_name']}**")
                st.caption(f"Model: {provider['default_model']}")

        # Step 3: Define Task
        st.header("3. Define Task")

        system_prompt = st.text_area(
            "AI Instructions (System Prompt)",
            height=200,
            value=st.session_state.get("comparison_system_prompt", """Analyze the provided data and classify it.

Respond with ONLY the classification result, no explanations.
Examples of valid responses: "positive", "negative", "neutral"

Be consistent and precise in your classification."""),
            key="comparison_prompt_input",
            help="This prompt will be sent to all selected models"
        )

        # Store in session state
        st.session_state["comparison_system_prompt"] = system_prompt

        if not system_prompt or not system_prompt.strip():
            return ToolConfig(
                is_valid=False,
                error_message="Please enter AI instructions",
                config_data={"df": df, "data_source": data_source, "selected_cols": selected_cols}
            )

        # Step 4: Options
        st.header("4. Options")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_agreement = st.checkbox(
                "Show agreement metrics",
                value=True,
                key="comparison_show_agreement",
                help="Calculate pairwise agreement between models"
            )
        with col_opt2:
            json_mode = st.checkbox(
                "JSON mode",
                value=False,
                key="comparison_json_mode",
                help="Force JSON output (where supported)"
            )

        # Build model configs
        model_configs = []
        for provider in selected_providers:
            try:
                provider_enum = LLMProvider(provider["provider_type"])
            except ValueError:
                provider_enum = LLMProvider.CUSTOM

            model_configs.append(ModelConfig(
                provider=provider_enum,
                api_key=provider["api_key"] or "dummy",
                base_url=provider["base_url"],
                model=provider["default_model"],
                display_name=provider["display_name"],
                temperature=provider.get("temperature"),
                max_tokens=provider.get("max_tokens"),
            ))

        config_data = {
            "df": df,
            "data_source": data_source,
            "selected_cols": selected_cols,
            "model_configs": model_configs,
            "system_prompt": system_prompt,
            "show_agreement": show_agreement,
            "json_mode": json_mode,
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
        """Execute the model comparison operation"""

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
            session = self.db.create_session("model_comparison", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)
            self.db.log(LogLevel.INFO, f"Created new session: {session.name}",
                       session_id=session_id)

        # Use first model's provider for the run record
        first_model = config["model_configs"][0]
        model_names = [mc.display_name for mc in config["model_configs"]]

        run = self.db.create_run(
            session_id=session_id,
            run_type=run_type,
            provider=first_model.provider.value,
            model=first_model.model,
            temperature=first_model.temperature,
            max_tokens=first_model.max_tokens,
            system_prompt=config["system_prompt"],
            schema={"comparison_models": model_names},
            variables={},
            input_file=config["data_source"],
            input_rows=len(target_df),
            json_mode=config["json_mode"],
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retry_attempts=config["max_retries"],
            run_settings=get_current_settings()
        )

        self.db.log(
            LogLevel.INFO,
            f"Started comparison {run_type} run with {len(target_df)} rows across {len(model_names)} models",
            {"models": model_names},
            run_id=run.run_id,
            session_id=session_id
        )

        comparison_config = ComparisonConfig(
            models=config["model_configs"],
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retries=config["max_retries"],
            save_path=config["save_path"],
            realtime_progress=config.get("realtime_progress", True) if not is_test else True,
            json_mode=config["json_mode"],
        )

        processor = ComparisonProcessor(comparison_config, run.run_id, session_id)

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

            self.db.log(
                LogLevel.INFO,
                f"Run completed: {result.success_count} success, {result.error_count} errors",
                {"duration": result.total_duration, "avg_latency": result.avg_latency},
                run_id=run.run_id
            )

            # Build result DataFrame
            for res_dict in result.results:
                for col, val in res_dict.items():
                    if col not in target_df.columns:
                        target_df[col] = None

            for i, res_dict in enumerate(result.results):
                for col, val in res_dict.items():
                    target_df.at[i, col] = val

            return ToolResult(
                success=True,
                data=target_df,
                stats={
                    "success": result.success_count,
                    "errors": result.error_count,
                    "retries": result.retry_count,
                    "avg_latency": f"{result.avg_latency:.2f}s",
                    "duration": f"{result.total_duration:.1f}s",
                    "num_models": len(config["model_configs"]),
                    "per_model_stats": result.per_model_stats,
                    "show_agreement": config["show_agreement"],
                    "model_configs": config["model_configs"],
                }
            )

        except Exception as e:
            self.db.update_run_status(run.run_id, RunStatus.FAILED)
            self.db.log(LogLevel.ERROR, f"Run failed: {str(e)}", run_id=run.run_id)
            return ToolResult(success=False, error_message=str(e))

    def render_results(self, result: ToolResult):
        """Render comparison results with optional agreement metrics"""
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Model comparison failed: {result.error_message}")
            return

        stats = result.stats

        # Summary stats
        st.success(
            f"Complete! "
            f"{stats.get('success', 0)} rows | "
            f"{stats.get('errors', 0)} errors | "
            f"{stats.get('retries', 0)} retries | "
            f"Avg: {stats.get('avg_latency', '0s')} | "
            f"Total: {stats.get('duration', '0s')}"
        )

        # Per-model stats
        per_model_stats = stats.get("per_model_stats", {})
        if per_model_stats:
            st.subheader("Per-Model Statistics")
            cols = st.columns(min(len(per_model_stats), 4))
            for i, (model_name, model_stats) in enumerate(per_model_stats.items()):
                with cols[i % 4]:
                    success_rate = (model_stats["success"] / (model_stats["success"] + model_stats["errors"]) * 100) if (model_stats["success"] + model_stats["errors"]) > 0 else 0
                    st.metric(
                        model_name,
                        f"{success_rate:.0f}% success",
                        f"Avg: {model_stats.get('avg_latency', 0):.2f}s"
                    )

        # Results DataFrame
        st.subheader("Comparison Results")
        st.dataframe(result.data, use_container_width=True)

        # Agreement Metrics
        if stats.get("show_agreement", False) and stats.get("num_models", 0) >= 2:
            st.divider()
            self._render_agreement_metrics(result.data, stats.get("model_configs", []))

        # Download buttons
        st.divider()
        render_download_buttons(result.data, filename_prefix="model_comparison_results",
                                key_prefix="comparison_download")

    def _render_agreement_metrics(self, df: pd.DataFrame, model_configs: List[ModelConfig]):
        """Render agreement metrics for the comparison results"""

        # Extract output columns for each model
        outputs: Dict[str, List[str]] = {}

        for mc in model_configs:
            # Match the column naming from the processor
            safe_name = self._sanitize_column_name(mc.display_name)
            output_col = f"{safe_name}_output"

            if output_col in df.columns:
                outputs[mc.display_name] = df[output_col].fillna("").astype(str).tolist()

        if len(outputs) < 2:
            st.info("Not enough model outputs for agreement analysis")
            return

        # Calculate and render metrics
        metrics = calculate_all_agreement_metrics(outputs)
        render_agreement_metrics(metrics)

    def _sanitize_column_name(self, name: str) -> str:
        """Convert display name to valid column name (must match processor)"""
        safe = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        safe = "".join(c for c in safe if c.isalnum() or c == "_")
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_")
