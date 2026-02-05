"""
Consensus Coder Tool
Multi-model consensus coding with inter-rater reliability analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, Callable, List, Tuple

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS
from core.processing import ConsensusProcessor, ConsensusConfig
from core.prompt_registry import get_effective_prompt, ensure_prompts_registered
from database import get_db, RunStatus, LogLevel
from ui.state import (
    get_current_settings, set_current_session_id, get_current_session_id
)

DEFAULT_WORKER_PROMPT = """Analyze the provided data and respond with ONLY the requested output values.

CRITICAL FORMAT REQUIREMENTS:
- Output MUST be in strict CSV format (comma-separated values)
- NO explanations, NO prose, NO markdown, NO code blocks
- NO headers or labels - just the raw values

Respond with ONLY the CSV-formatted data values. Nothing else."""

DEFAULT_JUDGE_PROMPT = """You are a judge synthesizing worker responses into a single best answer.

CRITICAL: Your best_answer MUST be in strict CSV/tabular format:
- Comma-separated values ONLY
- NO explanations, NO prose, NO markdown
- NO headers - just the data values

If workers disagree, choose the most accurate/complete values and format as CSV."""


def _get_enabled_providers():
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
            }
            for p in providers
        ]
    return []


def _render_model_selector(label: str, key_prefix: str, providers: list):
    """Render a provider/model selector for a worker or judge."""
    names = [p["display_name"] for p in providers]
    selected_name = st.selectbox(f"{label} Provider", names, key=f"{key_prefix}_provider")
    entry = providers[names.index(selected_name)]
    model = st.text_input(f"{label} Model", value=entry["default_model"], key=f"{key_prefix}_model")

    try:
        provider_enum = LLMProvider(entry["provider_type"])
    except ValueError:
        provider_enum = LLMProvider.CUSTOM

    return {
        "provider_enum": provider_enum,
        "api_key": entry["api_key"] or "dummy",
        "base_url": entry["base_url"],
        "model": model,
    }


class ConsensusTool(BaseTool):
    """Tool for multi-model consensus coding with inter-rater reliability"""

    id = "consensus"
    name = "Consensus Coder"
    description = "Multi-model consensus with inter-rater reliability"
    icon = ":material/groups:"
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render consensus coder configuration UI"""

        providers = _get_enabled_providers()

        # File upload
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=["csv", "xlsx", "xls", "json"],
            key="consensus_upload"
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

        if df is None:
            return ToolConfig(is_valid=False, error_message="Please upload a file")

        if df.empty:
            return ToolConfig(is_valid=False, error_message="The uploaded file is empty")

        all_cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Active Columns (sent to AI)",
            all_cols,
            default=all_cols,
            key="consensus_selected_cols"
        )

        st.dataframe(df.head(), height=200)

        # Configure models
        st.header("2. Configure Models")

        if not providers:
            st.warning("No providers configured. Please set up providers in the **LLM Providers** page.")
            return ToolConfig(is_valid=False, error_message="No providers configured",
                            config_data={"df": df, "data_source": data_source, "selected_cols": selected_cols})

        with st.expander("Workers", expanded=True):
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                st.subheader("Worker 1")
                worker1 = _render_model_selector("Worker 1", "cw1", providers)
            with w_col2:
                st.subheader("Worker 2")
                worker2 = _render_model_selector("Worker 2", "cw2", providers)

            use_worker3 = st.checkbox("Enable Worker 3", value=False, key="consensus_use_w3")
            worker3 = None
            if use_worker3:
                st.subheader("Worker 3")
                worker3 = _render_model_selector("Worker 3", "cw3", providers)

        with st.expander("Judge", expanded=True):
            judge = _render_model_selector("Judge", "cj", providers)

        # Prompts
        st.header("3. Prompts")

        # Get effective prompts (respects overrides from Settings > System Prompts)
        ensure_prompts_registered()
        default_worker_prompt = get_effective_prompt("consensus.worker_prompt")
        default_judge_prompt = get_effective_prompt("consensus.judge_prompt")

        col1, col2 = st.columns(2)
        with col1:
            worker_prompt = st.text_area(
                "Worker Instructions",
                height=200,
                key="consensus_worker_prompt",
                value=st.session_state.get("consensus_worker_prompt_val", default_worker_prompt),
            )
        with col2:
            judge_prompt = st.text_area(
                "Judge Instructions",
                height=200,
                key="consensus_judge_prompt",
                value=st.session_state.get("consensus_judge_prompt_val", default_judge_prompt),
            )

        include_reasoning = st.checkbox("Include Judge Reasoning", value=True, key="consensus_reasoning")

        if not worker_prompt or not worker_prompt.strip():
            return ToolConfig(is_valid=False, error_message="Please enter worker instructions",
                            config_data={"df": df, "data_source": data_source, "selected_cols": selected_cols})

        worker_configs = [worker1, worker2]
        if use_worker3 and worker3:
            worker_configs.append(worker3)

        config_data = {
            "df": df,
            "data_source": data_source,
            "selected_cols": selected_cols,
            "worker_configs": worker_configs,
            "judge_config": judge,
            "worker_prompt": worker_prompt,
            "judge_prompt": judge_prompt,
            "include_reasoning": include_reasoning,
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
        """Execute the consensus coding operation"""

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
            session = self.db.create_session("consensus", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)
            self.db.log(LogLevel.INFO, f"Created new session: {session.name}",
                       session_id=session_id)

        # Use first worker's provider for the run record
        first_worker = config["worker_configs"][0]
        run = self.db.create_run(
            session_id=session_id,
            run_type=run_type,
            provider=first_worker["provider_enum"].value,
            model=first_worker["model"],
            temperature=0.0,
            max_tokens=2048,
            system_prompt=config["worker_prompt"],
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

        self.db.log(LogLevel.INFO, f"Started consensus {run_type} run with {len(target_df)} rows",
                   {"workers": len(config["worker_configs"])},
                   run_id=run.run_id, session_id=session_id)

        consensus_config = ConsensusConfig(
            worker_configs=config["worker_configs"],
            judge_config=config["judge_config"],
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retries=config["max_retries"],
            save_path=config["save_path"],
            realtime_progress=config.get("realtime_progress", True) if not is_test else True,
            include_reasoning=config["include_reasoning"],
        )

        processor = ConsensusProcessor(consensus_config, run.run_id, session_id)

        try:
            result = await processor.process(
                target_df,
                config["worker_prompt"],
                config["judge_prompt"],
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

            # Build result DataFrame
            num_workers = len(config["worker_configs"])
            for i in range(num_workers):
                w_key = f"worker_{i+1}"
                target_df[f"{w_key}_output"] = [r.get(w_key, "N/A") for r in result.results]
                target_df[f"{w_key}_latency_s"] = [r.get(f"{w_key}_latency", 0) for r in result.results]

            target_df["judge_consensus"] = [r.get("judge_consensus", "N/A") for r in result.results]
            target_df["judge_best_answer"] = [r.get("judge_best_answer", "N/A") for r in result.results]
            if config["include_reasoning"]:
                target_df["judge_reasoning"] = [r.get("judge_reasoning", "N/A") for r in result.results]
            target_df["judge_latency_s"] = [r.get("judge_latency", 0) for r in result.results]

            return ToolResult(
                success=True,
                data=target_df,
                stats={
                    "success": result.success_count,
                    "errors": result.error_count,
                    "retries": result.retry_count,
                    "avg_latency": f"{result.avg_latency:.2f}s",
                    "duration": f"{result.total_duration:.1f}s",
                    "num_workers": num_workers,
                }
            )

        except Exception as e:
            self.db.update_run_status(run.run_id, RunStatus.FAILED)
            self.db.log(LogLevel.ERROR, f"Run failed: {str(e)}", run_id=run.run_id)
            return ToolResult(success=False, error_message=str(e))

    def render_results(self, result: ToolResult):
        """Render consensus results with inter-rater analytics"""
        from ui.components.result_inspector import render_result_inspector
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Consensus coding failed: {result.error_message}")
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

        # Inter-Rater Analytics
        st.divider()
        st.subheader("Inter-Rater Agreement Analysis")
        _render_analytics(result.data, stats.get("num_workers", 2))

        # Result inspector
        st.divider()
        render_result_inspector(result.data, output_column="judge_best_answer",
                                key_prefix="consensus_inspector")

        # Download buttons
        st.divider()
        render_download_buttons(result.data, filename_prefix="consensus_results",
                                key_prefix="consensus_download")


def _render_analytics(df: pd.DataFrame, num_workers: int):
    """Render inter-rater reliability analytics."""

    worker_cols = [col for col in df.columns if col.endswith("_output") and not col.startswith("judge")]

    if len(worker_cols) < 2:
        st.info("Need at least 2 workers for agreement statistics.")
        return

    worker_responses = {}
    for col in worker_cols:
        worker_name = col.replace("_output", "")
        worker_responses[worker_name] = df[col].fillna("").astype(str)

    # Pairwise Agreement
    st.markdown("### Pairwise Agreement")
    workers_list = list(worker_responses.keys())

    if len(workers_list) == 2:
        w1, w2 = workers_list
        exact_match = (worker_responses[w1] == worker_responses[w2]).mean() * 100
        col1, col2 = st.columns(2)
        col1.metric("Exact Agreement", f"{exact_match:.1f}%", help="Percentage of identical responses")

        try:
            from sklearn.metrics import cohen_kappa_score
            w1_cat = worker_responses[w1].str[:100]
            w2_cat = worker_responses[w2].str[:100]
            kappa = cohen_kappa_score(w1_cat, w2_cat)
            col2.metric("Cohen's Kappa", f"{kappa:.3f}", help="Agreement corrected for chance")
        except Exception:
            col2.metric("Cohen's Kappa", "N/A")
    else:
        agreements = []
        kappas = []
        for i in range(len(workers_list)):
            for j in range(i + 1, len(workers_list)):
                agree = (worker_responses[workers_list[i]] == worker_responses[workers_list[j]]).mean()
                agreements.append(agree)
                try:
                    from sklearn.metrics import cohen_kappa_score
                    k = cohen_kappa_score(
                        worker_responses[workers_list[i]].str[:100],
                        worker_responses[workers_list[j]].str[:100]
                    )
                    kappas.append(k)
                except Exception:
                    pass

        col1, col2 = st.columns(2)
        col1.metric("Avg Pairwise Agreement", f"{np.mean(agreements)*100:.1f}%")
        if kappas:
            col2.metric("Avg Cohen's Kappa", f"{np.mean(kappas):.3f}")

    # Judge alignment
    st.markdown("### Judge Alignment with Workers")
    if "judge_best_answer" in df.columns:
        judge_col = df["judge_best_answer"].fillna("").astype(str)

        judge_agreements = {}
        for worker_name, responses in worker_responses.items():
            fuzzy_match = sum(
                (str(j).lower() in str(w).lower()) or (str(w).lower() in str(j).lower())
                for j, w in zip(judge_col, responses)
            ) / len(judge_col) * 100
            judge_agreements[worker_name] = fuzzy_match

        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Bar(
                x=list(judge_agreements.keys()),
                y=list(judge_agreements.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(judge_agreements)],
                text=[f"{v:.1f}%" for v in judge_agreements.values()],
                textposition='auto',
            )])
            fig.update_layout(
                title="Judge Alignment with Workers",
                xaxis_title="Worker", yaxis_title="Match %",
                yaxis_range=[0, 100], height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for wn, pct in judge_agreements.items():
                st.metric(wn, f"{pct:.1f}%")

    # Consensus Distribution
    st.markdown("### Consensus Distribution")
    if "judge_consensus" in df.columns:
        consensus_counts = df["judge_consensus"].value_counts()
        try:
            import plotly.express as px
            fig2 = px.pie(
                values=consensus_counts.values,
                names=consensus_counts.index,
                title="Consensus Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            for val, count in consensus_counts.items():
                st.write(f"- {val}: {count}")

    # Latency stats
    st.markdown("### Latency Statistics")
    latency_cols_list = [col for col in df.columns if col.endswith("_latency_s")]
    if latency_cols_list:
        cols = st.columns(len(latency_cols_list))
        for i, col_name in enumerate(latency_cols_list):
            label = col_name.replace("_latency_s", "").replace("_", " ").title()
            avg_lat = df[col_name].mean()
            cols[i].metric(label, f"{avg_lat:.2f}s")

    # Jaccard Index
    st.markdown("### Position-Independent Agreement (Jaccard)")

    def extract_code_set(text):
        if pd.isna(text) or text == "":
            return set()
        codes = str(text).replace(",", " ").replace(";", " ").split()
        return set(c.strip().lower() for c in codes if c.strip())

    jaccard_scores = []
    for idx in range(len(df)):
        worker_sets = {wn: extract_code_set(responses.iloc[idx]) for wn, responses in worker_responses.items()}
        for i in range(len(workers_list)):
            for j in range(i + 1, len(workers_list)):
                s1, s2 = worker_sets[workers_list[i]], worker_sets[workers_list[j]]
                if len(s1) == 0 and len(s2) == 0:
                    jaccard = 1.0
                elif len(s1) == 0 or len(s2) == 0:
                    jaccard = 0.0
                else:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                jaccard_scores.append(jaccard)

    if jaccard_scores:
        jc1, jc2, jc3 = st.columns(3)
        jc1.metric("Avg Jaccard Index", f"{np.mean(jaccard_scores):.3f}", help="0=no overlap, 1=perfect")
        jc2.metric("Min", f"{min(jaccard_scores):.3f}")
        jc3.metric("Max", f"{max(jaccard_scores):.3f}")
