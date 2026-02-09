"""
Model Comparison Processor
Multi-model processing engine for comparing LLM outputs
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import pandas as pd

from .providers import LLMProvider, supports_json_mode
from .llm_client import get_client, create_http_client, call_llm_with_retry
from database import get_db, RunResult, ResultStatus, LogLevel


@dataclass
class ModelConfig:
    """Configuration for a single model in comparison"""
    provider: LLMProvider
    api_key: str
    base_url: Optional[str]
    model: str
    display_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class ComparisonConfig:
    """Configuration for a comparison processing run"""
    models: List[ModelConfig]
    max_concurrency: int = 5
    auto_retry: bool = True
    max_retries: int = 3
    save_path: Optional[str] = None
    realtime_progress: bool = True
    json_mode: bool = False


@dataclass
class ComparisonResult:
    """Result of a comparison processing run"""
    success_count: int
    error_count: int
    retry_count: int
    avg_latency: float
    total_duration: float
    results: List[Dict[str, Any]]
    per_model_stats: Dict[str, Dict[str, Any]]


class ComparisonProcessor:
    """Processor for running same task across multiple LLM models"""

    def __init__(self, config: ComparisonConfig, run_id: str, session_id: str):
        self.config = config
        self.run_id = run_id
        self.session_id = session_id
        self.db = get_db()

    async def process(
        self,
        df: pd.DataFrame,
        system_prompt: str,
        selected_cols: List[str],
        progress_callback: Optional[Callable] = None
    ) -> ComparisonResult:
        """
        Process a dataframe with multiple LLM models in parallel.

        Args:
            df: Input dataframe
            system_prompt: System prompt for all models
            selected_cols: Columns to include in the prompt
            progress_callback: Optional callback for progress updates
                              signature: (completed, total, success, errors, retries, log_entry, is_error)

        Returns:
            ComparisonResult with statistics and results
        """
        http_client = create_http_client()

        # Create clients for each model
        model_clients: Dict[str, Tuple[Any, ModelConfig]] = {}
        for mc in self.config.models:
            client = get_client(mc.provider, mc.api_key, mc.base_url, http_client)
            model_clients[mc.display_name] = (client, mc)

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        results_map: Dict[int, Dict[str, Any]] = {}
        all_db_results: List[RunResult] = []

        # Per-model statistics
        per_model_stats = {
            mc.display_name: {"success": 0, "errors": 0, "total_latency": 0.0}
            for mc in self.config.models
        }

        total = len(df)
        completed = 0
        success_count = 0
        error_count = 0
        retry_count = 0

        async def call_single_model(
            client: Any,
            mc: ModelConfig,
            user_content: str,
            row_idx: int
        ) -> Tuple[str, Optional[str], float, int]:
            """Call a single model and return (display_name, output, latency, retries)"""
            effective_json_mode = self.config.json_mode and supports_json_mode(
                mc.provider, mc.model
            )

            output, duration, error_info, attempts = await call_llm_with_retry(
                client, system_prompt, user_content, mc.model,
                mc.temperature, mc.max_tokens, effective_json_mode,
                self.run_id, row_idx,
                self.config.max_retries if self.config.auto_retry else 0,
                self.db,
                mc.provider
            )

            if error_info:
                return mc.display_name, f"Error: {error_info.message}", duration, attempts
            return mc.display_name, output, duration, attempts

        async def process_row(row_idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any], bool]:
            async with semaphore:
                # Build row data
                if selected_cols:
                    row_data = row[selected_cols].to_json()
                else:
                    row_data = row.to_json()

                user_content = f"Data: {row_data}"

                # Call all models in parallel for this row
                tasks = [
                    call_single_model(client, mc, user_content, row_idx)
                    for display_name, (client, mc) in model_clients.items()
                ]

                model_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Build result dict for this row
                row_result: Dict[str, Any] = {}
                row_has_error = False
                row_retries = 0

                for result in model_results:
                    if isinstance(result, Exception):
                        # Handle unexpected exceptions
                        model_name = "unknown"
                        row_result[f"{model_name}_output"] = f"Error: {str(result)}"
                        row_result[f"{model_name}_latency"] = 0.0
                        row_has_error = True
                    else:
                        display_name, output, latency, attempts = result
                        # Sanitize display name for column naming
                        safe_name = self._sanitize_column_name(display_name)
                        row_result[f"{safe_name}_output"] = output
                        row_result[f"{safe_name}_latency"] = round(latency, 3)

                        # Track per-model stats
                        if output and not output.startswith("Error:"):
                            per_model_stats[display_name]["success"] += 1
                            per_model_stats[display_name]["total_latency"] += latency
                        else:
                            per_model_stats[display_name]["errors"] += 1
                            row_has_error = True

                        row_retries += attempts

                return row_idx, row_result, row_has_error, row_retries

        # Create tasks for all rows
        tasks = [process_row(i, row) for i, row in df.iterrows()]
        start_time = time.time()

        # Process with progress updates
        for future in asyncio.as_completed(tasks):
            idx, res_dict, has_error, row_retries = await future
            results_map[idx] = res_dict
            completed += 1
            retry_count += row_retries

            if has_error:
                error_count += 1
            else:
                success_count += 1

            # Build log entry
            latencies = [v for k, v in res_dict.items() if k.endswith("_latency")]
            avg_row_latency = sum(latencies) / len(latencies) if latencies else 0
            log_entry = f"Row {idx}: avg {avg_row_latency:.2f}s"

            # Call progress callback
            if progress_callback:
                should_update = (
                    self.config.realtime_progress or
                    completed % 10 == 0 or
                    completed == total
                )
                if should_update:
                    progress_callback(
                        completed, total, success_count, error_count,
                        retry_count, log_entry, has_error
                    )

            # Auto-save partial results
            if self.config.save_path:
                save_freq = 5 if self.config.realtime_progress else 20
                if completed % save_freq == 0 or completed == total:
                    self._save_partial_results(df, results_map)

        # Create database result for this run
        combined_output = {idx: results_map[idx] for idx in sorted(results_map.keys())}
        db_result = RunResult.create(
            run_id=self.run_id,
            row_index=0,
            input_data=f"Comparison run with {len(self.config.models)} models",
            output=str(combined_output)[:10000],  # Truncate for DB
            status=ResultStatus.SUCCESS if error_count == 0 else ResultStatus.ERROR,
            latency=time.time() - start_time,
            error_type=None,
            error_message=None,
            retry_attempt=retry_count
        )
        all_db_results.append(db_result)

        # Save results to database
        self.db.save_results_batch(all_db_results)

        await http_client.aclose()

        # Calculate statistics
        all_latencies = []
        for res in results_map.values():
            for k, v in res.items():
                if k.endswith("_latency") and isinstance(v, (int, float)):
                    all_latencies.append(v)

        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        total_duration = time.time() - start_time

        # Calculate per-model average latencies
        for name, stats in per_model_stats.items():
            if stats["success"] > 0:
                stats["avg_latency"] = stats["total_latency"] / stats["success"]
            else:
                stats["avg_latency"] = 0.0

        # Build sorted results list
        sorted_results = [results_map[i] for i in sorted(results_map.keys())]

        return ComparisonResult(
            success_count=success_count,
            error_count=error_count,
            retry_count=retry_count,
            avg_latency=avg_latency,
            total_duration=total_duration,
            results=sorted_results,
            per_model_stats=per_model_stats
        )

    def _sanitize_column_name(self, name: str) -> str:
        """Convert display name to valid column name"""
        # Replace spaces and special chars with underscore
        safe = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        # Remove any remaining non-alphanumeric chars except underscore
        safe = "".join(c for c in safe if c.isalnum() or c == "_")
        # Collapse multiple underscores
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_")

    def _save_partial_results(self, df: pd.DataFrame, results_map: Dict):
        """Save partial results to disk"""
        import os
        try:
            os.makedirs(self.config.save_path, exist_ok=True)
            current_indices = sorted(results_map.keys())
            temp_df = df.loc[current_indices].copy()

            # Add result columns
            for idx in current_indices:
                for col, val in results_map[idx].items():
                    if col not in temp_df.columns:
                        temp_df[col] = None
                    temp_df.at[idx, col] = val

            temp_df.to_csv(
                os.path.join(self.config.save_path, "partial_comparison.csv"),
                index=False
            )
        except (OSError, IOError, KeyError):
            # Silently ignore file save errors during partial saves
            pass
