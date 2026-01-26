"""
Handai Processing Engines
Transform, Generate, and Document processing logic
"""

import asyncio
import time
import os
import json
import csv
from io import StringIO
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import pandas as pd

from .providers import LLMProvider, PROVIDER_CONFIGS, supports_json_mode
from .llm_client import get_client, create_http_client, call_llm_with_retry
from .document_reader import read_document, read_uploaded_file
from .document_templates import create_full_system_prompt
from database import get_db, RunResult, ResultStatus, RunStatus, LogLevel
from errors import ErrorClassifier


@dataclass
class ProcessingConfig:
    """Configuration for a processing run"""
    provider: LLMProvider
    api_key: str
    base_url: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    json_mode: bool
    max_concurrency: int
    auto_retry: bool
    max_retries: int
    save_path: Optional[str]
    realtime_progress: bool


@dataclass
class ProcessingResult:
    """Result of a processing run"""
    success_count: int
    error_count: int
    retry_count: int
    avg_latency: float
    total_duration: float
    results: List[Dict[str, Any]]


class TransformProcessor:
    """Processor for transforming existing datasets"""

    def __init__(self, config: ProcessingConfig, run_id: str, session_id: str):
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
    ) -> ProcessingResult:
        """
        Process a dataframe with the LLM.

        Args:
            df: Input dataframe
            system_prompt: System prompt for the LLM
            selected_cols: Columns to include in the prompt
            progress_callback: Optional callback for progress updates
                              signature: (completed, total, success, errors, retries, log_entry, is_error)

        Returns:
            ProcessingResult with statistics and results
        """
        http_client = create_http_client()
        client = get_client(
            self.config.provider,
            self.config.api_key,
            self.config.base_url,
            http_client
        )

        # Determine effective JSON mode
        effective_json_mode = self.config.json_mode and supports_json_mode(
            self.config.provider, self.config.model
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        results_map = {}
        all_results = []

        total = len(df)
        completed = 0
        success_count = 0
        error_count = 0
        retry_count = 0

        async def process_row(row_idx: int, row: pd.Series) -> Tuple[int, Dict, bool, RunResult]:
            async with semaphore:
                # Build row data
                if selected_cols:
                    row_data = row[selected_cols].to_json()
                else:
                    row_data = row.to_json()

                user_content = f"Data: {row_data}"

                output, duration, error_info, attempts = await call_llm_with_retry(
                    client, system_prompt, user_content, self.config.model,
                    self.config.temperature, self.config.max_tokens, effective_json_mode,
                    self.run_id, row_idx,
                    self.config.max_retries if self.config.auto_retry else 0,
                    self.db
                )

                if error_info:
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=row_idx,
                        input_data=row_data,
                        output=f"Error: {error_info.message}",
                        status=ResultStatus.ERROR,
                        latency=duration,
                        error_type=error_info.error_type.value,
                        error_message=error_info.original_error,
                        retry_attempt=attempts
                    )
                    return row_idx, {
                        "output": f"Error: {error_info.message}",
                        "latency": round(duration, 3),
                        "error": error_info
                    }, False, result
                else:
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=row_idx,
                        input_data=row_data,
                        output=output,
                        status=ResultStatus.SUCCESS,
                        latency=duration,
                        retry_attempt=attempts
                    )
                    return row_idx, {
                        "output": output,
                        "latency": round(duration, 3)
                    }, True, result

        # Create tasks
        tasks = [process_row(i, row) for i, row in df.iterrows()]
        start_time = time.time()

        # Process with progress updates
        for future in asyncio.as_completed(tasks):
            idx, res_dict, success, db_result = await future
            results_map[idx] = res_dict
            all_results.append(db_result)
            completed += 1

            if db_result.retry_attempt > 0:
                retry_count += db_result.retry_attempt

            if success:
                success_count += 1
                log_entry = f"Row {idx}: {res_dict['latency']}s"
            else:
                error_count += 1
                error_info = res_dict.get("error")
                log_entry = f"Row {idx}: {error_info.message if error_info else 'Error'}"

            # Call progress callback
            if progress_callback:
                should_update = (
                    self.config.realtime_progress or
                    completed % 10 == 0 or
                    completed == total
                )
                if should_update:
                    progress_callback(completed, total, success_count, error_count,
                                     retry_count, log_entry, not success)

            # Auto-save
            if self.config.save_path:
                save_freq = 5 if self.config.realtime_progress else 20
                if completed % save_freq == 0 or completed == total:
                    self._save_partial_results(df, results_map)

        # Save all results to database
        self.db.save_results_batch(all_results)

        await http_client.aclose()

        # Calculate statistics
        latencies = [r.latency for r in all_results if r.status == ResultStatus.SUCCESS.value]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_duration = time.time() - start_time

        # Build sorted results
        sorted_results = [results_map[i] for i in sorted(results_map.keys())]

        return ProcessingResult(
            success_count=success_count,
            error_count=error_count,
            retry_count=retry_count,
            avg_latency=avg_latency,
            total_duration=total_duration,
            results=sorted_results
        )

    def _save_partial_results(self, df: pd.DataFrame, results_map: Dict):
        """Save partial results to disk"""
        try:
            os.makedirs(self.config.save_path, exist_ok=True)
            current_indices = sorted(results_map.keys())
            temp_df = df.loc[current_indices].copy()
            temp_df["ai_output"] = [results_map[i].get("output") for i in current_indices]
            temp_df["latency_s"] = [results_map[i].get("latency", 0) for i in current_indices]
            temp_df.to_csv(os.path.join(self.config.save_path, "partial_handai.csv"), index=False)
        except Exception:
            pass


class GenerateProcessor:
    """Processor for generating new datasets"""

    def __init__(self, config: ProcessingConfig, run_id: str, session_id: str):
        self.config = config
        self.run_id = run_id
        self.session_id = session_id
        self.db = get_db()

    async def generate(
        self,
        count: int,
        generation_prompt: str,
        schema: Dict[str, str],
        variables: Dict[str, List[str]],
        use_freeform: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Generate new dataset rows.

        Args:
            count: Number of rows to generate
            generation_prompt: User's description of what to generate
            schema: Field definitions for structured generation
            variables: Variables to cycle through
            use_freeform: If True, let AI determine structure
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with statistics and generated data
        """
        http_client = create_http_client()
        client = get_client(
            self.config.provider,
            self.config.api_key,
            self.config.base_url,
            http_client
        )

        # Build system prompt
        if use_freeform:
            system_prompt = """You are a synthetic data generator. Based on the user's description, generate realistic, diverse data.

CRITICAL RULES:
1. Return ONLY valid JSON - a single object with appropriate fields
2. Determine the best schema based on the user's description
3. Each response should be unique and realistic
4. Vary the content naturally
5. Do not include any explanation, just the JSON object
6. Use sensible field names in snake_case"""
        else:
            schema_str = json.dumps(schema, indent=2)
            system_prompt = f"""You are a synthetic data generator. Generate realistic, diverse data following this exact schema:

{schema_str}

CRITICAL RULES:
1. Return ONLY valid JSON matching the schema exactly
2. Each response should be unique and realistic
3. Vary the content naturally
4. Do not include any explanation, just the JSON object"""

        # Determine if JSON mode is supported
        effective_json_mode = supports_json_mode(self.config.provider, self.config.model)

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        results = []
        all_db_results = []

        total = count
        completed = 0
        success_count = 0
        error_count = 0
        retry_count = 0

        async def generate_row(row_idx: int) -> Tuple[int, Dict, float, bool, RunResult]:
            async with semaphore:
                # Build prompt with variable substitution
                prompt = generation_prompt
                for var_name, var_values in variables.items():
                    if f"{{{var_name}}}" in prompt:
                        value = var_values[row_idx % len(var_values)]
                        prompt = prompt.replace(f"{{{var_name}}}", value)

                prompt = f"{prompt}\n\nGenerate row #{row_idx + 1}:"

                output, duration, error_info, attempts = await call_llm_with_retry(
                    client, system_prompt, prompt, self.config.model,
                    self.config.temperature, self.config.max_tokens, effective_json_mode,
                    self.run_id, row_idx,
                    self.config.max_retries if self.config.auto_retry else 0,
                    self.db
                )

                if error_info:
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=row_idx,
                        input_data=prompt,
                        output=f"Error: {error_info.message}",
                        status=ResultStatus.ERROR,
                        latency=duration,
                        error_type=error_info.error_type.value,
                        error_message=error_info.original_error,
                        retry_attempt=attempts
                    )
                    return row_idx, {"error": error_info.message}, round(duration, 3), False, result

                # Parse JSON
                try:
                    parsed = json.loads(output)
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=row_idx,
                        input_data=prompt,
                        output=output,
                        status=ResultStatus.SUCCESS,
                        latency=duration,
                        retry_attempt=attempts
                    )
                    return row_idx, parsed, round(duration, 3), True, result
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', output, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            result = RunResult.create(
                                run_id=self.run_id,
                                row_index=row_idx,
                                input_data=prompt,
                                output=json_match.group(),
                                status=ResultStatus.SUCCESS,
                                latency=duration,
                                retry_attempt=attempts
                            )
                            return row_idx, parsed, round(duration, 3), True, result
                        except:
                            pass

                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=row_idx,
                        input_data=prompt,
                        output=output,
                        status=ResultStatus.ERROR,
                        latency=duration,
                        error_type="json_parse_error",
                        error_message="Could not parse JSON from response",
                        retry_attempt=attempts
                    )
                    return row_idx, {"raw_output": output}, round(duration, 3), False, result

        # Create tasks
        tasks = [generate_row(i) for i in range(count)]
        start_time = time.time()

        # Process with progress updates
        for future in asyncio.as_completed(tasks):
            idx, data, latency, success, db_result = await future
            results.append((idx, data, latency, success))
            all_db_results.append(db_result)
            completed += 1

            if db_result.retry_attempt > 0:
                retry_count += db_result.retry_attempt

            if success:
                success_count += 1
                log_entry = f"Row {idx}: {latency}s"
            else:
                error_count += 1
                log_entry = f"Row {idx}: {data.get('error', 'Parse error')}"

            # Call progress callback
            if progress_callback:
                progress_callback(completed, total, success_count, error_count,
                                 retry_count, log_entry, not success)

        # Save results to database
        self.db.save_results_batch(all_db_results)

        await http_client.aclose()

        # Calculate statistics
        latencies = [r[2] for r in results if r[3]]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_duration = time.time() - start_time

        # Sort and format results
        results.sort(key=lambda x: x[0])
        formatted_results = [{"data": r[1], "latency": r[2], "success": r[3]} for r in results]

        return ProcessingResult(
            success_count=success_count,
            error_count=error_count,
            retry_count=retry_count,
            avg_latency=avg_latency,
            total_duration=total_duration,
            results=formatted_results
        )


class DocumentProcessor:
    """Processor for extracting data from documents"""

    def __init__(self, config: ProcessingConfig, run_id: str, session_id: str):
        self.config = config
        self.run_id = run_id
        self.session_id = session_id
        self.db = get_db()

    async def process(
        self,
        documents: List[Tuple[str, Any]],  # List of (name, source) tuples
        system_prompt: str,
        csv_columns: str,
        use_uploaded: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Process documents with the LLM.

        Args:
            documents: List of (document_name, document_source) tuples
            system_prompt: User's extraction prompt
            csv_columns: Expected CSV columns
            use_uploaded: True if sources are uploaded files
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with statistics and results
        """
        http_client = create_http_client()
        client = get_client(
            self.config.provider,
            self.config.api_key,
            self.config.base_url,
            http_client
        )

        # For document processing, we don't use JSON mode (we want CSV output)
        effective_json_mode = False

        # Create full system prompt with master prompt
        full_system_prompt = create_full_system_prompt(system_prompt)

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        results_map = {}
        all_results = []

        total = len(documents)
        completed = 0
        success_count = 0
        error_count = 0
        retry_count = 0

        async def process_doc(idx: int, doc_name: str, doc_source: Any) -> Tuple[int, Dict, bool, RunResult]:
            async with semaphore:
                # Read document
                try:
                    if use_uploaded:
                        content = read_uploaded_file(doc_source)
                    else:
                        content, _ = read_document(doc_source)
                    content_preview = content[:500] if content else ""
                except Exception as e:
                    error_info = ErrorClassifier.classify(e)
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=idx,
                        input_data=f"Document: {doc_name}",
                        output=f"Error reading: {str(e)}",
                        status=ResultStatus.ERROR,
                        latency=0.0,
                        error_type=error_info.error_type.value,
                        error_message=str(e),
                        retry_attempt=0
                    )
                    return idx, {
                        "document_name": doc_name,
                        "status": "ERROR",
                        "error": str(e),
                        "latency": 0.0
                    }, False, result

                user_content = f"Document: {doc_name}\n\nContent:\n{content}"

                output, duration, error_info, attempts = await call_llm_with_retry(
                    client, full_system_prompt, user_content, self.config.model,
                    self.config.temperature, self.config.max_tokens, effective_json_mode,
                    self.run_id, idx,
                    self.config.max_retries if self.config.auto_retry else 0,
                    self.db
                )

                if error_info:
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=idx,
                        input_data=f"Document: {doc_name}\nPreview: {content_preview}",
                        output=f"Error: {error_info.message}",
                        status=ResultStatus.ERROR,
                        latency=duration,
                        error_type=error_info.error_type.value,
                        error_message=error_info.original_error,
                        retry_attempt=attempts
                    )
                    return idx, {
                        "document_name": doc_name,
                        "status": "ERROR",
                        "error": error_info.message,
                        "latency": round(duration, 2)
                    }, False, result
                else:
                    result = RunResult.create(
                        run_id=self.run_id,
                        row_index=idx,
                        input_data=f"Document: {doc_name}\nPreview: {content_preview}",
                        output=output,
                        status=ResultStatus.SUCCESS,
                        latency=duration,
                        retry_attempt=attempts
                    )
                    return idx, {
                        "document_name": doc_name,
                        "status": "SUCCESS",
                        "output": output,
                        "latency": round(duration, 2)
                    }, True, result

        # Create tasks
        tasks = [process_doc(i, name, source) for i, (name, source) in enumerate(documents)]
        start_time = time.time()

        # Process with progress updates
        for future in asyncio.as_completed(tasks):
            idx, res_dict, success, db_result = await future
            results_map[idx] = res_dict
            all_results.append(db_result)
            completed += 1

            if db_result.retry_attempt > 0:
                retry_count += db_result.retry_attempt

            if success:
                success_count += 1
                log_entry = f"{res_dict['document_name']}: {res_dict['latency']}s"
            else:
                error_count += 1
                log_entry = f"{res_dict['document_name']}: {res_dict.get('error', 'Error')}"

            # Call progress callback
            if progress_callback:
                should_update = (
                    self.config.realtime_progress or
                    completed % 5 == 0 or
                    completed == total
                )
                if should_update:
                    progress_callback(completed, total, success_count, error_count,
                                     retry_count, log_entry, not success)

            # Auto-save
            if self.config.save_path:
                save_freq = 5 if self.config.realtime_progress else 10
                if completed % save_freq == 0 or completed == total:
                    self._save_partial_results(results_map, csv_columns)

        # Save all results to database
        self.db.save_results_batch(all_results)

        await http_client.aclose()

        # Calculate statistics
        latencies = [r.latency for r in all_results if r.status == ResultStatus.SUCCESS.value]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_duration = time.time() - start_time

        # Build sorted results
        sorted_results = [results_map[i] for i in sorted(results_map.keys())]

        return ProcessingResult(
            success_count=success_count,
            error_count=error_count,
            retry_count=retry_count,
            avg_latency=avg_latency,
            total_duration=total_duration,
            results=sorted_results
        )

    def _save_partial_results(self, results_map: Dict, csv_columns: str):
        """Save partial results to disk"""
        try:
            partial_results = [results_map[i] for i in sorted(results_map.keys())]
            partial_df = build_results_dataframe(partial_results, csv_columns)
            partial_df.to_csv(self.config.save_path, index=False)
        except Exception:
            pass


def clean_csv_output(text: str) -> str:
    """
    Remove markdown code blocks and clean up CSV output.

    Args:
        text: Raw LLM output

    Returns:
        Cleaned CSV text
    """
    if not text:
        return ""

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```csv"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    # Remove common prefixes the AI might add
    lines = text.strip().split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, headers, or explanatory text
        if not line:
            continue
        if line.lower().startswith(('note:', 'important:', 'here', 'the ', 'this ', 'i ', '*')):
            continue
        clean_lines.append(line)

    return '\n'.join(clean_lines)


def build_results_dataframe(results: List[Dict], columns: str) -> pd.DataFrame:
    """
    Build a clean DataFrame from document processing results.

    Args:
        results: List of result dicts with document_name, status, output/error
        columns: Comma-separated column names

    Returns:
        DataFrame with parsed results
    """
    col_list = [c.strip() for c in columns.split(',')]
    all_rows = []

    for r in results:
        doc_name = r["document_name"]
        status = r["status"]

        if status == "SUCCESS":
            output = clean_csv_output(r.get("output", ""))
            if not output or output.isspace():
                all_rows.append({
                    "document_name": doc_name,
                    "error": "Empty or whitespace-only response"
                })
            elif output:
                try:
                    reader = csv.reader(StringIO(output))
                    for csv_row in reader:
                        if not csv_row or all(cell.strip() == '' for cell in csv_row):
                            continue
                        row_dict = {"document_name": doc_name}
                        # Map CSV values to column names
                        for i, val in enumerate(csv_row):
                            if i < len(col_list):
                                row_dict[col_list[i]] = val.strip()
                            else:
                                row_dict[f"extra_{i}"] = val.strip()
                        # Fill missing columns with N/A
                        for col in col_list:
                            if col not in row_dict:
                                row_dict[col] = "N/A"
                        all_rows.append(row_dict)
                except Exception as e:
                    # Fallback: treat whole output as single field
                    row_dict = {
                        "document_name": doc_name,
                        "raw_output": output,
                        "parse_error": str(e)
                    }
                    all_rows.append(row_dict)
        else:
            all_rows.append({"document_name": doc_name, "error": r.get("error", "Unknown error")})

    if not all_rows:
        return pd.DataFrame(columns=["document_name"] + col_list)

    df = pd.DataFrame(all_rows)
    # Ensure document_name is first, then expected columns, then any extras
    final_cols = ["document_name"] + [c for c in col_list if c in df.columns]
    other_cols = [c for c in df.columns if c not in final_cols]
    return df[final_cols + other_cols]
