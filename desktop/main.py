"""
Handai v4.0 (Flet) - AI Data Transformer & Generator
Desktop app - Full layout with file picker
"""

import flet as ft
import pandas as pd
import asyncio
from openai import AsyncOpenAI
import httpx
import json
import os
import time
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path

# ==========================================
# PROVIDERS
# ==========================================

class LLMProvider(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google Gemini"
    GROQ = "Groq"
    TOGETHER = "Together AI"
    OPENROUTER = "OpenRouter"
    LM_STUDIO = "LM Studio"
    OLLAMA = "Ollama"
    CUSTOM = "Custom"

@dataclass
class ProviderConfig:
    base_url: Optional[str]
    default_model: str
    models: List[str]
    requires_key: bool

PROVIDERS = {
    LLMProvider.OPENAI: ProviderConfig(None, "gpt-4o", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], True),
    LLMProvider.ANTHROPIC: ProviderConfig("https://api.anthropic.com/v1", "claude-sonnet-4-20250514",
        ["claude-sonnet-4-20250514", "claude-opus-4-20250514"], True),
    LLMProvider.GOOGLE: ProviderConfig("https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.0-flash", ["gemini-2.0-flash", "gemini-1.5-pro"], True),
    LLMProvider.GROQ: ProviderConfig("https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"], True),
    LLMProvider.TOGETHER: ProviderConfig("https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo", ["meta-llama/Llama-3.3-70B-Instruct-Turbo"], True),
    LLMProvider.OPENROUTER: ProviderConfig("https://openrouter.ai/api/v1",
        "anthropic/claude-sonnet-4", ["anthropic/claude-sonnet-4", "openai/gpt-4o"], True),
    LLMProvider.LM_STUDIO: ProviderConfig("http://localhost:1234/v1", "local-model", ["local-model"], False),
    LLMProvider.OLLAMA: ProviderConfig("http://localhost:11434/v1", "llama3.2", ["llama3.2", "mistral"], False),
    LLMProvider.CUSTOM: ProviderConfig(None, "model", ["model"], False),
}


def get_settings_path():
    """Get path for settings file in user's home directory"""
    return Path.home() / ".handai_settings.json"

def load_settings():
    """Load settings from JSON file"""
    try:
        path = get_settings_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except:
        pass
    return {}

def save_settings_to_file(settings):
    """Save settings to JSON file"""
    try:
        path = get_settings_path()
        with open(path, "w") as f:
            json.dump(settings, f)
    except:
        pass

async def main(page: ft.Page):
    # State
    state = {
        "df": None, "results_df": None,
        "provider": LLMProvider.OPENAI,
        "api_key": "", "base_url": "", "model": "gpt-4o",
        "temperature": 0.0, "max_tokens": 2048, "max_concurrency": 5,
        "file_path": None,
    }

    # Run control
    run_control = {
        "stop_requested": False,
        "is_running": False,
        "partial_results": None,
        "partial_df": None,
        "completed_count": 0,
        "total_count": 0,
    }

    # Stats tracking
    stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_tokens_est": 0,
        "total_time": 0.0,
        "latencies": [],  # List of individual request latencies
        "last_run_time": 0.0,
        "last_run_rows": 0,
        "last_run_errors": 0,
        "session_start": time.time(),
    }

    # Load settings from JSON file
    saved = load_settings()
    if saved:
        try: state["provider"] = LLMProvider(saved.get("selected_provider", "OpenAI"))
        except: pass
        state["api_key"] = saved.get("api_key", "")
        state["base_url"] = saved.get("base_url", "")
        state["model"] = saved.get("model_name", "gpt-4o")
        state["temperature"] = saved.get("temperature", 0.0)
        state["max_tokens"] = saved.get("max_tokens", 2048)
        state["max_concurrency"] = saved.get("max_concurrency", 5)

    # Page setup
    page.title = "Handai - AI Data Transformer"
    page.theme_mode = ft.ThemeMode.LIGHT  # Always light theme
    page.bgcolor = ft.Colors.WHITE
    page.padding = 0
    page.window.width = 1000
    page.window.height = 700

    # ==========================================
    # FILE PICKER
    # ==========================================
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def pick_and_load_file(e):
        files = await file_picker.pick_files(
            dialog_title="Select a data file",
            allowed_extensions=["csv", "xlsx", "xls", "json"],
            allow_multiple=False
        )
        if files and len(files) > 0:
            path = files[0].path
            state["file_path"] = path
            try:
                ext = path.split(".")[-1].lower()
                if ext == "csv":
                    state["df"] = pd.read_csv(path)
                elif ext in ["xlsx", "xls"]:
                    state["df"] = pd.read_excel(path)
                elif ext == "json":
                    state["df"] = pd.read_json(path)

                file_label.value = f"Loaded: {os.path.basename(path)} ({len(state['df'])} rows, {len(state['df'].columns)} cols)"
                data_preview.value = state["df"].head(5).to_string()
                status.value = f"File loaded: {len(state['df'])} rows"
            except Exception as ex:
                status.value = f"Error loading file: {ex}"
            page.update()

    # Status
    status = ft.Text("Ready", size=12, color=ft.Colors.GREY_600)

    # ==========================================
    # TRANSFORM VIEW
    # ==========================================
    file_label = ft.Text("No file loaded", size=14)
    data_preview = ft.Text("", selectable=True, size=11)

    # Text area for pasting data directly
    paste_data_field = ft.TextField(
        label="Or paste CSV/JSON data here",
        hint_text="Paste CSV data (with headers) or JSON array...",
        multiline=True,
        min_lines=3,
        max_lines=6,
        expand=True,
    )

    def load_pasted_data(e):
        text = paste_data_field.value
        if not text or not text.strip():
            status.value = "No data to parse"
            page.update()
            return
        try:
            # Try JSON first
            import json
            import io
            text = text.strip()
            if text.startswith('[') or text.startswith('{'):
                data = json.loads(text)
                if isinstance(data, dict):
                    data = [data]
                state["df"] = pd.DataFrame(data)
            else:
                # Try CSV
                state["df"] = pd.read_csv(io.StringIO(text))

            file_label.value = f"Pasted data: {len(state['df'])} rows, {len(state['df'].columns)} cols"
            data_preview.value = state["df"].head(5).to_string()
            status.value = f"Data loaded: {len(state['df'])} rows"
        except Exception as ex:
            status.value = f"Parse error: {ex}"
        page.update()

    prompt_field = ft.TextField(
        label="AI Instructions (System Prompt)",
        hint_text="Example: For each row, extract the main topic and classify the sentiment as positive/negative/neutral. Return JSON with 'topic' and 'sentiment' fields.",
        multiline=True,
        min_lines=4,
        max_lines=8,
        expand=True,
    )

    results_text = ft.Text("", selectable=True, size=11, font_family="monospace")
    results_field = ft.Container(
        content=ft.Column([results_text], scroll=ft.ScrollMode.AUTO, expand=True),
        bgcolor=ft.Colors.GREY_100,
        border=ft.Border.all(1, ft.Colors.GREY_400),
        border_radius=5,
        padding=10,
        expand=True,
        height=250,
    )

    progress = ft.ProgressBar(visible=False)

    def stop_run(e):
        run_control["stop_requested"] = True
        status.value = "Stopping... (waiting for current requests to finish)"
        page.update()

    stop_btn = ft.Button("Stop", icon=ft.Icons.STOP, bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE,
                          on_click=stop_run, visible=False)
    resume_btn = ft.Button("Resume", icon=ft.Icons.PLAY_ARROW, bgcolor=ft.Colors.ORANGE_400, color=ft.Colors.WHITE,
                            on_click=lambda _: asyncio.create_task(run_transform(test=False, resume=True)), visible=False)

    async def run_transform(test=True, resume=False):
        if not resume:
            if state["df"] is None:
                status.value = "Please load a file first"
                page.update()
                return

            if not prompt_field.value:
                status.value = "Please enter AI instructions"
                page.update()
                return

        cfg = PROVIDERS[state["provider"]]
        if cfg.requires_key and not state["api_key"]:
            status.value = "Set API key in Settings tab"
            page.update()
            return

        # Handle resume from partial results
        if resume and run_control["partial_results"] is not None:
            df = run_control["partial_df"]
            results = run_control["partial_results"]
            start_idx = run_control["completed_count"]
        else:
            df = state["df"].head(10 if test else len(state["df"])).reset_index(drop=True)
            results = [None] * len(df)
            start_idx = 0

        run_control["is_running"] = True
        run_control["stop_requested"] = False
        run_control["partial_df"] = df
        run_control["total_count"] = len(df)

        progress.visible = True
        stop_btn.visible = True
        status.value = "Processing..." if not resume else "Resuming..."
        page.update()

        url = state["base_url"] or cfg.base_url
        completed = [start_idx]
        errors = [0]
        semaphore = asyncio.Semaphore(state["max_concurrency"])
        run_start = time.time()

        async def process_row(i, row, client):
            if run_control["stop_requested"]:
                return  # Skip if stop requested

            async with semaphore:
                if run_control["stop_requested"]:
                    return  # Check again after acquiring semaphore

                req_start = time.time()
                try:
                    # Build system prompt with format instructions
                    system_prompt = f"""{prompt_field.value}

CRITICAL OUTPUT FORMAT RULES:
- Return ONLY a valid JSON object (not array, not plain text)
- The JSON must be flat/tabular - suitable for CSV export
- Each key becomes a column, each value becomes a cell
- Use simple types only: strings, numbers, booleans (no nested objects/arrays)
- Example valid output: {{"category": "tech", "sentiment": "positive", "score": 0.85}}
- Do NOT include markdown, explanations, or anything outside the JSON"""

                    params = {
                        "model": state["model"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Process this data row:\n{row.to_json()}"}
                        ],
                    }
                    if state["provider"] == LLMProvider.OPENAI:
                        params["max_completion_tokens"] = state["max_tokens"]
                        model_lower = state["model"].lower()
                        if model_lower.startswith(("gpt-3", "gpt-4")) and not "o1" in model_lower:
                            params["temperature"] = state["temperature"]
                    else:
                        params["max_tokens"] = state["max_tokens"]
                        params["temperature"] = state["temperature"]

                    r = await client.chat.completions.create(**params)
                    latency = time.time() - req_start
                    stats["latencies"].append(latency)
                    stats["successful_requests"] += 1
                    out = r.choices[0].message.content.strip()
                    # Clean markdown if present
                    if out.startswith("```"):
                        out = out.split("\n", 1)[1] if "\n" in out else out
                        out = out.rsplit("```", 1)[0] if "```" in out else out
                    # Parse as JSON for proper tabular output
                    try:
                        parsed = json.loads(out.strip())
                        if isinstance(parsed, dict):
                            results[i] = parsed
                        elif isinstance(parsed, list) and parsed:
                            results[i] = parsed[0] if isinstance(parsed[0], dict) else {"output": str(parsed)}
                        else:
                            results[i] = {"output": str(parsed)}
                    except:
                        results[i] = {"output": out[:500]}
                except Exception as ex:
                    results[i] = {"error": str(ex)[:200]}
                    errors[0] += 1
                    stats["failed_requests"] += 1

                stats["total_requests"] += 1
                completed[0] += 1
                run_control["completed_count"] = completed[0]
                progress.value = completed[0] / len(df)
                status.value = f"Processing {completed[0]}/{len(df)} (concurrency: {state['max_concurrency']})..."
                page.update()

        async with httpx.AsyncClient(timeout=120) as http:
            client = AsyncOpenAI(
                api_key=state["api_key"] if cfg.requires_key else "x",
                base_url=url, http_client=http
            )
            # Only process rows that haven't been done yet
            tasks = [process_row(i, row, client) for i, row in df.iterrows() if results[i] is None]
            await asyncio.gather(*tasks)

        # Update stats
        run_time = time.time() - run_start
        stats["total_time"] += run_time
        stats["last_run_time"] = run_time
        stats["last_run_rows"] = completed[0]
        stats["last_run_errors"] = errors[0]

        # Check if stopped
        was_stopped = run_control["stop_requested"]
        run_control["is_running"] = False
        stop_btn.visible = False

        # Save partial results for potential resume
        run_control["partial_results"] = results
        run_control["partial_df"] = df

        # Count how many were actually completed
        actual_completed = sum(1 for r in results if r is not None)

        # Expand JSON results into columns and merge with original data
        # Filter out None results for display
        completed_results = [r if r is not None else {"status": "not_processed"} for r in results]
        ai_df = pd.DataFrame(completed_results)
        ai_df.columns = [f"ai_{col}" for col in ai_df.columns]
        result_df = pd.concat([df.reset_index(drop=True), ai_df.reset_index(drop=True)], axis=1)
        state["results_df"] = result_df
        with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
            results_text.value = result_df.to_string()
        progress.visible = False

        if was_stopped:
            status.value = f"Stopped! {actual_completed}/{len(df)} rows completed. Use Resume to continue."
            resume_btn.visible = True
        else:
            status.value = f"Done! Processed {len(df)} rows - {len(ai_df.columns)} AI columns added"
            run_control["partial_results"] = None  # Clear partial on complete
            resume_btn.visible = False
        page.update()

    async def export_results(e):
        if state["results_df"] is None or state["results_df"].empty:
            status.value = "No data to export"
            page.update()
            return
        try:
            # Use file picker to save
            save_path = await file_picker.save_file(
                dialog_title="Save CSV",
                file_name="handai_results.csv",
                allowed_extensions=["csv"]
            )
            if save_path:
                state["results_df"].to_csv(save_path, index=False)
                status.value = f"Exported to {save_path}"
            else:
                # Fallback to current directory
                state["results_df"].to_csv("handai_results.csv", index=False)
                status.value = "Exported to handai_results.csv (current folder)"
        except Exception as ex:
            status.value = f"Export error: {ex}"
        page.update()

    transform_view = ft.Container(
        content=ft.Column([
            ft.Text("Enrich / Process Data", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Load your CSV/JSON data and let AI process each row (e.g., classify, extract, summarize, translate)",
                    size=12, color=ft.Colors.GREY_600),
            ft.Divider(),

            # File section
            ft.Text("1. Load Data", size=16, weight=ft.FontWeight.W_500),
            ft.Row([
                ft.Button(
                    "Select File",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=pick_and_load_file,
                ),
                file_label,
            ]),
            paste_data_field,
            ft.Button("Load Pasted Data", icon=ft.Icons.CONTENT_PASTE, on_click=load_pasted_data),
            ft.Container(
                content=data_preview,
                bgcolor=ft.Colors.GREY_100,
                padding=10,
                border_radius=5,
                height=120,
            ),

            ft.Divider(),

            # Prompt section
            ft.Text("2. AI Instructions", size=16, weight=ft.FontWeight.W_500),
            prompt_field,

            ft.Divider(),

            # Run section
            ft.Text("3. Execute", size=16, weight=ft.FontWeight.W_500),
            ft.Row([
                ft.Button("Test (10 rows)", icon=ft.Icons.SCIENCE,
                    on_click=lambda _: asyncio.create_task(run_transform(True))),
                ft.Button("Full Run", icon=ft.Icons.PLAY_ARROW,
                    bgcolor=ft.Colors.GREEN_400, color=ft.Colors.WHITE,
                    on_click=lambda _: asyncio.create_task(run_transform(False))),
                stop_btn,
                resume_btn,
                ft.Button("Export CSV", icon=ft.Icons.DOWNLOAD, on_click=lambda _: asyncio.create_task(export_results(_))),
            ]),
            progress,

            ft.Divider(),

            # Results section
            ft.Text("4. Results", size=16, weight=ft.FontWeight.W_500),
            results_field,
        ],
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
        expand=True),
        padding=20,
        expand=True,
    )

    # ==========================================
    # GENERATE VIEW
    # ==========================================
    gen_prompt = ft.TextField(
        label="Describe the data to generate",
        hint_text="Example: Customer profiles with name, email, age (18-65), city, and favorite product category",
        multiline=True, min_lines=3, expand=True,
    )
    gen_rows = ft.TextField(label="Rows", value="10", width=100)

    # Sample prompt for student data
    SAMPLE_STUDENT_PROMPT = """University student academic records with the following fields:
- student_id: unique 6-digit ID starting with 'STU'
- full_name: realistic full name (diverse international names)
- gender: Male, Female, or Non-binary
- age: between 18-28
- country: country of origin (diverse global representation)
- program: academic program (e.g., Computer Science, Business, Engineering, Medicine, Arts)
- year: 1st, 2nd, 3rd, or 4th year
- gpa: GPA score between 2.0 and 4.0 (realistic distribution)
- courses_enrolled: number of courses this semester (3-6)
- extracurricular: one main activity (Sports, Music, Debate, Volunteering, Research, None)
- motivation_level: High, Medium, or Low
- study_hours_weekly: average study hours per week (5-40)
- scholarship: true or false
- attendance_rate: percentage between 60-100"""

    def fill_sample_prompt(e):
        gen_prompt.value = SAMPLE_STUDENT_PROMPT
        gen_rows.value = "20"
        page.update()
    gen_results_text = ft.Text("", selectable=True, size=11, font_family="monospace")
    gen_results = ft.Container(
        content=ft.Column([gen_results_text], scroll=ft.ScrollMode.AUTO, expand=True),
        bgcolor=ft.Colors.GREY_100,
        border=ft.Border.all(1, ft.Colors.GREY_400),
        border_radius=5,
        padding=10,
        expand=True,
        height=300,
    )
    gen_progress = ft.ProgressBar(visible=False)

    gen_stop_btn = ft.Button("Stop", icon=ft.Icons.STOP, bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE,
                              on_click=lambda e: stop_run(e), visible=False)
    gen_resume_btn = ft.Button("Resume", icon=ft.Icons.PLAY_ARROW, bgcolor=ft.Colors.ORANGE_400, color=ft.Colors.WHITE,
                                on_click=lambda _: asyncio.create_task(run_generate(resume=True)), visible=False)

    async def run_generate(resume=False):
        if not resume and not gen_prompt.value:
            status.value = "Describe the data first"
            page.update()
            return

        cfg = PROVIDERS[state["provider"]]
        if cfg.requires_key and not state["api_key"]:
            status.value = "Set API key in Settings"
            page.update()
            return

        try: n = int(gen_rows.value)
        except: n = 10

        # Handle resume
        if resume and run_control["partial_results"] is not None:
            results = run_control["partial_results"]
            n = len(results)
        else:
            results = [None] * n

        run_control["is_running"] = True
        run_control["stop_requested"] = False
        run_control["total_count"] = n

        gen_progress.visible = True
        gen_stop_btn.visible = True
        status.value = "Generating..." if not resume else "Resuming..."
        page.update()

        url = state["base_url"] or cfg.base_url
        completed = [sum(1 for r in results if r is not None)]
        errors = [0]
        semaphore = asyncio.Semaphore(state["max_concurrency"])
        run_start = time.time()

        async def generate_item(i, client):
            if run_control["stop_requested"]:
                return

            async with semaphore:
                if run_control["stop_requested"]:
                    return

                req_start = time.time()
                try:
                    # Build system prompt with format instructions
                    gen_system_prompt = f"""Generate synthetic data based on this description: {gen_prompt.value}

CRITICAL OUTPUT FORMAT RULES:
- Return ONLY a valid JSON object (not array, not plain text)
- The JSON must be flat/tabular - suitable for CSV export
- Each key becomes a column name, each value becomes a cell value
- Use simple types only: strings, numbers, booleans
- Do NOT use nested objects or arrays as values
- Keep the same keys/columns for every generated item (consistent schema)
- Example valid output: {{"name": "John Doe", "age": 32, "city": "New York", "active": true}}
- Do NOT include markdown code blocks, explanations, or anything outside the JSON
- Generate realistic, varied, and unique data for each item"""

                    gen_params = {
                        "model": state["model"],
                        "messages": [
                            {"role": "system", "content": gen_system_prompt},
                            {"role": "user", "content": f"Generate unique item #{i+1}"}
                        ],
                    }
                    if state["provider"] == LLMProvider.OPENAI:
                        gen_params["max_completion_tokens"] = state["max_tokens"]
                        model_lower = state["model"].lower()
                        if model_lower.startswith(("gpt-3", "gpt-4")) and not "o1" in model_lower:
                            gen_params["temperature"] = 0.9
                    else:
                        gen_params["max_tokens"] = state["max_tokens"]
                        gen_params["temperature"] = 0.9

                    r = await client.chat.completions.create(**gen_params)
                    latency = time.time() - req_start
                    stats["latencies"].append(latency)
                    stats["successful_requests"] += 1
                    out = r.choices[0].message.content.strip()
                    # Clean markdown if present
                    if out.startswith("```"):
                        out = out.split("\n", 1)[1] if "\n" in out else out
                        out = out.rsplit("```", 1)[0] if "```" in out else out
                    try:
                        parsed = json.loads(out)
                        if isinstance(parsed, dict):
                            results[i] = parsed
                        elif isinstance(parsed, list) and parsed:
                            results[i] = parsed[0] if isinstance(parsed[0], dict) else {"data": parsed}
                        else:
                            results[i] = {"value": parsed}
                    except:
                        results[i] = {"raw": out[:150]}
                except Exception as ex:
                    results[i] = {"error": str(ex)[:80]}
                    errors[0] += 1
                    stats["failed_requests"] += 1

                stats["total_requests"] += 1
                completed[0] += 1
                run_control["completed_count"] = completed[0]
                gen_progress.value = completed[0] / n
                status.value = f"Generated {completed[0]}/{n} (concurrency: {state['max_concurrency']})..."
                page.update()

        async with httpx.AsyncClient(timeout=120) as http:
            client = AsyncOpenAI(
                api_key=state["api_key"] if cfg.requires_key else "x",
                base_url=url, http_client=http
            )
            # Only generate items that haven't been done yet
            tasks = [generate_item(i, client) for i in range(n) if results[i] is None]
            await asyncio.gather(*tasks)

        # Update stats
        run_time = time.time() - run_start
        stats["total_time"] += run_time
        stats["last_run_time"] = run_time
        stats["last_run_rows"] = completed[0]
        stats["last_run_errors"] = errors[0]

        # Check if stopped
        was_stopped = run_control["stop_requested"]
        run_control["is_running"] = False
        gen_stop_btn.visible = False

        # Save partial results
        run_control["partial_results"] = results

        # Count completed
        actual_completed = sum(1 for r in results if r is not None)

        # Build dataframe from completed results
        completed_results = [r if r is not None else {"status": "not_generated"} for r in results]
        state["results_df"] = pd.DataFrame(completed_results)
        with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
            gen_results_text.value = state["results_df"].to_string()
        gen_progress.visible = False

        if was_stopped:
            status.value = f"Stopped! {actual_completed}/{n} rows generated. Use Resume to continue."
            gen_resume_btn.visible = True
        else:
            status.value = f"Generated {n} rows in {run_time:.1f}s"
            run_control["partial_results"] = None
            gen_resume_btn.visible = False
        page.update()

    generate_view = ft.Container(
        content=ft.Column([
            ft.Text("Generate New Data", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Create synthetic data from scratch - describe what you want and AI will generate rows",
                    size=12, color=ft.Colors.GREY_600),
            ft.Divider(),

            ft.Text("1. Describe Your Data", size=16, weight=ft.FontWeight.W_500),
            gen_prompt,
            ft.Row([
                ft.Button("Sample: Student Data", icon=ft.Icons.SCHOOL,
                    on_click=fill_sample_prompt,
                    tooltip="Fill with sample prompt for student academic records"),
                gen_rows,
                ft.Text("rows to generate")
            ]),

            ft.Divider(),

            ft.Text("2. Generate", size=16, weight=ft.FontWeight.W_500),
            ft.Row([
                ft.Button("Generate", icon=ft.Icons.AUTO_AWESOME,
                    bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE,
                    on_click=lambda _: asyncio.create_task(run_generate())),
                gen_stop_btn,
                gen_resume_btn,
                ft.Button("Export CSV", icon=ft.Icons.DOWNLOAD, on_click=lambda _: asyncio.create_task(export_results(_))),
            ]),
            gen_progress,

            ft.Divider(),

            ft.Text("3. Results", size=16, weight=ft.FontWeight.W_500),
            gen_results,
        ],
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
        expand=True),
        padding=20,
        expand=True,
    )

    # ==========================================
    # STATS VIEW
    # ==========================================
    stats_total_requests = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    stats_success_rate = ft.Text("0%", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_600)
    stats_avg_latency = ft.Text("0.0s", size=28, weight=ft.FontWeight.BOLD)
    stats_throughput = ft.Text("0/min", size=28, weight=ft.FontWeight.BOLD)
    stats_last_run_info = ft.Text("No runs yet", size=14)
    stats_errors_count = ft.Text("0", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_400)
    stats_session_time = ft.Text("0m", size=14)

    def refresh_stats():
        stats_total_requests.value = str(stats["total_requests"])

        if stats["total_requests"] > 0:
            success_rate = (stats["successful_requests"] / stats["total_requests"]) * 100
            stats_success_rate.value = f"{success_rate:.1f}%"
            stats_success_rate.color = ft.Colors.GREEN_600 if success_rate >= 90 else (ft.Colors.ORANGE_400 if success_rate >= 70 else ft.Colors.RED_400)
        else:
            stats_success_rate.value = "-%"

        if stats["latencies"]:
            avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
            stats_avg_latency.value = f"{avg_latency:.2f}s"
        else:
            stats_avg_latency.value = "-"

        if stats["total_time"] > 0:
            throughput = (stats["total_requests"] / stats["total_time"]) * 60
            stats_throughput.value = f"{throughput:.1f}/min"
        else:
            stats_throughput.value = "-"

        stats_errors_count.value = str(stats["failed_requests"])

        if stats["last_run_rows"] > 0:
            speed = stats["last_run_rows"] / stats["last_run_time"] if stats["last_run_time"] > 0 else 0
            stats_last_run_info.value = f"Last run: {stats['last_run_rows']} rows in {stats['last_run_time']:.1f}s ({speed:.1f} rows/sec) | {stats['last_run_errors']} errors"

        session_mins = (time.time() - stats["session_start"]) / 60
        stats_session_time.value = f"Session: {session_mins:.0f} min"

        page.update()

    def reset_stats(e):
        stats["total_requests"] = 0
        stats["successful_requests"] = 0
        stats["failed_requests"] = 0
        stats["total_time"] = 0.0
        stats["latencies"] = []
        stats["last_run_time"] = 0.0
        stats["last_run_rows"] = 0
        stats["last_run_errors"] = 0
        stats["session_start"] = time.time()
        refresh_stats()

    def make_stat_card(title, value_widget, icon, color):
        return ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(icon, color=color, size=20), ft.Text(title, size=12, color=ft.Colors.GREY_600)]),
                value_widget,
            ], spacing=5),
            bgcolor=ft.Colors.WHITE,
            border=ft.Border.all(1, ft.Colors.GREY_300),
            border_radius=10,
            padding=15,
            width=180,
        )

    stats_view = ft.Container(
        content=ft.Column([
            ft.Text("Performance Stats", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Real-time metrics for your AI processing tasks", size=12, color=ft.Colors.GREY_600),
            ft.Divider(),

            ft.Row([
                make_stat_card("Total Requests", stats_total_requests, ft.Icons.SEND, ft.Colors.BLUE_400),
                make_stat_card("Success Rate", stats_success_rate, ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN_600),
                make_stat_card("Avg Latency", stats_avg_latency, ft.Icons.TIMER, ft.Colors.ORANGE_400),
            ], wrap=True, spacing=15),

            ft.Row([
                make_stat_card("Throughput", stats_throughput, ft.Icons.SPEED, ft.Colors.PURPLE_400),
                make_stat_card("Errors", stats_errors_count, ft.Icons.ERROR, ft.Colors.RED_400),
            ], wrap=True, spacing=15),

            ft.Divider(),

            stats_last_run_info,
            stats_session_time,

            ft.Divider(),

            ft.Row([
                ft.Button("Refresh Stats", icon=ft.Icons.REFRESH, on_click=lambda _: refresh_stats()),
                ft.Button("Reset Stats", icon=ft.Icons.RESTART_ALT, on_click=reset_stats),
            ]),
        ],
        spacing=15,
        expand=True),
        padding=20,
        expand=True,
    )

    # ==========================================
    # SETTINGS VIEW
    # ==========================================
    provider_dd = ft.Dropdown(
        label="AI Provider", width=300,
        value=state["provider"].value,
        options=[ft.dropdown.Option(p.value) for p in LLMProvider],
    )
    api_key_tf = ft.TextField(label="API Key", width=400, value=state["api_key"],
                               password=True, can_reveal_password=True)
    base_url_tf = ft.TextField(label="Base URL (optional)", width=400, value=state["base_url"],
                                hint_text="Leave empty for default")
    model_tf = ft.TextField(label="Model", width=300, value=state["model"])
    temp_slider = ft.Slider(min=0, max=2, value=state["temperature"], divisions=20, label="{value}", width=300)
    tokens_tf = ft.TextField(label="Max Tokens", width=150, value=str(state["max_tokens"]))
    concurrency_tf = ft.TextField(label="Concurrent Requests", width=150, value=str(state["max_concurrency"]),
                                   hint_text="1-100")

    def save_settings(e):
        try: state["provider"] = LLMProvider(provider_dd.value)
        except: pass
        state["api_key"] = api_key_tf.value
        state["base_url"] = base_url_tf.value
        state["model"] = model_tf.value
        state["temperature"] = temp_slider.value
        try: state["max_tokens"] = int(tokens_tf.value)
        except: pass
        try: state["max_concurrency"] = max(1, min(100, int(concurrency_tf.value)))
        except: pass

        save_settings_to_file({
            "selected_provider": state["provider"].value,
            "api_key": state["api_key"],
            "base_url": state["base_url"],
            "model_name": state["model"],
            "temperature": state["temperature"],
            "max_tokens": state["max_tokens"],
            "max_concurrency": state["max_concurrency"],
        })
        status.value = "Settings saved!"
        page.update()

    def on_provider_change(e):
        try:
            p = LLMProvider(provider_dd.value)
            cfg = PROVIDERS[p]
            model_tf.value = cfg.default_model
            base_url_tf.value = cfg.base_url or ""
            page.update()
        except: pass

    provider_dd.on_change = on_provider_change

    settings_view = ft.Container(
        content=ft.Column([
            ft.Text("Settings", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(),

            ft.Text("AI Provider", size=16, weight=ft.FontWeight.W_500),
            provider_dd,

            ft.Divider(),

            ft.Text("Authentication", size=16, weight=ft.FontWeight.W_500),
            api_key_tf,
            base_url_tf,

            ft.Divider(),

            ft.Text("Model Settings", size=16, weight=ft.FontWeight.W_500),
            model_tf,
            ft.Text("Temperature (0 = deterministic, 2 = creative)", size=12, color=ft.Colors.GREY_600),
            temp_slider,
            ft.Row([tokens_tf, concurrency_tf]),
            ft.Text("Higher concurrency = faster but may hit rate limits", size=12, color=ft.Colors.GREY_600),

            ft.Divider(),

            ft.Button("Save Settings", icon=ft.Icons.SAVE,
                bgcolor=ft.Colors.GREEN_400, color=ft.Colors.WHITE,
                on_click=save_settings),
        ],
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
        expand=True),
        padding=20,
        expand=True,
    )

    # ==========================================
    # NAVIGATION
    # ==========================================
    content = ft.Container(content=generate_view, expand=True)  # Start with Generate

    def nav_change(e):
        idx = e.control.selected_index
        views = [generate_view, transform_view, stats_view, settings_view]  # Generate first
        content.content = views[idx]
        if idx == 2:
            refresh_stats()
        page.update()

    nav = ft.NavigationBar(
        selected_index=0,
        on_change=nav_change,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.AUTO_AWESOME, label="Generate"),
            ft.NavigationBarDestination(icon=ft.Icons.TABLE_ROWS, label="Enrich"),
            ft.NavigationBarDestination(icon=ft.Icons.ANALYTICS, label="Stats"),
            ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="Settings"),
        ],
    )

    # Main layout
    page.add(
        ft.Column([
            content,
            ft.Container(
                content=ft.Row([status], alignment=ft.MainAxisAlignment.CENTER),
                bgcolor=ft.Colors.GREY_200,
                padding=5,
            ),
            nav,
        ], expand=True, spacing=0)
    )


if __name__ == "__main__":
    ft.run(main)
