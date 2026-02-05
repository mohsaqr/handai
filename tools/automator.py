"""
Automator Tool
Swiss-army-knife batch processing tool that applies any user-defined task across a dataset
"""

import streamlit as st
import pandas as pd
import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field, asdict
import asyncio
import time

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS, supports_json_mode
from core.llm_client import get_client, create_http_client, call_llm_with_retry
from core.prompt_registry import get_effective_prompt, ensure_prompts_registered
from database import get_db, RunResult, ResultStatus, RunStatus, LogLevel
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model,
    get_current_settings, set_current_session_id, get_current_session_id
)

# Sample data for testing
SAMPLE_AUTOMATOR_DATA = {
    "text": [
        "The new iPhone 15 Pro has an amazing camera system and titanium design.",
        "Climate change is causing unprecedented weather patterns across the globe.",
        "The quarterly earnings report shows a 15% increase in revenue.",
        "A new study reveals the health benefits of Mediterranean diet.",
        "Tech stocks rallied today after positive AI industry news.",
        "The local community center is hosting a charity event this weekend.",
        "Scientists discover a new species of deep-sea fish near Japan.",
        "The city council approved a new affordable housing development.",
        "A famous artist's painting sold for $50 million at auction.",
        "The startup raised $10 million in Series A funding."
    ],
    "source": ["Tech Blog", "News Wire", "Business Daily", "Health Journal", "Market Watch",
               "Local News", "Science Today", "City Report", "Art Weekly", "Startup News"],
    "date": ["2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12", "2024-01-11",
             "2024-01-10", "2024-01-09", "2024-01-08", "2024-01-07", "2024-01-06"]
}


@dataclass
class OutputField:
    """Definition for a single output field"""
    name: str
    field_type: str  # text, number, decimal, boolean, list
    required: bool = True
    constraints: Optional[str] = None  # e.g., "one of: positive, negative, neutral"
    default: Optional[str] = None  # fallback for edge cases


@dataclass
class FewShotExample:
    """A few-shot example for the automator"""
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]


@dataclass
class PipelineStep:
    """Single step in a multi-step pipeline"""
    step_id: str                      # "step_0", "step_1", etc.
    name: str                         # User-friendly: "Translate", "Code"
    task_description: str             # What this step does
    input_fields: List[str]           # Original cols OR previous step outputs
    output_fields: List[OutputField]  # Fields this step produces
    output_format: str = "json"       # json, csv, text


@dataclass
class RowPipelineContext:
    """Tracks a row through the pipeline"""
    row_idx: int
    original_data: Dict[str, Any]
    step_outputs: Dict[str, Dict]     # step_id -> {field: value}
    cumulative_data: Dict[str, Any]   # All fields available
    step_latencies: Dict[str, float]
    step_errors: Dict[str, str]
    status: str                       # "success", "partial", "failed"


@dataclass
class AutomatorConfig:
    """Complete configuration for an automator run"""
    task_description: str
    input_columns: List[str]
    output_fields: List[OutputField]
    output_format: str  # json, csv, text, xml
    few_shot_examples: List[FewShotExample] = field(default_factory=list)
    confidence_enabled: bool = False
    confidence_threshold: int = 70
    consistency_prompt: Optional[str] = None
    handle_empty: str = "skip"  # skip, default, error
    system_prompt_override: Optional[str] = None
    # Multi-step pipeline: if empty, use single-step mode (backward compatible)
    steps: List[PipelineStep] = field(default_factory=list)


def sanitize_llm_output(text: str) -> str:
    """
    Remove potentially harmful or malformed tags from LLM output.
    Strips patterns like <|channel|>, <|constrain|>, <|message|>, etc.
    """
    if not text:
        return text

    # Remove <|...|> style tags (prompt injection artifacts)
    text = re.sub(r'<\|[^|>]+\|>', '', text)

    # Remove common prompt injection patterns
    text = re.sub(r'<\|channel\|>[^<]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|constrain\|>[^<]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|message\|>[^<]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|result\|>[^<]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|final\|>[^<]*', '', text, flags=re.IGNORECASE)

    # Clean up any leftover artifacts
    text = re.sub(r'\s*result\s*result\s*result\s*', ' ', text)

    return text.strip()


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from text that may contain extra content."""
    if not text:
        return None

    # Sanitize first
    text = sanitize_llm_output(text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Look for { ... } pattern
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def extract_xml(text: str, fields: List[OutputField]) -> Optional[Dict]:
    """Extract data from XML output."""
    if not text:
        return None

    # Sanitize and clean up the text
    text = sanitize_llm_output(text)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        text = text.strip()

    # Try to parse XML
    try:
        # Wrap in root if needed
        if not text.startswith("<"):
            return None
        if not text.startswith("<?xml") and not text.startswith("<root"):
            text = f"<root>{text}</root>"

        root = ET.fromstring(text)
        result = {}

        for field in fields:
            elem = root.find(f".//{field.name}")
            if elem is not None and elem.text:
                result[field.name] = elem.text.strip()
            elif field.name == "confidence":
                # Look for confidence attribute or element
                conf_elem = root.find(".//confidence")
                if conf_elem is not None and conf_elem.text:
                    result["confidence"] = conf_elem.text.strip()

        return result if result else None
    except ET.ParseError:
        return None


def extract_csv_row(text: str, fields: List[OutputField]) -> Optional[Dict]:
    """Extract data from CSV-formatted output."""
    if not text:
        return None

    # Sanitize first
    text = sanitize_llm_output(text)
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        text = text.strip()

    # Parse CSV
    import csv
    from io import StringIO

    try:
        reader = csv.reader(StringIO(text))
        for row in reader:
            if row:
                result = {}
                for i, field in enumerate(fields):
                    if i < len(row):
                        result[field.name] = row[i].strip().strip('"')
                    elif field.default is not None:
                        result[field.name] = field.default
                return result
    except csv.Error:
        pass

    return None


def validate_output(data: Dict, fields: List[OutputField]) -> Tuple[bool, List[str]]:
    """Validate output against field definitions."""
    errors = []

    for field in fields:
        if field.name not in data or data[field.name] is None:
            if field.required:
                if field.default is not None:
                    data[field.name] = field.default
                else:
                    errors.append(f"Missing required field: {field.name}")
            continue

        value = data[field.name]

        # Type validation
        if field.field_type == "number":
            try:
                data[field.name] = int(value) if isinstance(value, str) else value
            except (ValueError, TypeError):
                errors.append(f"Invalid number for {field.name}: {value}")

        elif field.field_type == "decimal":
            try:
                data[field.name] = float(value) if isinstance(value, str) else value
            except (ValueError, TypeError):
                errors.append(f"Invalid decimal for {field.name}: {value}")

        elif field.field_type == "boolean":
            if isinstance(value, str):
                data[field.name] = value.lower() in ("true", "yes", "1")
            elif not isinstance(value, bool):
                errors.append(f"Invalid boolean for {field.name}: {value}")

        elif field.field_type == "list":
            if isinstance(value, str):
                # Try to parse as JSON list or comma-separated
                try:
                    data[field.name] = json.loads(value)
                except json.JSONDecodeError:
                    data[field.name] = [v.strip() for v in value.split(",")]

        # Constraint validation
        if field.constraints and "one of:" in field.constraints.lower():
            allowed = [v.strip() for v in field.constraints.lower().split("one of:")[1].split(",")]
            if str(value).lower().strip() not in allowed:
                errors.append(f"Invalid value for {field.name}: '{value}' not in {allowed}")

    return len(errors) == 0, errors


def build_system_prompt(config: AutomatorConfig) -> str:
    """Build the system prompt from configuration."""
    if config.system_prompt_override:
        return config.system_prompt_override

    prompt_parts = [
        f"TASK: {config.task_description}",
        "",
        "OUTPUT SCHEMA:",
    ]

    for field in config.output_fields:
        req = "required" if field.required else "optional"
        constraint = f" ({field.constraints})" if field.constraints else ""
        default = f" [default: {field.default}]" if field.default else ""
        prompt_parts.append(f"- {field.name}: {field.field_type}, {req}{constraint}{default}")

    prompt_parts.append(f"\nOUTPUT FORMAT: {config.output_format.upper()}")

    # Format instructions
    if config.output_format == "json":
        prompt_parts.append("Return a valid JSON object with the fields specified above.")
    elif config.output_format == "csv":
        field_names = [f.name for f in config.output_fields]
        if config.confidence_enabled:
            field_names.append("confidence")
        prompt_parts.append(f"Return values as a single CSV row in this order: {', '.join(field_names)}")
        prompt_parts.append("Wrap text values containing commas in double quotes.")
    elif config.output_format == "xml":
        prompt_parts.append("Return XML with each field as an element.")
    else:  # plain text
        prompt_parts.append("Return plain text output.")

    # Few-shot examples
    if config.few_shot_examples:
        prompt_parts.append("\nEXAMPLES:")
        for i, ex in enumerate(config.few_shot_examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {json.dumps(ex.input_data)}")
            if config.output_format == "json":
                prompt_parts.append(f"Output: {json.dumps(ex.output_data)}")
            elif config.output_format == "csv":
                values = [str(ex.output_data.get(f.name, "")) for f in config.output_fields]
                prompt_parts.append(f"Output: {','.join(values)}")
            else:
                prompt_parts.append(f"Output: {json.dumps(ex.output_data)}")

    # Confidence
    if config.confidence_enabled:
        prompt_parts.append("\nCONFIDENCE: Include a 'confidence' field (0-100) indicating your certainty in the output.")

    # Consistency rules
    if config.consistency_prompt:
        prompt_parts.append(f"\nCONSISTENCY RULES:\n{config.consistency_prompt}")

    # Get critical rules from prompt registry (respects overrides)
    ensure_prompts_registered()
    critical_rules = get_effective_prompt("automator.critical_rules")
    prompt_parts.append(f"\n{critical_rules}")

    return "\n".join(prompt_parts)


def build_step_prompt(step: PipelineStep, confidence_enabled: bool = False) -> str:
    """Build the system prompt for a single pipeline step."""
    prompt_parts = [
        f"TASK: {step.task_description}",
        "",
        "OUTPUT SCHEMA:",
    ]

    for field in step.output_fields:
        req = "required" if field.required else "optional"
        constraint = f" ({field.constraints})" if field.constraints else ""
        default = f" [default: {field.default}]" if field.default else ""
        prompt_parts.append(f"- {field.name}: {field.field_type}, {req}{constraint}{default}")

    prompt_parts.append(f"\nOUTPUT FORMAT: {step.output_format.upper()}")

    # Format instructions
    if step.output_format == "json":
        prompt_parts.append("Return a valid JSON object with the fields specified above.")
    elif step.output_format == "csv":
        field_names = [f.name for f in step.output_fields]
        if confidence_enabled:
            field_names.append("confidence")
        prompt_parts.append(f"Return values as a single CSV row in this order: {', '.join(field_names)}")
        prompt_parts.append("Wrap text values containing commas in double quotes.")
    elif step.output_format == "xml":
        prompt_parts.append("Return XML with each field as an element.")
    else:  # plain text
        prompt_parts.append("Return plain text output.")

    # Confidence
    if confidence_enabled:
        prompt_parts.append("\nCONFIDENCE: Include a 'confidence' field (0-100) indicating your certainty in the output.")

    # Get critical rules from prompt registry (respects overrides)
    ensure_prompts_registered()
    critical_rules = get_effective_prompt("automator.critical_rules")
    prompt_parts.append(f"\n{critical_rules}")

    return "\n".join(prompt_parts)


class AutomatorTool(BaseTool):
    """General purpose batch processing tool"""

    id = "automator"
    name = "General Purpose Automator"
    description = "Apply any AI task across your dataset row by row"
    icon = ":material/precision_manufacturing:"
    category = "Processing"

    def __init__(self):
        self.db = get_db()

    def render_config(self) -> ToolConfig:
        """Render automator configuration UI"""

        # Initialize session state for dynamic fields
        if "automator_output_fields" not in st.session_state:
            st.session_state["automator_output_fields"] = [
                {"name": "result", "type": "text", "required": True, "constraints": "", "default": ""}
            ]
        if "automator_few_shot_examples" not in st.session_state:
            st.session_state["automator_few_shot_examples"] = []
        if "automator_pipeline_enabled" not in st.session_state:
            st.session_state["automator_pipeline_enabled"] = False
        if "automator_pipeline_steps" not in st.session_state:
            # Initialize with one empty step
            st.session_state["automator_pipeline_steps"] = [
                {
                    "name": "Step 1",
                    "task": "",
                    "input_fields": [],
                    "output_fields": [{"name": "result", "type": "text", "required": True, "constraints": ""}],
                    "output_format": "json"
                }
            ]

        # Section 1: Data Upload
        st.header("1. Upload Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls", "json"],
                key="automator_upload"
            )
        with col_sample:
            st.write("")
            use_sample = st.button("Use Sample Data", help="Load sample data for testing", key="automator_sample_btn")

        if use_sample:
            st.session_state["automator_use_sample"] = True

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

        elif st.session_state.get("automator_use_sample"):
            df = pd.DataFrame(SAMPLE_AUTOMATOR_DATA)
            data_source = "sample_automator_data.csv"
            st.success("Using sample data (10 news articles)")

        if df is None:
            return ToolConfig(is_valid=False, error_message="Please upload a file or use sample data")

        if df.empty:
            return ToolConfig(is_valid=False, error_message="The uploaded file is empty")

        # Data preview
        st.dataframe(df.head(5), height=180)
        st.caption(f"Total rows: {len(df):,} | Columns: {len(df.columns)}")

        # Section 2: Input Configuration
        st.header("2. Select Input Columns")

        all_cols = df.columns.tolist()
        input_columns = st.multiselect(
            "Columns to send to AI",
            all_cols,
            default=all_cols,
            key="automator_input_cols",
            help="Select which columns should be included in the prompt for each row"
        )

        if input_columns:
            st.caption("**Sample row data:**")
            sample_row = df[input_columns].iloc[0]
            for col in input_columns:
                value = sample_row[col]
                # Truncate long text
                display_val = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                st.text(f"• {col}: {display_val}")

        # Section 3: Define Pipeline Steps
        st.header("3. Define Pipeline Steps")

        pipeline_enabled = st.checkbox(
            "Enable Multi-Step Pipeline",
            value=st.session_state.get("automator_pipeline_enabled", False),
            key="automator_pipeline_toggle",
            help="Create a multi-step workflow where each step can use outputs from previous steps"
        )
        st.session_state["automator_pipeline_enabled"] = pipeline_enabled

        # Initialize variables for backward compatibility
        task_description = ""
        output_fields = []
        output_format = "json"
        system_prompt_override = ""

        if not pipeline_enabled:
            # Single-step mode (backward compatible)
            task_description = st.text_area(
                "Task Description",
                height=150,
                placeholder="Describe what you want the AI to do with each row.\n\nExamples:\n- Classify the sentiment as positive, negative, or neutral\n- Extract key entities (people, organizations, locations)\n- Summarize the text in 1-2 sentences\n- Translate to Spanish\n- Score the quality from 1-10",
                key="automator_task_desc"
            )

            with st.expander("Advanced: Custom System Prompt", expanded=False):
                system_prompt_override = st.text_area(
                    "System Prompt Override",
                    height=200,
                    placeholder="Leave empty to auto-generate from task description and output schema.\nProvide a complete system prompt to override the auto-generated one.",
                    key="automator_system_override"
                )

            # Output Schema for single-step mode
            st.subheader("Output Schema")

            output_fields = st.session_state["automator_output_fields"]

            # Display current fields
            fields_to_remove = []
            for idx, field_def in enumerate(output_fields):
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1, 2, 0.5])

                with col1:
                    output_fields[idx]["name"] = st.text_input(
                        "Field Name",
                        value=field_def.get("name", ""),
                        key=f"field_name_{idx}",
                        label_visibility="collapsed" if idx > 0 else "visible"
                    )

                with col2:
                    type_options = ["text", "number", "decimal", "boolean", "list"]
                    current_type = field_def.get("type", "text")
                    type_idx = type_options.index(current_type) if current_type in type_options else 0
                    output_fields[idx]["type"] = st.selectbox(
                        "Type",
                        type_options,
                        index=type_idx,
                        key=f"field_type_{idx}",
                        label_visibility="collapsed" if idx > 0 else "visible"
                    )

                with col3:
                    output_fields[idx]["required"] = st.checkbox(
                        "Required",
                        value=field_def.get("required", True),
                        key=f"field_required_{idx}",
                        label_visibility="collapsed" if idx > 0 else "visible"
                    )

                with col4:
                    output_fields[idx]["constraints"] = st.text_input(
                        "Constraints",
                        value=field_def.get("constraints", ""),
                        key=f"field_constraints_{idx}",
                        placeholder="e.g., one of: A, B, C",
                        label_visibility="collapsed" if idx > 0 else "visible"
                    )

                with col5:
                    if idx > 0:  # Don't allow removing the first field
                        if st.button(":material/delete:", key=f"remove_field_{idx}"):
                            fields_to_remove.append(idx)

            # Remove marked fields
            for idx in reversed(fields_to_remove):
                output_fields.pop(idx)
            st.session_state["automator_output_fields"] = output_fields

            # Add field button
            if st.button("+ Add Field", key="add_output_field"):
                output_fields.append({"name": "", "type": "text", "required": False, "constraints": "", "default": ""})
                st.session_state["automator_output_fields"] = output_fields
                st.rerun()

            # Output format selection
            output_format = st.radio(
                "Output Format",
                ["JSON", "CSV", "Plain Text", "XML"],
                horizontal=True,
                key="automator_output_format",
                help="JSON: Structured object | CSV: Comma-separated values | Plain Text: Raw output | XML: Tagged structure"
            )

        else:
            # Multi-step pipeline mode
            pipeline_steps = st.session_state["automator_pipeline_steps"]

            # Calculate available fields for each step (original columns + previous step outputs)
            def get_available_fields_for_step(step_idx: int) -> List[str]:
                """Get fields available as input for a given step."""
                available = list(all_cols)  # Start with original columns
                for i in range(step_idx):
                    # Add output fields from previous steps
                    for field in pipeline_steps[i].get("output_fields", []):
                        field_name = field.get("name", "").strip()
                        if field_name and field_name not in available:
                            available.append(field_name)
                return available

            # Render each step in expanders for cleaner UI
            steps_to_remove = []
            for step_idx, step in enumerate(pipeline_steps):
                step_name = step.get("name", f"Step {step_idx + 1}")
                # Build output names from output_fields list
                output_field_list = step.get("output_fields", [])
                output_names = ", ".join([f.get("name", "") for f in output_field_list if f.get("name", "")])
                step_label = f"Step {step_idx + 1}: {step_name}" if step_name != f"Step {step_idx + 1}" else f"Step {step_idx + 1}"
                if output_names:
                    step_label += f" → {output_names}"

                with st.expander(step_label, expanded=(step_idx == len(pipeline_steps) - 1)):
                    col_name, col_delete = st.columns([4, 1])
                    with col_name:
                        pipeline_steps[step_idx]["name"] = st.text_input(
                            "Step Name",
                            value=step.get("name", f"Step {step_idx + 1}"),
                            key=f"step_name_{step_idx}",
                            placeholder="e.g., Translate, Extract Codes, Summarize"
                        )
                    with col_delete:
                        if len(pipeline_steps) > 1:
                            st.write("")  # Spacing
                            if st.button("Delete", key=f"delete_step_{step_idx}", type="secondary"):
                                steps_to_remove.append(step_idx)

                    # Task description
                    pipeline_steps[step_idx]["task"] = st.text_area(
                        "Task Description",
                        value=step.get("task", ""),
                        height=80,
                        key=f"step_task_{step_idx}",
                        placeholder="Describe what this step should do..."
                    )

                    # Input fields selector
                    available_fields = get_available_fields_for_step(step_idx)
                    current_input_fields = step.get("input_fields", [])
                    valid_current = [f for f in current_input_fields if f in available_fields]

                    selected_inputs = st.multiselect(
                        "Input Fields",
                        available_fields,
                        default=valid_current if valid_current else available_fields[:min(3, len(available_fields))],
                        key=f"step_inputs_{step_idx}",
                        help="Fields to pass to this step"
                    )
                    pipeline_steps[step_idx]["input_fields"] = selected_inputs

                    # Row-based output field editor
                    st.markdown("**Output Fields**")
                    step_output_fields = step.get("output_fields", [])
                    if not step_output_fields:
                        step_output_fields = [{"name": "", "type": "text", "required": True, "constraints": ""}]
                        pipeline_steps[step_idx]["output_fields"] = step_output_fields

                    fields_to_remove_step = []
                    for field_idx, field_def in enumerate(step_output_fields):
                        col_name, col_type, col_delete = st.columns([3, 2, 0.5])

                        with col_name:
                            step_output_fields[field_idx]["name"] = st.text_input(
                                "Name" if field_idx == 0 else "Name ",
                                value=field_def.get("name", ""),
                                key=f"step_{step_idx}_field_name_{field_idx}",
                                placeholder="field name",
                                label_visibility="collapsed" if field_idx > 0 else "visible"
                            )

                        with col_type:
                            type_options = ["text", "number", "list"]
                            current_type = field_def.get("type", "text")
                            # Map old types to simplified options
                            if current_type in ["decimal", "boolean"]:
                                current_type = "text"
                            type_idx = type_options.index(current_type) if current_type in type_options else 0
                            step_output_fields[field_idx]["type"] = st.selectbox(
                                "Type" if field_idx == 0 else "Type ",
                                type_options,
                                index=type_idx,
                                key=f"step_{step_idx}_field_type_{field_idx}",
                                label_visibility="collapsed" if field_idx > 0 else "visible"
                            )

                        with col_delete:
                            if field_idx > 0:  # Don't allow removing the first field
                                if st.button("✕", key=f"step_{step_idx}_remove_field_{field_idx}"):
                                    fields_to_remove_step.append(field_idx)

                    # Remove marked fields
                    for idx in reversed(fields_to_remove_step):
                        step_output_fields.pop(idx)
                    pipeline_steps[step_idx]["output_fields"] = step_output_fields

                    # Add field button
                    if st.button("+ Add Field", key=f"step_{step_idx}_add_field"):
                        step_output_fields.append({"name": "", "type": "text", "required": True, "constraints": ""})
                        pipeline_steps[step_idx]["output_fields"] = step_output_fields
                        st.rerun()

                    # Keep output format as JSON by default (hidden complexity)
                    pipeline_steps[step_idx]["output_format"] = "json"

            # Remove marked steps
            for idx in reversed(steps_to_remove):
                pipeline_steps.pop(idx)
            st.session_state["automator_pipeline_steps"] = pipeline_steps

            # Add step button
            if st.button("+ Add Step", key="add_pipeline_step", type="primary"):
                new_step_num = len(pipeline_steps) + 1
                pipeline_steps.append({
                    "name": f"Step {new_step_num}",
                    "task": "",
                    "input_fields": [],
                    "output_fields": [{"name": "", "type": "text", "required": True, "constraints": ""}],
                    "output_format": "json"
                })
                st.session_state["automator_pipeline_steps"] = pipeline_steps
                st.rerun()

            # Pipeline flow visualization
            if len(pipeline_steps) > 0:
                st.caption("**Pipeline Flow:** " + " → ".join([
                    f"{i+1}. {s.get('name', f'Step {i+1}')}"
                    for i, s in enumerate(pipeline_steps)
                ]))

        # Section 4: Few-Shot Examples (only for single-step mode)
        few_shot_examples = []
        if not pipeline_enabled:
            st.header("4. Few-Shot Examples (Optional)")

            few_shot_examples = st.session_state["automator_few_shot_examples"]

            if few_shot_examples:
                for idx, example in enumerate(few_shot_examples):
                    with st.expander(f"Example {idx + 1}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Input:**")
                            st.json(example.get("input", {}))
                        with col2:
                            st.markdown("**Output:**")
                            st.json(example.get("output", {}))
                        if st.button("Remove", key=f"remove_example_{idx}"):
                            few_shot_examples.pop(idx)
                            st.session_state["automator_few_shot_examples"] = few_shot_examples
                            st.rerun()

            col_add, col_gen = st.columns(2)
            with col_add:
                if st.button("Add Example Manually", key="add_example_manual"):
                    st.session_state["automator_show_add_example"] = True

            with col_gen:
                if st.button("Generate from First Row", key="gen_example_from_row", disabled=not input_columns):
                    # Create example from first row
                    if input_columns and len(output_fields) > 0:
                        sample_input = df[input_columns].iloc[0].to_dict()
                        sample_output = {f["name"]: "" for f in output_fields if f["name"]}
                        few_shot_examples.append({"input": sample_input, "output": sample_output})
                        st.session_state["automator_few_shot_examples"] = few_shot_examples
                        st.session_state["automator_show_add_example"] = True
                        st.rerun()

            # Manual example input form
            if st.session_state.get("automator_show_add_example"):
                with st.container():
                    st.markdown("---")
                    st.markdown("**Add Example**")

                    # If we have a pending example, edit it
                    if few_shot_examples and not few_shot_examples[-1].get("output", {}).get(output_fields[0]["name"] if output_fields else "result"):
                        edit_idx = len(few_shot_examples) - 1
                        example = few_shot_examples[edit_idx]

                        st.markdown("**Input (from data):**")
                        st.json(example.get("input", {}))

                        st.markdown("**Expected Output:**")
                        updated_output = {}
                        for field_def in output_fields:
                            if field_def["name"]:
                                updated_output[field_def["name"]] = st.text_input(
                                    field_def["name"],
                                    value=example.get("output", {}).get(field_def["name"], ""),
                                    key=f"example_output_{field_def['name']}"
                                )

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save Example", key="save_example"):
                                few_shot_examples[edit_idx]["output"] = updated_output
                                st.session_state["automator_few_shot_examples"] = few_shot_examples
                                st.session_state["automator_show_add_example"] = False
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key="cancel_example"):
                                few_shot_examples.pop(edit_idx)
                                st.session_state["automator_few_shot_examples"] = few_shot_examples
                                st.session_state["automator_show_add_example"] = False
                                st.rerun()

        # Section 5: Quality Controls
        section_num = 5 if not pipeline_enabled else 4
        st.header(f"{section_num}. Quality Controls")

        col_conf, col_empty = st.columns(2)

        with col_conf:
            confidence_enabled = st.checkbox(
                "Enable Confidence Flagging",
                value=False,
                key="automator_confidence_enabled",
                help="AI will output a confidence score (0-100) per row"
            )

            if confidence_enabled:
                confidence_threshold = st.slider(
                    "Flag rows below threshold",
                    min_value=0,
                    max_value=100,
                    value=70,
                    key="automator_confidence_threshold",
                    help="Rows with confidence below this will be flagged for review"
                )
            else:
                confidence_threshold = 70

        with col_empty:
            handle_empty = st.selectbox(
                "Handle Empty Rows",
                ["skip", "default", "error"],
                key="automator_handle_empty",
                help="skip: Skip empty rows | default: Use default values | error: Mark as error"
            )

        with st.expander("Consistency Rules (Optional)", expanded=False):
            consistency_prompt = st.text_area(
                "Consistency Instructions",
                height=100,
                placeholder="Add rules to ensure consistent output across all rows.\n\nExample:\n- Always use lowercase for sentiment labels\n- Score scales from 1-10, where 5 is neutral\n- Extract at most 3 entities per row",
                key="automator_consistency_prompt"
            )

        # Validate configuration
        if pipeline_enabled:
            # Pipeline mode validation
            pipeline_steps = st.session_state["automator_pipeline_steps"]

            # Check each step has a task and at least one output field
            for i, step in enumerate(pipeline_steps):
                if not step.get("task", "").strip():
                    return ToolConfig(
                        is_valid=False,
                        error_message=f"Step {i+1}: Please enter a task description",
                        config_data={"df": df, "data_source": data_source, "input_columns": input_columns}
                    )
                step_output_fields = step.get("output_fields", [])
                valid_step_fields = [f for f in step_output_fields if f.get("name", "").strip()]
                if not valid_step_fields:
                    return ToolConfig(
                        is_valid=False,
                        error_message=f"Step {i+1}: Please define at least one output field with a name",
                        config_data={"df": df, "data_source": data_source, "input_columns": input_columns}
                    )
        else:
            # Single-step mode validation
            if not task_description or not task_description.strip():
                return ToolConfig(
                    is_valid=False,
                    error_message="Please enter a task description",
                    config_data={
                        "df": df,
                        "data_source": data_source,
                        "input_columns": input_columns
                    }
                )

            valid_fields = [f for f in output_fields if f.get("name", "").strip()]
            if not valid_fields:
                return ToolConfig(
                    is_valid=False,
                    error_message="Please define at least one output field with a name",
                    config_data={
                        "df": df,
                        "data_source": data_source,
                        "input_columns": input_columns,
                        "task_description": task_description
                    }
                )

        # Get provider settings
        provider = st.session_state.get("_automator_provider")
        api_key = st.session_state.get("_automator_api_key")
        base_url = st.session_state.get("_automator_base_url")
        model = st.session_state.get("model_name") or get_selected_model()

        if not provider:
            return ToolConfig(
                is_valid=False,
                error_message="No AI provider configured. Go to LLM Providers to set one up, or start LM Studio.",
                config_data={
                    "df": df,
                    "data_source": data_source,
                    "input_columns": input_columns,
                    "task_description": task_description
                }
            )

        if PROVIDER_CONFIGS[provider].requires_api_key and (not api_key or api_key == "dummy"):
            return ToolConfig(
                is_valid=False,
                error_message="Provider requires an API key. Configure it in LLM Providers page.",
                config_data={
                    "df": df,
                    "data_source": data_source,
                    "input_columns": input_columns,
                    "task_description": task_description
                }
            )

        # Build pipeline steps if in pipeline mode
        parsed_steps = []
        if pipeline_enabled:
            pipeline_steps = st.session_state["automator_pipeline_steps"]
            for i, step in enumerate(pipeline_steps):
                step_output_fields = [
                    OutputField(
                        name=f["name"],
                        field_type=f.get("type", "text"),
                        required=f.get("required", True),
                        constraints=f.get("constraints", "") or None,
                        default=f.get("default", "") or None
                    )
                    for f in step.get("output_fields", [])
                    if f.get("name", "").strip()
                ]
                parsed_steps.append(PipelineStep(
                    step_id=f"step_{i}",
                    name=step.get("name", f"Step {i+1}"),
                    task_description=step.get("task", ""),
                    input_fields=step.get("input_fields", []),
                    output_fields=step_output_fields,
                    output_format=step.get("output_format", "json")
                ))

        # Build output fields for single-step mode
        parsed_output_fields = []
        if not pipeline_enabled:
            parsed_output_fields = [
                OutputField(
                    name=f["name"],
                    field_type=f["type"],
                    required=f.get("required", True),
                    constraints=f.get("constraints", "") or None,
                    default=f.get("default", "") or None
                )
                for f in output_fields
                if f.get("name", "").strip()
            ]

        # Build few-shot examples
        parsed_examples = [
            FewShotExample(input_data=ex["input"], output_data=ex["output"])
            for ex in few_shot_examples
            if ex.get("input") and ex.get("output")
        ]

        # Build automator config
        automator_config = AutomatorConfig(
            task_description=task_description,
            input_columns=input_columns,
            output_fields=parsed_output_fields,
            output_format=output_format.lower().replace(" ", "_").replace("plain_text", "text") if isinstance(output_format, str) else "json",
            few_shot_examples=parsed_examples,
            confidence_enabled=confidence_enabled,
            confidence_threshold=confidence_threshold,
            consistency_prompt=consistency_prompt if consistency_prompt and consistency_prompt.strip() else None,
            handle_empty=handle_empty,
            system_prompt_override=system_prompt_override if system_prompt_override and system_prompt_override.strip() else None,
            steps=parsed_steps
        )

        config_data = {
            "df": df,
            "data_source": data_source,
            "automator_config": automator_config,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "max_concurrency": st.session_state.get("max_concurrency", 5),
            "auto_retry": st.session_state.get("auto_retry", True),
            "max_retries": st.session_state.get("max_retries", 3),
            "realtime_progress": st.session_state.get("realtime_progress", True),
            "test_batch_size": st.session_state.get("test_batch_size", 10),
        }

        return ToolConfig(is_valid=True, config_data=config_data)

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute the automator processing"""

        df = config["df"]
        automator_config: AutomatorConfig = config["automator_config"]
        is_test = config.get("is_test", False)
        test_batch_size = config.get("test_batch_size", 10)

        # Prepare target dataframe
        if is_test:
            target_df = df.head(test_batch_size).copy()
        else:
            target_df = df.copy()
        target_df = target_df.reset_index(drop=True)

        # Check if pipeline mode
        is_pipeline = len(automator_config.steps) > 0

        # Build system prompt (for single-step mode)
        system_prompt = build_system_prompt(automator_config) if not is_pipeline else ""

        # Create or get session
        session_id = get_current_session_id()
        if not session_id:
            session = self.db.create_session("automator", get_current_settings())
            session_id = session.session_id
            set_current_session_id(session_id)
            self.db.log(LogLevel.INFO, f"Created new session: {session.name}", session_id=session_id)

        run_type = "test" if is_test else "full"

        # Build schema for run
        if is_pipeline:
            schema = {
                "pipeline_steps": [
                    {
                        "step_id": step.step_id,
                        "name": step.name,
                        "output_fields": [asdict(f) for f in step.output_fields]
                    }
                    for step in automator_config.steps
                ]
            }
        else:
            schema = {"output_fields": [asdict(f) for f in automator_config.output_fields]}

        # Create run
        run = self.db.create_run(
            session_id=session_id,
            run_type=run_type,
            provider=config["provider"].value,
            model=config["model"],
            temperature=None,  # Use provider default
            max_tokens=None,   # Use provider default
            system_prompt=system_prompt if not is_pipeline else f"Pipeline with {len(automator_config.steps)} steps",
            schema=schema,
            variables={},
            input_file=config["data_source"],
            input_rows=len(target_df),
            json_mode=automator_config.output_format == "json" if not is_pipeline else True,
            max_concurrency=config["max_concurrency"],
            auto_retry=config["auto_retry"],
            max_retry_attempts=config["max_retries"],
            run_settings=get_current_settings()
        )

        mode_str = f"pipeline ({len(automator_config.steps)} steps)" if is_pipeline else "single-step"
        self.db.log(LogLevel.INFO, f"Started {run_type} automator run ({mode_str}) with {len(target_df)} rows",
                   {"provider": config["provider"].value, "model": config["model"]},
                   run_id=run.run_id, session_id=session_id)

        # Create HTTP client and LLM client
        http_client = create_http_client()
        client = get_client(
            config["provider"],
            config["api_key"],
            config["base_url"],
            http_client
        )

        # Determine JSON mode (for single-step)
        use_json_mode = automator_config.output_format == "json" and supports_json_mode(
            config["provider"], config["model"]
        )

        semaphore = asyncio.Semaphore(config["max_concurrency"])
        results_map = {}
        all_results = []

        total = len(target_df)
        completed = 0
        success_count = 0
        error_count = 0
        retry_count = 0
        flagged_count = 0
        partial_count = 0

        async def process_row_pipeline(row_idx: int, row: pd.Series) -> Tuple[int, Dict, bool, RunResult]:
            """Process a single row through the multi-step pipeline."""
            async with semaphore:
                # Initialize pipeline context
                original_data = row.to_dict()
                cumulative_data = original_data.copy()
                step_outputs: Dict[str, Dict] = {}
                step_latencies: Dict[str, float] = {}
                step_errors: Dict[str, str] = {}
                total_latency = 0.0
                total_attempts = 0
                row_status = "success"

                # Check for empty data
                if automator_config.input_columns:
                    check_data = {k: v for k, v in original_data.items() if k in automator_config.input_columns}
                else:
                    check_data = original_data

                is_empty = all(
                    pd.isna(v) or (isinstance(v, str) and not v.strip())
                    for v in check_data.values()
                )

                if is_empty:
                    if automator_config.handle_empty == "skip":
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=json.dumps(original_data),
                            output="SKIPPED: Empty row",
                            status=ResultStatus.SUCCESS,
                            latency=0.0,
                            retry_attempt=0
                        )
                        return row_idx, {"_automator_status": "skipped", "_automator_latency": 0.0}, True, result
                    elif automator_config.handle_empty == "error":
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=json.dumps(original_data),
                            output="Error: Empty row",
                            status=ResultStatus.ERROR,
                            latency=0.0,
                            error_type="empty_row",
                            error_message="Row contains no data",
                            retry_attempt=0
                        )
                        return row_idx, {"_automator_status": "error", "_automator_error": "Empty row", "_automator_latency": 0.0}, False, result

                # Process each step
                for step in automator_config.steps:
                    step_start = time.time()

                    # Build input for this step from cumulative data
                    step_input = {}
                    for field_name in step.input_fields:
                        if field_name in cumulative_data:
                            step_input[field_name] = cumulative_data[field_name]

                    # Build step-specific prompt
                    step_prompt = build_step_prompt(step, automator_config.confidence_enabled)

                    # Determine if JSON mode for this step
                    step_json_mode = step.output_format == "json" and supports_json_mode(
                        config["provider"], config["model"]
                    )

                    user_content = f"Data: {json.dumps(step_input)}"

                    # Call LLM for this step
                    output, duration, error_info, attempts = await call_llm_with_retry(
                        client, step_prompt, user_content, config["model"],
                        None, None, step_json_mode,
                        run.run_id, row_idx,
                        config["max_retries"] if config["auto_retry"] else 0,
                        self.db,
                        config["provider"]
                    )

                    step_latencies[step.step_id] = round(duration, 3)
                    total_latency += duration
                    total_attempts += attempts

                    if error_info:
                        step_errors[step.step_id] = error_info.message
                        row_status = "partial"
                        continue  # Try next step

                    # Parse output based on step's format
                    parsed_data = None
                    parse_error = None

                    if step.output_format == "json":
                        parsed_data = extract_json(output)
                        if not parsed_data:
                            parse_error = "Failed to parse JSON output"
                    elif step.output_format == "csv":
                        parsed_data = extract_csv_row(output, step.output_fields)
                        if not parsed_data:
                            parse_error = "Failed to parse CSV output"
                    else:  # text
                        parsed_data = {"text": sanitize_llm_output(output)}

                    if parse_error:
                        step_errors[step.step_id] = parse_error
                        row_status = "partial"
                        continue

                    # Validate output
                    if parsed_data and step.output_format != "text":
                        is_valid, validation_errors = validate_output(parsed_data, step.output_fields)
                        if not is_valid:
                            step_errors[step.step_id] = "; ".join(validation_errors)
                            row_status = "partial"
                            # Still use partial data
                            step_outputs[step.step_id] = parsed_data
                            cumulative_data.update(parsed_data)
                            continue

                    # Store step results
                    step_outputs[step.step_id] = parsed_data
                    cumulative_data.update(parsed_data)

                # Build final result dict
                result_dict = {}

                # Add all step outputs
                for step in automator_config.steps:
                    step_data = step_outputs.get(step.step_id, {})
                    for field in step.output_fields:
                        result_dict[field.name] = step_data.get(field.name, "")

                # Add metadata
                result_dict["_automator_status"] = row_status
                result_dict["_automator_latency"] = round(total_latency, 3)

                # Add per-step latencies
                for step in automator_config.steps:
                    result_dict[f"_{step.step_id}_latency"] = step_latencies.get(step.step_id, 0)

                # Add step errors if any
                if step_errors:
                    result_dict["_automator_step_errors"] = json.dumps(step_errors)

                # Determine overall success
                success = row_status == "success"
                if row_status == "partial":
                    # Partial is still considered a "success" for counting purposes
                    # but we track it separately
                    success = True

                result = RunResult.create(
                    run_id=run.run_id,
                    row_index=row_idx,
                    input_data=json.dumps(original_data),
                    output=json.dumps({k: v for k, v in result_dict.items() if not k.startswith("_")}),
                    status=ResultStatus.SUCCESS if success else ResultStatus.ERROR,
                    latency=total_latency,
                    error_type="step_errors" if step_errors and not success else None,
                    error_message=json.dumps(step_errors) if step_errors and not success else None,
                    retry_attempt=total_attempts
                )

                return row_idx, result_dict, success, result

        async def process_row(row_idx: int, row: pd.Series) -> Tuple[int, Dict, bool, RunResult]:
            """Process a single row in single-step mode."""
            async with semaphore:
                # Build row data from selected columns
                if automator_config.input_columns:
                    row_data = row[automator_config.input_columns].to_dict()
                else:
                    row_data = row.to_dict()

                # Check for empty data
                is_empty = all(
                    pd.isna(v) or (isinstance(v, str) and not v.strip())
                    for v in row_data.values()
                )

                if is_empty:
                    if automator_config.handle_empty == "skip":
                        # Return skip result
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=json.dumps(row_data),
                            output="SKIPPED: Empty row",
                            status=ResultStatus.SUCCESS,
                            latency=0.0,
                            retry_attempt=0
                        )
                        return row_idx, {
                            "_automator_status": "skipped",
                            "_automator_latency": 0.0
                        }, True, result

                    elif automator_config.handle_empty == "default":
                        # Use defaults
                        output_data = {}
                        for field in automator_config.output_fields:
                            output_data[field.name] = field.default or ""
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=json.dumps(row_data),
                            output=json.dumps(output_data),
                            status=ResultStatus.SUCCESS,
                            latency=0.0,
                            retry_attempt=0
                        )
                        return row_idx, {
                            **output_data,
                            "_automator_status": "default",
                            "_automator_latency": 0.0
                        }, True, result

                    else:  # error
                        result = RunResult.create(
                            run_id=run.run_id,
                            row_index=row_idx,
                            input_data=json.dumps(row_data),
                            output="Error: Empty row",
                            status=ResultStatus.ERROR,
                            latency=0.0,
                            error_type="empty_row",
                            error_message="Row contains no data",
                            retry_attempt=0
                        )
                        return row_idx, {
                            "_automator_status": "error",
                            "_automator_error": "Empty row",
                            "_automator_latency": 0.0
                        }, False, result

                user_content = f"Data: {json.dumps(row_data)}"

                output, duration, error_info, attempts = await call_llm_with_retry(
                    client, system_prompt, user_content, config["model"],
                    None, None, use_json_mode,  # Use provider defaults
                    run.run_id, row_idx,
                    config["max_retries"] if config["auto_retry"] else 0,
                    self.db,
                    config["provider"]
                )

                if error_info:
                    result = RunResult.create(
                        run_id=run.run_id,
                        row_index=row_idx,
                        input_data=json.dumps(row_data),
                        output=f"Error: {error_info.message}",
                        status=ResultStatus.ERROR,
                        latency=duration,
                        error_type=error_info.error_type.value,
                        error_message=error_info.original_error,
                        retry_attempt=attempts
                    )
                    return row_idx, {
                        "_automator_status": "error",
                        "_automator_error": error_info.message,
                        "_automator_latency": round(duration, 3)
                    }, False, result

                # Parse output based on format
                parsed_data = None
                parse_error = None

                if automator_config.output_format == "json":
                    parsed_data = extract_json(output)
                    if not parsed_data:
                        parse_error = "Failed to parse JSON output"

                elif automator_config.output_format == "csv":
                    fields_with_conf = list(automator_config.output_fields)
                    if automator_config.confidence_enabled:
                        fields_with_conf.append(OutputField(name="confidence", field_type="number"))
                    parsed_data = extract_csv_row(output, fields_with_conf)
                    if not parsed_data:
                        parse_error = "Failed to parse CSV output"

                elif automator_config.output_format == "xml":
                    fields_with_conf = list(automator_config.output_fields)
                    if automator_config.confidence_enabled:
                        fields_with_conf.append(OutputField(name="confidence", field_type="number"))
                    parsed_data = extract_xml(output, fields_with_conf)
                    if not parsed_data:
                        parse_error = "Failed to parse XML output"

                else:  # plain text
                    parsed_data = {"text": sanitize_llm_output(output)}

                if parse_error:
                    # Try once more to extract any structure
                    if not parsed_data:
                        parsed_data = {"raw_output": output}

                # Validate output
                if parsed_data and automator_config.output_format != "text":
                    is_valid, validation_errors = validate_output(
                        parsed_data, automator_config.output_fields
                    )
                    if not is_valid:
                        parse_error = "; ".join(validation_errors)

                # Check confidence
                status = "success"
                if automator_config.confidence_enabled and parsed_data:
                    confidence = parsed_data.get("confidence", 100)
                    try:
                        confidence = int(confidence)
                    except (ValueError, TypeError):
                        confidence = 0

                    if confidence < automator_config.confidence_threshold:
                        status = "low_confidence"

                result = RunResult.create(
                    run_id=run.run_id,
                    row_index=row_idx,
                    input_data=json.dumps(row_data),
                    output=json.dumps(parsed_data) if parsed_data else output,
                    status=ResultStatus.SUCCESS if not parse_error else ResultStatus.ERROR,
                    latency=duration,
                    error_type="parse_error" if parse_error else None,
                    error_message=parse_error,
                    retry_attempt=attempts
                )

                result_dict = {
                    **(parsed_data or {}),
                    "_automator_status": status,
                    "_automator_latency": round(duration, 3),
                }
                if parse_error:
                    result_dict["_automator_error"] = parse_error
                    result_dict["_automator_raw_output"] = output

                return row_idx, result_dict, not parse_error, result

        # Create tasks based on mode
        if is_pipeline:
            tasks = [process_row_pipeline(i, row) for i, row in target_df.iterrows()]
        else:
            tasks = [process_row(i, row) for i, row in target_df.iterrows()]

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
                status = res_dict.get("_automator_status", "success")
                if status == "low_confidence":
                    flagged_count += 1
                elif status == "partial":
                    partial_count += 1
                log_entry = f"Row {idx}: {res_dict.get('_automator_latency', 0)}s ({status})"
            else:
                error_count += 1
                log_entry = f"Row {idx}: {res_dict.get('_automator_error', 'Error')}"

            # Call progress callback
            if progress_callback:
                should_update = (
                    config.get("realtime_progress", True) or
                    completed % 10 == 0 or
                    completed == total
                )
                if should_update:
                    progress_callback(completed, total, success_count, error_count,
                                     retry_count, log_entry, not success)

        # Save all results to database
        self.db.save_results_batch(all_results)

        await http_client.aclose()

        # Update run status
        latencies = [r.latency for r in all_results if r.status == ResultStatus.SUCCESS.value]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_duration = time.time() - start_time

        self.db.update_run_status(
            run.run_id, RunStatus.COMPLETED,
            success_count=success_count,
            error_count=error_count,
            retry_count=retry_count,
            avg_latency=avg_latency,
            total_duration=total_duration
        )

        self.db.log(LogLevel.INFO, f"Run completed: {success_count} success, {error_count} errors, {flagged_count} flagged, {partial_count} partial",
                   {"duration": total_duration, "avg_latency": avg_latency},
                   run_id=run.run_id)

        # Build results dataframe
        sorted_results = [results_map[i] for i in sorted(results_map.keys())]

        # Add results to dataframe based on mode
        if is_pipeline:
            # Add all output fields from all steps
            for step in automator_config.steps:
                for field in step.output_fields:
                    target_df[field.name] = [r.get(field.name, "") for r in sorted_results]
            # Add per-step latencies
            for step in automator_config.steps:
                target_df[f"_{step.step_id}_latency"] = [r.get(f"_{step.step_id}_latency", 0) for r in sorted_results]
        else:
            # Single-step mode
            for field in automator_config.output_fields:
                target_df[field.name] = [r.get(field.name, "") for r in sorted_results]

        # Add metadata columns
        target_df["_automator_status"] = [r.get("_automator_status", "unknown") for r in sorted_results]
        target_df["_automator_latency"] = [r.get("_automator_latency", 0) for r in sorted_results]

        if automator_config.confidence_enabled:
            target_df["_automator_confidence"] = [r.get("confidence", None) for r in sorted_results]

        # Build stats
        stats = {
            "success": success_count,
            "errors": error_count,
            "flagged": flagged_count,
            "retries": retry_count,
            "avg_latency": f"{avg_latency:.2f}s",
            "duration": f"{total_duration:.1f}s"
        }
        if is_pipeline:
            stats["partial"] = partial_count
            stats["steps"] = len(automator_config.steps)

        return ToolResult(
            success=True,
            data=target_df,
            stats=stats
        )

    def render_results(self, result: ToolResult):
        """Render automator results with filtering"""
        from ui.components.download_buttons import render_download_buttons

        if not result.success:
            st.error(f"Automator failed: {result.error_message}")
            return

        stats = result.stats
        df = result.data
        is_pipeline = stats.get("steps", 0) > 0

        # Success message with stats
        if is_pipeline:
            st.success(
                f"Complete! ({stats.get('steps', 0)} steps) "
                f"✓ {stats.get('success', 0)} | "
                f"✗ {stats.get('errors', 0)} | "
                f"◐ {stats.get('partial', 0)} partial | "
                f"↻ {stats.get('retries', 0)} retries | "
                f"Avg: {stats.get('avg_latency', '0s')} | "
                f"Total: {stats.get('duration', '0s')}"
            )
        else:
            st.success(
                f"Complete! "
                f"✓ {stats.get('success', 0)} | "
                f"✗ {stats.get('errors', 0)} | "
                f"⚑ {stats.get('flagged', 0)} flagged | "
                f"↻ {stats.get('retries', 0)} retries | "
                f"Avg: {stats.get('avg_latency', '0s')} | "
                f"Total: {stats.get('duration', '0s')}"
            )

        # Filter options
        st.subheader("Results")

        if "_automator_status" in df.columns:
            status_counts = df["_automator_status"].value_counts().to_dict()

            filter_options = ["All"]
            if status_counts.get("success", 0) > 0:
                filter_options.append(f"Success ({status_counts.get('success', 0)})")
            if status_counts.get("partial", 0) > 0:
                filter_options.append(f"Partial ({status_counts.get('partial', 0)})")
            if status_counts.get("error", 0) > 0:
                filter_options.append(f"Errors ({status_counts.get('error', 0)})")
            if status_counts.get("low_confidence", 0) > 0:
                filter_options.append(f"Low Confidence ({status_counts.get('low_confidence', 0)})")
            if status_counts.get("skipped", 0) > 0:
                filter_options.append(f"Skipped ({status_counts.get('skipped', 0)})")

            filter_choice = st.radio(
                "Filter results",
                filter_options,
                horizontal=True,
                key="automator_result_filter"
            )

            # Apply filter
            if "Success" in filter_choice and "Partial" not in filter_choice:
                display_df = df[df["_automator_status"] == "success"]
            elif "Partial" in filter_choice:
                display_df = df[df["_automator_status"] == "partial"]
            elif "Errors" in filter_choice:
                display_df = df[df["_automator_status"] == "error"]
            elif "Low Confidence" in filter_choice:
                display_df = df[df["_automator_status"] == "low_confidence"]
            elif "Skipped" in filter_choice:
                display_df = df[df["_automator_status"] == "skipped"]
            else:
                display_df = df
        else:
            display_df = df

        # Data preview
        st.dataframe(display_df, use_container_width=True)

        # Result inspector
        st.divider()
        st.subheader("Result Inspector")

        if len(display_df) > 0:
            col1, col2 = st.columns([1, 3])

            with col1:
                row_to_inspect = st.number_input(
                    "Row",
                    min_value=0,
                    max_value=max(0, len(display_df) - 1),
                    value=0,
                    key="automator_inspect_row"
                )

            with col2:
                view_options = ["Output Fields", "Full Row", "Raw Output"]
                if is_pipeline:
                    view_options = ["Output Fields", "Step Details", "Full Row"]
                view_mode = st.radio(
                    "View",
                    view_options,
                    horizontal=True,
                    key="automator_view_mode"
                )

            if row_to_inspect < len(display_df):
                sel_row = display_df.iloc[row_to_inspect]

                # Show status
                status = sel_row.get("_automator_status", "unknown")
                if status == "success":
                    st.success(f"Status: Success | Latency: {sel_row.get('_automator_latency', 0)}s")
                elif status == "partial":
                    st.warning(f"Status: Partial (some steps failed) | Latency: {sel_row.get('_automator_latency', 0)}s")
                elif status == "low_confidence":
                    conf = sel_row.get("_automator_confidence", "N/A")
                    st.warning(f"Status: Low Confidence ({conf}) | Latency: {sel_row.get('_automator_latency', 0)}s")
                elif status == "error":
                    st.error(f"Status: Error | {sel_row.get('_automator_error', 'Unknown error')}")
                elif status == "skipped":
                    st.info("Status: Skipped (empty row)")

                if view_mode == "Output Fields":
                    # Show only output fields (exclude internal columns)
                    output_cols = [c for c in display_df.columns if not c.startswith("_")]
                    st.json(sel_row[output_cols].to_dict())

                elif view_mode == "Step Details" and is_pipeline:
                    # Show per-step latencies and errors
                    step_cols = [c for c in display_df.columns if c.startswith("_step_")]
                    step_latencies = {c.replace("_", "").replace("_latency", ""): sel_row.get(c, 0) for c in step_cols if "_latency" in c}

                    st.markdown("**Step Latencies:**")
                    for step_name, latency in step_latencies.items():
                        st.text(f"  {step_name}: {latency}s")

                    # Show step errors if any
                    step_errors_str = sel_row.get("_automator_step_errors", None)
                    if step_errors_str and step_errors_str != "":
                        try:
                            step_errors = json.loads(step_errors_str)
                            if step_errors:
                                st.markdown("**Step Errors:**")
                                for step_id, error_msg in step_errors.items():
                                    st.error(f"  {step_id}: {error_msg}")
                        except (json.JSONDecodeError, TypeError):
                            pass

                elif view_mode == "Full Row":
                    st.json(sel_row.to_dict())

                else:  # Raw Output
                    raw = sel_row.get("_automator_raw_output", "N/A")
                    if raw != "N/A":
                        st.code(raw, language="text")
                    else:
                        st.info("No raw output available (parsing was successful)")

        # Download buttons
        st.divider()
        render_download_buttons(df, filename_prefix="automator_results", key_prefix="automator_dl")
