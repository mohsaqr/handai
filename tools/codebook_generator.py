"""
Codebook Generator Tool
Analyze qualitative data to automatically generate a codebook with themes, definitions, and examples
"""

import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict

from .base import BaseTool, ToolConfig, ToolResult
from core.providers import LLMProvider, PROVIDER_CONFIGS, supports_json_mode
from core.llm_client import get_client, call_llm_simple
from core.prompt_registry import get_effective_prompt, ensure_prompts_registered
from ui.state import (
    get_selected_provider, get_effective_api_key, get_selected_model
)
from core.sample_data import get_sample_data, get_dataset_info

# Sample qualitative data for testing
def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from text that may contain extra content around it."""
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    # Look for ```json ... ``` blocks first
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


SAMPLE_QUALITATIVE_DATA = {
    "response": [
        "The online learning platform made it easy to access materials anytime. I appreciated the flexibility but sometimes felt isolated from other students.",
        "I found the group discussions very engaging. Hearing different perspectives helped me understand the material better, though some sessions ran too long.",
        "The professor's feedback was always constructive and timely. I felt supported throughout the course even when struggling with difficult concepts.",
        "Technical issues with the video conferencing were frustrating. When the technology worked, the experience was good, but reliability was a problem.",
        "I loved the hands-on projects. They helped me apply what I learned to real-world situations. The theory lectures were less engaging.",
        "The workload was overwhelming at times. I wish there was better balance between readings, assignments, and exams throughout the semester.",
        "Office hours were very helpful for clarifying confusing topics. The professor was approachable and took time to explain things multiple ways.",
        "Collaboration with classmates was the highlight. We formed study groups that continued beyond the course. The social aspect enhanced learning.",
        "The course materials were outdated in some areas. I had to supplement with external resources to get current industry perspectives.",
        "Self-paced modules allowed me to learn at my own speed. However, I sometimes procrastinated without regular deadlines pushing me forward.",
        "The assessment methods seemed fair and aligned with learning objectives. I appreciated having multiple ways to demonstrate my understanding.",
        "Navigating the learning management system was confusing initially. Once I figured it out, accessing resources became straightforward.",
        "Guest speakers from industry provided valuable real-world insights. These sessions were among the most memorable parts of the course.",
        "I struggled with the lack of immediate feedback on assignments. Waiting weeks for grades made it hard to know if I was on track.",
        "The course built a strong foundation in the subject. I feel confident applying these skills in my future career.",
    ],
    "participant_id": [f"P{i:03d}" for i in range(1, 16)],
    "context": ["online", "in-person", "online", "hybrid", "in-person",
                "online", "in-person", "hybrid", "online", "online",
                "in-person", "online", "hybrid", "online", "in-person"]
}


@dataclass
class Code:
    """A single code in the codebook"""
    name: str
    definition: str
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    parent_category: str = ""
    is_borderline: bool = False
    borderline_notes: Optional[str] = None


@dataclass
class Codebook:
    """Complete codebook structure"""
    title: str
    approach: str  # inductive/deductive/hybrid
    codes: List[Code] = field(default_factory=list)
    categories: Dict[str, List[str]] = field(default_factory=dict)  # category -> code names
    data_source: str = ""
    total_items_analyzed: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            "title": self.title,
            "approach": self.approach,
            "codes": [asdict(code) for code in self.codes],
            "categories": self.categories,
            "data_source": self.data_source,
            "total_items_analyzed": self.total_items_analyzed
        }

    def to_markdown(self) -> str:
        """Convert to Markdown documentation format"""
        lines = [
            f"# {self.title}",
            "",
            f"**Approach:** {self.approach.title()}",
            f"**Data Source:** {self.data_source}",
            f"**Items Analyzed:** {self.total_items_analyzed}",
            "",
            "---",
            ""
        ]

        # Group codes by category
        for category, code_names in self.categories.items():
            lines.append(f"## {category}")
            lines.append("")

            for code_name in code_names:
                code = next((c for c in self.codes if c.name == code_name), None)
                if not code:
                    continue

                borderline_marker = " :warning:" if code.is_borderline else ""
                lines.append(f"### {code.name}{borderline_marker}")
                lines.append("")
                lines.append(f"**Definition:** {code.definition}")
                lines.append("")

                if code.inclusion_criteria:
                    lines.append("**Apply when:**")
                    for criterion in code.inclusion_criteria:
                        lines.append(f"- {criterion}")
                    lines.append("")

                if code.exclusion_criteria:
                    lines.append("**Do NOT apply when:**")
                    for criterion in code.exclusion_criteria:
                        lines.append(f"- {criterion}")
                    lines.append("")

                if code.examples:
                    lines.append("**Examples:**")
                    for example in code.examples:
                        lines.append(f'> "{example}"')
                        lines.append("")

                if code.is_borderline and code.borderline_notes:
                    lines.append(f"**Borderline Notes:** {code.borderline_notes}")
                    lines.append("")

                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def to_csv_rows(self) -> List[Dict]:
        """Convert to flat CSV format"""
        rows = []
        for code in self.codes:
            rows.append({
                "code_name": code.name,
                "definition": code.definition,
                "parent_category": code.parent_category,
                "inclusion_criteria": "; ".join(code.inclusion_criteria),
                "exclusion_criteria": "; ".join(code.exclusion_criteria),
                "examples": " | ".join(code.examples),
                "is_borderline": code.is_borderline,
                "borderline_notes": code.borderline_notes or ""
            })
        return rows

    def to_qualitative_coder_prompt(self) -> str:
        """Generate system prompt for Qualitative Coder module"""
        lines = [
            "You are a qualitative coder. Apply the following codebook to each text.",
            "",
            "## Codebook",
            ""
        ]

        # Group by category
        for category, code_names in self.categories.items():
            lines.append(f"### Category: {category}")
            lines.append("")

            for code_name in code_names:
                code = next((c for c in self.codes if c.name == code_name), None)
                if not code:
                    continue

                lines.append(f"**{code.name}**")
                lines.append(f"Definition: {code.definition}")

                if code.inclusion_criteria:
                    lines.append(f"Apply when: {'; '.join(code.inclusion_criteria)}")

                if code.exclusion_criteria:
                    lines.append(f"Do NOT apply when: {'; '.join(code.exclusion_criteria)}")

                lines.append("")

        lines.extend([
            "## Instructions",
            "For each text, identify ALL applicable codes from the codebook above.",
            "Return the codes as a comma-separated list.",
            "",
            "Example output format:",
            "code1, code2, code3",
            "",
            "If no codes apply, return: none",
            "Only use codes from the codebook - do not create new codes."
        ])

        return "\n".join(lines)


class CodebookGeneratorTool(BaseTool):
    """Tool for generating qualitative codebooks with AI"""

    id = "codebook-generator"
    name = "Codebook Generator"
    description = "Generate qualitative codebooks with AI"
    icon = ":material/book:"
    category = "Analysis"

    def render_config(self) -> ToolConfig:
        """Render codebook generator configuration UI"""

        # Section 1: Data Upload
        st.header("1. Upload Data")

        col_upload, col_sample = st.columns([3, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=["csv", "xlsx", "xls", "json"],
                key="codebook_upload",
                help="Upload qualitative data (interviews, surveys, observations)"
            )
        with col_sample:
            st.write("")
            use_sample = st.button(
                "Use Sample Data",
                help="Load sample qualitative data for testing",
                key="codebook_sample_btn"
            )

        if use_sample:
            st.session_state["codebook_use_sample"] = True

        # Sample data selector
        if st.session_state.get("codebook_use_sample"):
            sample_options = {
                "learning_experience": "Learning Experience (20 student responses)",
                "healthcare_interviews": "Healthcare Interviews (15 worker experiences)",
                "exit_interviews": "Exit Interviews (15 employee departures)",
            }
            selected_sample = st.selectbox(
                "Choose sample dataset",
                options=list(sample_options.keys()),
                format_func=lambda x: sample_options[x],
                key="codebook_sample_choice"
            )

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

        elif st.session_state.get("codebook_use_sample"):
            selected = st.session_state.get("codebook_sample_choice", "learning_experience")
            df = pd.DataFrame(get_sample_data(selected))
            info = get_dataset_info()[selected]
            data_source = f"{selected}.csv"
            st.success(f"Using sample data: {info['name']} ({info['rows']} rows)")

        if df is None:
            return ToolConfig(is_valid=False, error_message="Please upload a file or use sample data")

        if df.empty:
            return ToolConfig(is_valid=False, error_message="The uploaded file is empty")

        # Large file handling
        total_rows = len(df)
        is_large_file = total_rows > 500

        if is_large_file:
            st.info(f"Large dataset detected ({total_rows:,} rows). A sample will be used for analysis.")

        # Data preview (limit to 5 rows for display)
        st.dataframe(df.head(5), height=150)
        st.caption(f"Total rows: {total_rows:,} | Columns: {len(df.columns)}")

        # Column selection - more prominent
        st.header("2. Select Text Field")

        all_cols = df.columns.tolist()

        # Try to auto-detect text columns (longer average length)
        text_col_candidates = []
        for col in all_cols:
            if df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 50:  # Likely a text field
                    text_col_candidates.append(col)

        # Default to first text candidate, or first column
        default_idx = 0
        if text_col_candidates:
            default_idx = all_cols.index(text_col_candidates[0])

        text_column = st.selectbox(
            "Which column contains the text to analyze?",
            options=all_cols,
            index=default_idx,
            key="codebook_text_column",
            help="Select the column containing qualitative text data (interviews, responses, etc.)"
        )

        # Preview selected column
        if text_column:
            st.markdown(f"**Preview of `{text_column}`:**")
            sample_texts = df[text_column].dropna().head(3).tolist()
            for i, text in enumerate(sample_texts, 1):
                preview = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
                st.markdown(f"> {i}. *{preview}*")

        # Section 3: Coding Approach
        st.header("3. Coding Approach")

        approach = st.radio(
            "Select Approach",
            options=["Inductive", "Deductive", "Hybrid"],
            key="codebook_approach",
            help=(
                "**Inductive:** AI discovers themes from the data\n\n"
                "**Deductive:** You provide initial themes, AI maps and refines\n\n"
                "**Hybrid:** Combines both approaches"
            ),
            horizontal=True
        )

        initial_themes = ""
        if approach in ["Deductive", "Hybrid"]:
            initial_themes = st.text_area(
                "Initial Themes/Codes",
                key="codebook_initial_themes",
                height=150,
                help="Enter your initial themes or codes (one per line)",
                placeholder="e.g.,\nPositive Experience\nTechnical Issues\nLearning Outcomes\nSocial Interaction"
            )

        # Custom instructions for guiding the analysis
        with st.expander("Advanced: Custom Instructions", expanded=False):
            custom_instructions = st.text_area(
                "Additional Analysis Instructions",
                key="codebook_custom_instructions",
                height=200,
                help="Provide additional context to guide the analysis",
                placeholder="""Examples of what you can add:

THEORETICAL FRAMEWORK:
- Use Grounded Theory approach
- Apply Self-Determination Theory lens
- Focus on Thematic Analysis per Braun & Clarke

FOCUS AREAS:
- Pay special attention to emotional language
- Identify power dynamics in responses
- Look for barriers and enablers

DOMAIN CONTEXT:
- This is healthcare worker feedback
- Participants are first-generation students
- Data is from exit interviews

EXAMPLE CODES:
- "Autonomy" = feeling of control over decisions
- "Burnout" = exhaustion, cynicism, reduced efficacy"""
            )

        # Get custom instructions (empty string if not provided)
        custom_instructions = st.session_state.get("codebook_custom_instructions", "")

        col_codes, col_granularity = st.columns(2)
        with col_codes:
            num_codes = st.slider(
                "Number of Codes to Generate",
                min_value=5,
                max_value=30,
                value=10,
                key="codebook_num_codes",
                help="Target number of codes for the codebook"
            )
        with col_granularity:
            granularity = st.select_slider(
                "Granularity Level",
                options=["Low", "Medium", "High"],
                value="Medium",
                key="codebook_granularity",
                help="Low = broader themes, High = more specific codes"
            )

        # Section 4: Output Options
        st.header("4. Output Options")

        # Chunked processing for large files
        if is_large_file:
            st.subheader("Large File Processing")
            chunk_size = st.slider(
                "Rows per Chunk",
                min_value=50,
                max_value=200,
                value=100,
                step=25,
                key="codebook_chunk_size",
                help="Process data in chunks of this size"
            )
            max_chunks = st.slider(
                "Maximum Chunks to Process",
                min_value=1,
                max_value=min(10, (total_rows // chunk_size) + 1),
                value=min(5, (total_rows // chunk_size) + 1),
                key="codebook_max_chunks",
                help="More chunks = better coverage but slower. Themes are merged across chunks."
            )
            st.caption(f"Will analyze up to {chunk_size * max_chunks:,} rows across {max_chunks} chunks")
        else:
            chunk_size = total_rows
            max_chunks = 1

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            include_examples = st.checkbox(
                "Include Examples",
                value=True,
                key="codebook_include_examples",
                help="Include example quotes from data for each code"
            )
            if include_examples:
                examples_per_code = st.slider(
                    "Examples per Code",
                    min_value=1,
                    max_value=5,
                    value=2,
                    key="codebook_examples_count"
                )
            else:
                examples_per_code = 0

        with col_opt2:
            flag_borderline = st.checkbox(
                "Flag Borderline Cases",
                value=True,
                key="codebook_flag_borderline",
                help="Identify and flag ambiguous or borderline codes"
            )
            hierarchical = st.checkbox(
                "Hierarchical Grouping",
                value=True,
                key="codebook_hierarchical",
                help="Organize codes into parent categories"
            )

        # Get provider settings from page (set by _get_active_provider)
        provider = st.session_state.get("_codebook_provider")
        api_key = st.session_state.get("_codebook_api_key")
        base_url = st.session_state.get("_codebook_base_url")
        model = st.session_state.get("model_name") or get_selected_model()

        if not provider:
            return ToolConfig(
                is_valid=False,
                error_message="No AI provider configured. Go to LLM Providers to set one up, or start LM Studio.",
                config_data={
                    "df": df,
                    "data_source": data_source,
                    "text_column": text_column
                }
            )

        if PROVIDER_CONFIGS[provider].requires_api_key and (not api_key or api_key == "dummy"):
            return ToolConfig(
                is_valid=False,
                error_message="Provider requires an API key. Configure it in LLM Providers page.",
                config_data={
                    "df": df,
                    "data_source": data_source,
                    "text_column": text_column
                }
            )

        config_data = {
            "df": df,
            "data_source": data_source,
            "text_column": text_column,
            "approach": approach.lower(),
            "initial_themes": initial_themes,
            "custom_instructions": custom_instructions,
            "num_codes": num_codes,
            "granularity": granularity.lower(),
            "include_examples": include_examples,
            "examples_per_code": examples_per_code,
            "flag_borderline": flag_borderline,
            "hierarchical": hierarchical,
            "chunk_size": chunk_size,
            "max_chunks": max_chunks,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        }

        return ToolConfig(is_valid=True, config_data=config_data)

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute the codebook generation process"""

        df = config["df"]
        text_column = config["text_column"]
        approach = config["approach"]
        num_codes = config["num_codes"]
        granularity = config["granularity"]
        include_examples = config["include_examples"]
        examples_per_code = config["examples_per_code"]
        flag_borderline = config["flag_borderline"]
        hierarchical = config["hierarchical"]
        initial_themes = config.get("initial_themes", "")

        # Create LLM client
        client = get_client(
            config["provider"],
            config["api_key"],
            config["base_url"]
        )
        model = config["model"]
        provider = config["provider"]

        # Check if JSON mode is supported (not for local models)
        use_json_mode = supports_json_mode(provider, model)

        # Step 1: Prepare chunks
        chunk_size = config.get("chunk_size", 100)
        max_chunks = config.get("max_chunks", 1)

        # Create chunks from the dataframe
        texts_all = df[text_column].dropna().tolist()
        total_texts = len(texts_all)

        chunks = []
        for i in range(0, min(total_texts, chunk_size * max_chunks), chunk_size):
            chunk_texts = texts_all[i:i + chunk_size]
            if chunk_texts:
                chunks.append(chunk_texts)

        if not chunks:
            return ToolResult(success=False, error_message="No text data found in selected column")

        total_steps = len(chunks) + 2  # chunks + merge + code definition
        current_step = 0

        if progress_callback:
            progress_callback(current_step, total_steps, 0, 0, 0, f"Processing {len(chunks)} chunk(s) of data...", False)

        custom_instructions = config.get("custom_instructions", "")

        granularity_instruction = {
            "low": "Focus on broad, overarching themes that capture major patterns.",
            "medium": "Balance between broad themes and specific patterns.",
            "high": "Identify specific, granular patterns and subtle distinctions."
        }

        approach_instruction = {
            "inductive": "Discover themes purely from the data without preconceptions.",
            "deductive": f"Map the data to these predefined themes and refine them:\n{initial_themes}",
            "hybrid": f"Start with these initial themes but also discover new ones from the data:\n{initial_themes}"
        }

        # Build custom instructions section if provided
        custom_section = ""
        if custom_instructions.strip():
            custom_section = f"""
ADDITIONAL INSTRUCTIONS:
{custom_instructions}
"""

        # Step 2: Discover themes from each chunk
        all_chunk_themes = []

        for chunk_idx, chunk_texts in enumerate(chunks):
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, 0, 0, 0, f"Analyzing chunk {chunk_idx + 1}/{len(chunks)}...", False)

            corpus = "\n---\n".join([f"[{i+1}] {text}" for i, text in enumerate(chunk_texts)])

            theme_prompt = f"""Analyze this qualitative data and identify up to {num_codes} recurring themes/patterns.

APPROACH: {approach.upper()}
{approach_instruction[approach]}

GRANULARITY: {granularity.upper()}
{granularity_instruction[granularity]}
{custom_section}
For each theme, provide:
1. Theme name (concise, descriptive)
2. Brief description (1-2 sentences)
3. 1-2 example quotes from the data

DATA:
{corpus}

Respond in JSON format:
{{
    "themes": [
        {{
            "name": "Theme Name",
            "description": "Brief description of the theme",
            "examples": ["quote 1", "quote 2"]
        }}
    ]
}}"""

            # Get theme discovery system prompt (respects overrides)
            ensure_prompts_registered()
            theme_discovery_prompt = get_effective_prompt("codebook.theme_discovery")

            theme_result, theme_error = await call_llm_simple(
                client=client,
                system_prompt=theme_discovery_prompt,
                user_content=theme_prompt,
                model=model,
                json_mode=use_json_mode,
                provider=provider
            )

            if theme_error:
                continue  # Skip failed chunks, try to continue with others

            if theme_result:
                chunk_themes = extract_json(theme_result)
                if chunk_themes and chunk_themes.get("themes"):
                    all_chunk_themes.extend(chunk_themes["themes"])

        if not all_chunk_themes:
            return ToolResult(success=False, error_message="Failed to discover themes from any chunk")

        # Step 3: Merge themes if multiple chunks
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, 0, 0, 0, "Consolidating themes across chunks...", False)

        if len(chunks) > 1 and len(all_chunk_themes) > num_codes:
            # Consolidate themes from multiple chunks
            themes_summary = "\n".join([f"- {t['name']}: {t['description']}" for t in all_chunk_themes])

            merge_prompt = f"""You have discovered these themes from analyzing multiple chunks of qualitative data:

{themes_summary}

Consolidate these into exactly {num_codes} distinct themes by:
1. Merging similar/overlapping themes
2. Keeping the most significant patterns
3. Ensuring themes are mutually exclusive

Respond in JSON format:
{{
    "themes": [
        {{
            "name": "Theme Name",
            "description": "Brief description of the theme",
            "examples": ["quote 1", "quote 2"]
        }}
    ]
}}"""

            # Get theme consolidation system prompt (respects overrides)
            theme_consolidation_prompt = get_effective_prompt("codebook.theme_consolidation")

            merge_result, merge_error = await call_llm_simple(
                client=client,
                system_prompt=theme_consolidation_prompt,
                user_content=merge_prompt,
                model=model,
                json_mode=use_json_mode,
                provider=provider
            )

            if merge_result:
                merged = extract_json(merge_result)
                if merged and merged.get("themes"):
                    themes = merged["themes"]
                else:
                    themes = all_chunk_themes[:num_codes]
            else:
                themes = all_chunk_themes[:num_codes]
        else:
            themes = all_chunk_themes[:num_codes]

        # Step 4: Code Definition
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, 0, 0, 0, "Defining codes with criteria...", False)

        themes_list = "\n".join([f"- {t['name']}: {t['description']}" for t in themes])

        examples_instruction = ""
        if include_examples:
            examples_instruction = f"Include {examples_per_code} example quotes from the data for each code."

        borderline_instruction = ""
        if flag_borderline:
            borderline_instruction = "Identify any codes that may be ambiguous or have borderline cases. Mark these with is_borderline: true and provide notes explaining the ambiguity."

        # Use a sample of texts for examples
        sample_texts = texts_all[:min(50, len(texts_all))]
        sample_corpus = "\n---\n".join([f"[{i+1}] {text}" for i, text in enumerate(sample_texts)])

        code_prompt = f"""For each identified theme, create a formal code entry for a qualitative codebook.

THEMES:
{themes_list}
{custom_section}
For each code, provide:
1. Code name
2. Definition (1-2 sentences explaining what this code captures)
3. Inclusion criteria (specific conditions when this code should be applied)
4. Exclusion criteria (conditions when this code should NOT be applied)
{examples_instruction}
{borderline_instruction}
5. Suggested parent category for hierarchical organization

SAMPLE DATA (for examples):
{sample_corpus}

Respond in JSON format:
{{
    "codes": [
        {{
            "name": "Code Name",
            "definition": "Clear definition",
            "inclusion_criteria": ["criterion 1", "criterion 2"],
            "exclusion_criteria": ["criterion 1", "criterion 2"],
            "examples": ["example quote 1", "example quote 2"],
            "parent_category": "Category Name",
            "is_borderline": false,
            "borderline_notes": null
        }}
    ]
}}"""

        # Get code definition system prompt (respects overrides)
        code_definition_prompt = get_effective_prompt("codebook.code_definition")

        code_result, code_error = await call_llm_simple(
            client=client,
            system_prompt=code_definition_prompt,
            user_content=code_prompt,
            model=model,
            json_mode=use_json_mode,
            provider=provider
        )

        if code_error:
            return ToolResult(success=False, error_message=f"Code definition failed: {code_error}")

        if not code_result:
            return ToolResult(success=False, error_message="Code definition returned empty response")

        codes_data = extract_json(code_result)
        if not codes_data:
            preview = code_result[:500] if len(code_result) > 500 else code_result
            return ToolResult(success=False, error_message=f"Failed to parse code definition response. Raw response: {preview}")

        codes_list = codes_data.get("codes", [])

        # Build Code objects
        codes = []
        for c in codes_list:
            code = Code(
                name=c.get("name", ""),
                definition=c.get("definition", ""),
                inclusion_criteria=c.get("inclusion_criteria", []),
                exclusion_criteria=c.get("exclusion_criteria", []),
                examples=c.get("examples", [])[:examples_per_code] if include_examples else [],
                parent_category=c.get("parent_category", "General"),
                is_borderline=c.get("is_borderline", False) if flag_borderline else False,
                borderline_notes=c.get("borderline_notes") if flag_borderline else None
            )
            codes.append(code)

        # Build categories dictionary
        categories = {}
        if hierarchical:
            for code in codes:
                cat = code.parent_category or "General"
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(code.name)
        else:
            categories["All Codes"] = [code.name for code in codes]

        # Create codebook
        total_analyzed = min(len(texts_all), chunk_size * len(chunks))
        codebook = Codebook(
            title=f"Codebook - {config['data_source']}",
            approach=approach,
            codes=codes,
            categories=categories,
            data_source=config["data_source"],
            total_items_analyzed=total_analyzed
        )

        total_analyzed = min(len(texts_all), chunk_size * len(chunks))

        if progress_callback:
            progress_callback(total_steps, total_steps, 1, 0, 0, "Codebook generation complete!", False)

        # Calculate stats
        borderline_count = sum(1 for c in codes if c.is_borderline)

        return ToolResult(
            success=True,
            data=codebook,
            stats={
                "codes": len(codes),
                "categories": len(categories),
                "borderline": borderline_count,
                "items_analyzed": total_analyzed,
                "chunks_processed": len(chunks)
            }
        )

    def render_results(self, result: ToolResult):
        """Render the generated codebook"""
        from ui.components.download_buttons import render_single_download

        if not result.success:
            st.error(f"Codebook generation failed: {result.error_message}")
            return

        codebook: Codebook = result.data
        stats = result.stats

        # Stats summary
        chunks = stats.get("chunks_processed", 1)
        if chunks > 1:
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col5 = None

        with col1:
            st.metric("Codes Generated", stats.get("codes", 0))
        with col2:
            st.metric("Categories", stats.get("categories", 0))
        with col3:
            st.metric("Borderline Cases", stats.get("borderline", 0))
        with col4:
            st.metric("Items Analyzed", stats.get("items_analyzed", 0))
        if col5:
            with col5:
                st.metric("Chunks Processed", chunks)

        st.divider()

        # Codebook display
        st.subheader("Generated Codebook")

        for category, code_names in codebook.categories.items():
            with st.expander(f"**{category}** ({len(code_names)} codes)", expanded=True):
                for code_name in code_names:
                    code = next((c for c in codebook.codes if c.name == code_name), None)
                    if not code:
                        continue

                    # Code header with borderline indicator
                    if code.is_borderline:
                        st.markdown(f"#### :warning: {code.name}")
                    else:
                        st.markdown(f"#### {code.name}")

                    st.markdown(f"**Definition:** {code.definition}")

                    col_inc, col_exc = st.columns(2)
                    with col_inc:
                        if code.inclusion_criteria:
                            st.markdown("**Apply when:**")
                            for criterion in code.inclusion_criteria:
                                st.markdown(f"- {criterion}")

                    with col_exc:
                        if code.exclusion_criteria:
                            st.markdown("**Do NOT apply when:**")
                            for criterion in code.exclusion_criteria:
                                st.markdown(f"- {criterion}")

                    if code.examples:
                        st.markdown("**Examples:**")
                        for example in code.examples:
                            st.markdown(f'> *"{example}"*')

                    if code.is_borderline and code.borderline_notes:
                        st.warning(f"**Borderline Notes:** {code.borderline_notes}")

                    st.markdown("---")

        # Export options
        st.divider()
        st.subheader("Export Codebook")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            json_data = json.dumps(codebook.to_dict(), indent=2)
            st.download_button(
                "JSON",
                json_data,
                "codebook.json",
                "application/json",
                key="codebook_download_json",
                use_container_width=True
            )

        with col2:
            csv_df = pd.DataFrame(codebook.to_csv_rows())
            csv_data = csv_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "CSV",
                csv_data,
                "codebook.csv",
                "text/csv",
                key="codebook_download_csv",
                use_container_width=True
            )

        with col3:
            markdown_data = codebook.to_markdown()
            st.download_button(
                "Markdown",
                markdown_data,
                "codebook.md",
                "text/markdown",
                key="codebook_download_md",
                use_container_width=True
            )

        with col4:
            qual_prompt = codebook.to_qualitative_coder_prompt()
            st.download_button(
                "Coder Prompt",
                qual_prompt,
                "qualitative_coder_prompt.txt",
                "text/plain",
                key="codebook_download_prompt",
                use_container_width=True
            )

        # Use in Qualitative Coder button
        st.divider()
        if st.button(
            ":material/psychology: Use in Qualitative Coder",
            type="primary",
            use_container_width=True,
            key="codebook_use_in_qualitative"
        ):
            # Store the generated prompt in session state for qualitative coder
            st.session_state["qualitative_prompt_value"] = codebook.to_qualitative_coder_prompt()
            st.success("Codebook prompt copied! Navigate to Qualitative Coder to use it.")
            st.info("The coding instructions have been pre-filled in the Qualitative Coder. Go to **Qualitative Coder** in the navigation to apply this codebook to your data.")
