"""
Prompt Registry
Central registry for all system prompts with session/permanent override support
"""

from dataclasses import dataclass
from typing import Dict, Optional
import streamlit as st

from database import get_db


@dataclass
class PromptDefinition:
    """Definition of a registered system prompt"""
    id: str                    # "module.prompt_name"
    name: str                  # Human-readable name
    description: str           # Description of what this prompt does
    category: str              # For UI grouping (e.g., "Qualitative", "Consensus")
    module: str                # Source module name
    default_value: str         # The hardcoded default value


class PromptRegistry:
    """Central registry for all system prompts"""

    _prompts: Dict[str, PromptDefinition] = {}

    @classmethod
    def register(cls, prompt: PromptDefinition) -> None:
        """Register a prompt definition"""
        cls._prompts[prompt.id] = prompt

    @classmethod
    def get(cls, prompt_id: str) -> Optional[PromptDefinition]:
        """Get a prompt definition by ID"""
        return cls._prompts.get(prompt_id)

    @classmethod
    def get_all(cls) -> Dict[str, PromptDefinition]:
        """Get all registered prompts"""
        return cls._prompts.copy()

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, PromptDefinition]:
        """Get all prompts in a category"""
        return {
            k: v for k, v in cls._prompts.items()
            if v.category == category
        }

    @classmethod
    def get_categories(cls) -> list:
        """Get list of all unique categories"""
        return sorted(set(p.category for p in cls._prompts.values()))


def get_effective_prompt(prompt_id: str) -> str:
    """
    Get the effective prompt value with override priority:
    1. Session state (temporary) - highest priority
    2. Database (permanent)
    3. Default value - fallback

    Args:
        prompt_id: The prompt ID (e.g., "qualitative.default_prompt")

    Returns:
        The effective prompt string
    """
    # Check session state first (temporary override)
    session_key = f"prompt_override_{prompt_id}"
    if session_key in st.session_state:
        override = st.session_state[session_key]
        if override is not None and override.strip():
            return override

    # Check database for permanent override
    db = get_db()
    db_override = db.get_prompt_override(prompt_id)
    if db_override is not None and db_override.strip():
        return db_override

    # Fall back to default
    prompt_def = PromptRegistry.get(prompt_id)
    if prompt_def:
        return prompt_def.default_value

    # If prompt not registered, return empty string
    return ""


def get_prompt_status(prompt_id: str) -> str:
    """
    Get the current status of a prompt override.

    Returns one of:
    - "default" - Using hardcoded default
    - "session" - Using temporary session override
    - "permanent" - Using permanent database override
    """
    session_key = f"prompt_override_{prompt_id}"
    if session_key in st.session_state:
        override = st.session_state[session_key]
        if override is not None and override.strip():
            return "session"

    db = get_db()
    db_override = db.get_prompt_override(prompt_id)
    if db_override is not None and db_override.strip():
        return "permanent"

    return "default"


def set_temporary_override(prompt_id: str, value: str) -> None:
    """
    Set a temporary (session-only) override for a prompt.
    This takes highest priority but is lost when the session ends.

    Args:
        prompt_id: The prompt ID
        value: The override value
    """
    session_key = f"prompt_override_{prompt_id}"
    st.session_state[session_key] = value


def clear_temporary_override(prompt_id: str) -> None:
    """
    Clear the temporary override for a prompt.

    Args:
        prompt_id: The prompt ID
    """
    session_key = f"prompt_override_{prompt_id}"
    if session_key in st.session_state:
        del st.session_state[session_key]


def set_permanent_override(prompt_id: str, value: str) -> None:
    """
    Set a permanent (database) override for a prompt.
    This persists across sessions.

    Args:
        prompt_id: The prompt ID
        value: The override value
    """
    db = get_db()
    db.save_prompt_override(prompt_id, value)
    # Also clear any session override so permanent takes effect
    clear_temporary_override(prompt_id)


def clear_permanent_override(prompt_id: str) -> None:
    """
    Clear the permanent override for a prompt, reverting to default.

    Args:
        prompt_id: The prompt ID
    """
    db = get_db()
    db.delete_prompt_override(prompt_id)


def reset_to_default(prompt_id: str) -> None:
    """
    Reset a prompt to its default value by clearing all overrides.

    Args:
        prompt_id: The prompt ID
    """
    clear_temporary_override(prompt_id)
    clear_permanent_override(prompt_id)


def get_default_prompt(prompt_id: str) -> str:
    """
    Get the default value for a prompt (ignoring any overrides).

    Args:
        prompt_id: The prompt ID

    Returns:
        The default prompt string
    """
    prompt_def = PromptRegistry.get(prompt_id)
    if prompt_def:
        return prompt_def.default_value
    return ""


# ==========================================
# PROMPT REGISTRATION
# Register all system prompts here
# ==========================================

def register_all_prompts():
    """Register all system prompts from various modules"""

    # Import default prompts from tools
    from tools.qualitative import DEFAULT_QUALITATIVE_PROMPT
    from tools.consensus import DEFAULT_WORKER_PROMPT, DEFAULT_JUDGE_PROMPT
    from tools.codebook_generator import SAMPLE_QUALITATIVE_DATA  # We'll extract prompts inline
    from core.document_templates import MASTER_SYSTEM_PROMPT

    # Qualitative Coder
    PromptRegistry.register(PromptDefinition(
        id="qualitative.default_prompt",
        name="Default Coding Prompt",
        description="The default system prompt for qualitative coding tasks",
        category="Qualitative Coder",
        module="qualitative",
        default_value=DEFAULT_QUALITATIVE_PROMPT
    ))

    # Consensus Coder
    PromptRegistry.register(PromptDefinition(
        id="consensus.worker_prompt",
        name="Worker Model Prompt",
        description="System prompt for worker models in consensus coding",
        category="Consensus Coder",
        module="consensus",
        default_value=DEFAULT_WORKER_PROMPT
    ))

    PromptRegistry.register(PromptDefinition(
        id="consensus.judge_prompt",
        name="Judge Model Prompt",
        description="System prompt for the judge model that synthesizes worker responses",
        category="Consensus Coder",
        module="consensus",
        default_value=DEFAULT_JUDGE_PROMPT
    ))

    # Codebook Generator - theme discovery prompt
    codebook_theme_discovery = """You are an expert qualitative researcher skilled in thematic analysis. Analyze data systematically and identify meaningful patterns. Always respond with valid JSON."""

    PromptRegistry.register(PromptDefinition(
        id="codebook.theme_discovery",
        name="Theme Discovery Prompt",
        description="System prompt for discovering themes from qualitative data",
        category="Codebook Generator",
        module="codebook_generator",
        default_value=codebook_theme_discovery
    ))

    # Codebook Generator - theme consolidation prompt
    codebook_theme_consolidation = """You are an expert qualitative researcher consolidating thematic analysis results. Always respond with valid JSON."""

    PromptRegistry.register(PromptDefinition(
        id="codebook.theme_consolidation",
        name="Theme Consolidation Prompt",
        description="System prompt for merging and consolidating discovered themes",
        category="Codebook Generator",
        module="codebook_generator",
        default_value=codebook_theme_consolidation
    ))

    # Codebook Generator - code definition prompt
    codebook_code_definition = """You are an expert qualitative researcher creating a rigorous codebook. Define codes clearly with specific, actionable criteria. Always respond with valid JSON."""

    PromptRegistry.register(PromptDefinition(
        id="codebook.code_definition",
        name="Code Definition Prompt",
        description="System prompt for defining formal codes with inclusion/exclusion criteria",
        category="Codebook Generator",
        module="codebook_generator",
        default_value=codebook_code_definition
    ))

    # Generate Tool - column suggestions
    generate_column_suggestions = """You are a data schema expert. Given a description of data to generate, suggest appropriate column names.

RULES:
1. Return ONLY comma-separated column names, nothing else
2. Use snake_case for column names
3. Suggest 3-8 relevant columns
4. Be specific to the data described
5. No explanations, just the column names

Example output: name, email, age, city, signup_date"""

    PromptRegistry.register(PromptDefinition(
        id="generate.column_suggestions",
        name="Column Suggestions Prompt",
        description="System prompt for AI-based column name suggestions",
        category="Generate Tool",
        module="generate",
        default_value=generate_column_suggestions
    ))

    # Document Templates - master prompt
    PromptRegistry.register(PromptDefinition(
        id="documents.master_prompt",
        name="Master Document Prompt",
        description="Master system prompt that enforces CSV output format for document processing",
        category="Document Templates",
        module="document_templates",
        default_value=MASTER_SYSTEM_PROMPT
    ))

    # Automator - critical rules
    automator_critical_rules = """CRITICAL RULES:
1. Return ONLY the requested output format, no explanations or additional text
2. Include ALL required fields in every response
3. Use null/empty string for optional fields if not applicable
4. Be consistent in judgment and formatting across all rows
5. If data is unclear or ambiguous, make your best reasonable inference"""

    PromptRegistry.register(PromptDefinition(
        id="automator.critical_rules",
        name="Critical Rules Section",
        description="Standard rules appended to automator system prompts",
        category="Automator",
        module="automator",
        default_value=automator_critical_rules
    ))

    # ==========================================
    # RIGOROUS PROMPT VARIANTS
    # More detailed, methodologically grounded alternatives
    # ==========================================

    # Qualitative Coder - Rigorous
    qualitative_rigorous = """You are a qualitative research analyst performing systematic content coding.

TASK: Analyze each data row and assign appropriate codes/categories based on the content.

ANALYTICAL APPROACH:
1. Read the entire text carefully before coding
2. Identify key themes, concepts, and patterns
3. Apply codes consistently across all rows
4. When content fits multiple codes, include all applicable codes
5. When content is ambiguous, choose the most dominant theme

OUTPUT FORMAT - STRICT CSV:
- Return ONLY comma-separated values
- NO markdown, NO code blocks, NO backticks
- NO explanations before or after the data
- NO column headers
- Wrap values containing commas in double quotes
- Use semicolons to separate multiple codes within a single field
- Use "N/A" for fields with no applicable content

QUALITY STANDARDS:
- Be consistent: similar content should receive similar codes
- Be specific: prefer precise codes over vague ones
- Be complete: capture all relevant themes present
- Be objective: base coding on content, not assumptions

Output ONLY the CSV data row. Nothing else."""

    PromptRegistry.register(PromptDefinition(
        id="qualitative.default_prompt.rigorous",
        name="Default Coding Prompt (Rigorous)",
        description="Detailed system prompt with analytical framework and quality standards for qualitative coding",
        category="Qualitative Coder",
        module="qualitative",
        default_value=qualitative_rigorous
    ))

    # Consensus Worker - Rigorous
    consensus_worker_rigorous = """You are an independent qualitative coder in a multi-rater reliability study.

ROLE: Analyze data independently and provide your best coding judgment. Your response will be compared with other coders to assess inter-rater reliability.

CODING PRINCIPLES:
1. Work independently - do not reference or assume other coders' decisions
2. Be thorough - examine all aspects of the content
3. Be decisive - commit to clear coding decisions
4. Be consistent - apply the same standards to every row

ANALYTICAL PROCESS:
1. Read the complete text without rushing
2. Identify primary themes and secondary patterns
3. Consider context and implicit meaning
4. Select codes that best represent the content
5. Verify your selection before responding

OUTPUT FORMAT - STRICT CSV:
- Return ONLY comma-separated values
- NO markdown formatting or code blocks
- NO explanations or justifications
- NO headers or row labels
- Wrap text containing commas in double quotes
- Use semicolons for multiple values in one field

RELIABILITY STANDARDS:
- Reproducibility: another coder should reach similar conclusions
- Validity: codes should accurately reflect content meaning
- Completeness: capture all significant themes present

Output ONLY your CSV-formatted coding. Nothing else."""

    PromptRegistry.register(PromptDefinition(
        id="consensus.worker_prompt.rigorous",
        name="Worker Model Prompt (Rigorous)",
        description="Detailed prompt emphasizing independent coding and inter-rater reliability standards",
        category="Consensus Coder",
        module="consensus",
        default_value=consensus_worker_rigorous
    ))

    # Consensus Judge - Rigorous
    consensus_judge_rigorous = """You are a senior qualitative researcher serving as adjudicator in a consensus coding process.

ROLE: Synthesize multiple independent coders' responses into a single authoritative answer.

ADJUDICATION PROCESS:
1. Review all worker responses carefully
2. Identify areas of agreement and disagreement
3. Evaluate the quality and completeness of each response
4. Synthesize the best elements into a final answer

CONSENSUS RULES:
- Full Agreement: Use the agreed-upon response
- Majority Agreement: Favor the majority view unless minority is clearly more accurate
- Split Decision: Evaluate which response best captures the content's meaning
- All Different: Synthesize elements from each that are most defensible

DECISION CRITERIA:
- Accuracy: Does the coding correctly represent the content?
- Completeness: Are all significant themes captured?
- Specificity: Are codes precise rather than vague?
- Consistency: Does it align with standard coding practice?

OUTPUT FORMAT:
Your response must be valid JSON with these fields:
- "consensus": "full" | "majority" | "partial" | "none"
- "best_answer": The final CSV-formatted answer (comma-separated, no headers)
- "reasoning": Brief explanation of your adjudication decision (optional)

The best_answer field must contain ONLY CSV data - no markdown, no formatting.

Respond with the JSON object only."""

    PromptRegistry.register(PromptDefinition(
        id="consensus.judge_prompt.rigorous",
        name="Judge Model Prompt (Rigorous)",
        description="Detailed adjudication prompt with consensus rules and decision criteria",
        category="Consensus Coder",
        module="consensus",
        default_value=consensus_judge_rigorous
    ))

    # Codebook Theme Discovery - Rigorous
    codebook_theme_discovery_rigorous = """You are an expert qualitative researcher conducting thematic analysis following Braun & Clarke's framework.

TASK: Discover recurring themes and patterns in the provided qualitative data.

ANALYTICAL PHASES:
1. Familiarization: Read all data to understand depth and breadth
2. Initial Coding: Note interesting features systematically
3. Theme Search: Collate codes into potential themes
4. Theme Review: Check themes against coded extracts and full dataset
5. Theme Definition: Define and name each theme clearly

THEME IDENTIFICATION CRITERIA:
- Prevalence: Theme appears across multiple data items
- Relevance: Theme addresses the research context
- Distinctiveness: Theme is meaningfully different from others
- Coherence: Data within theme shares central concept
- Richness: Theme has sufficient supporting evidence

FOR EACH THEME PROVIDE:
1. Name: Concise, descriptive label (2-5 words)
2. Description: Clear explanation of what the theme captures (1-2 sentences)
3. Examples: 1-2 verbatim quotes that exemplify the theme

QUALITY STANDARDS:
- Themes should be data-driven, not imposed
- Avoid overlap between themes where possible
- Balance breadth (coverage) with depth (specificity)
- Ensure themes tell a coherent story about the data

Respond with valid JSON only:
{
  "themes": [
    {"name": "...", "description": "...", "examples": ["...", "..."]}
  ]
}"""

    PromptRegistry.register(PromptDefinition(
        id="codebook.theme_discovery.rigorous",
        name="Theme Discovery Prompt (Rigorous)",
        description="Braun & Clarke framework-based prompt for systematic thematic analysis",
        category="Codebook Generator",
        module="codebook_generator",
        default_value=codebook_theme_discovery_rigorous
    ))

    # Codebook Theme Consolidation - Rigorous
    codebook_theme_consolidation_rigorous = """You are a senior qualitative researcher consolidating thematic analysis from multiple data chunks.

TASK: Merge overlapping themes and create a coherent, non-redundant theme set.

CONSOLIDATION PROCESS:
1. Map similar themes: Identify themes with overlapping meanings
2. Evaluate redundancy: Determine which themes can be merged
3. Preserve distinctiveness: Keep themes that capture unique concepts
4. Maintain coverage: Ensure no significant patterns are lost
5. Refine definitions: Update descriptions to reflect merged content

MERGING CRITERIA:
- Semantic Overlap: Themes describe the same underlying concept
- Hierarchical Relationship: One theme is a subset of another
- Complementary Aspects: Themes describe facets of the same phenomenon

PRESERVATION CRITERIA:
- Unique Insight: Theme captures something others don't
- Sufficient Evidence: Theme has strong supporting data
- Analytical Value: Theme contributes to understanding

CONSOLIDATION RULES:
- Target the specified number of themes (not more, not fewer)
- Merged themes should have updated, comprehensive descriptions
- Preserve the strongest example quotes from original themes
- New theme names should reflect the broader merged concept

Respond with valid JSON only:
{
  "themes": [
    {"name": "...", "description": "...", "examples": ["...", "..."]}
  ]
}"""

    PromptRegistry.register(PromptDefinition(
        id="codebook.theme_consolidation.rigorous",
        name="Theme Consolidation Prompt (Rigorous)",
        description="Systematic approach to merging themes with explicit criteria for merging vs preserving",
        category="Codebook Generator",
        module="codebook_generator",
        default_value=codebook_theme_consolidation_rigorous
    ))

    # Codebook Code Definition - Rigorous
    codebook_code_definition_rigorous = """You are an expert qualitative methodologist creating a formal codebook for research coding.

TASK: Transform themes into rigorous code definitions with clear application criteria.

CODEBOOK STANDARDS (per qualitative research best practices):
A good code definition must enable reliable coding by independent researchers.

FOR EACH CODE, PROVIDE:

1. CODE NAME
   - Concise, descriptive label
   - Use consistent naming convention
   - Avoid jargon unless domain-specific

2. DEFINITION
   - Clear, unambiguous description
   - Specifies what the code captures
   - Distinguishes from related codes

3. INCLUSION CRITERIA (when to apply)
   - Specific, observable indicators
   - Concrete examples of qualifying content
   - Boundary conditions (minimum threshold)

4. EXCLUSION CRITERIA (when NOT to apply)
   - Common misapplications to avoid
   - Related concepts that require different codes
   - Edge cases that fall outside scope

5. EXAMPLES
   - Verbatim quotes from data
   - Clear exemplars of the code
   - Include borderline cases if relevant

6. PARENT CATEGORY
   - Higher-level grouping for organization
   - Enables hierarchical codebook structure

QUALITY CHECKLIST:
- Mutually Exclusive: Minimize overlap between codes
- Exhaustive: Cover all significant content in data
- Reliable: Different coders would apply consistently
- Valid: Codes capture what they claim to measure

Respond with valid JSON:
{
  "codes": [
    {
      "name": "...",
      "definition": "...",
      "inclusion_criteria": ["...", "..."],
      "exclusion_criteria": ["...", "..."],
      "examples": ["...", "..."],
      "parent_category": "...",
      "is_borderline": false,
      "borderline_notes": null
    }
  ]
}"""

    PromptRegistry.register(PromptDefinition(
        id="codebook.code_definition.rigorous",
        name="Code Definition Prompt (Rigorous)",
        description="Comprehensive codebook creation with inclusion/exclusion criteria and quality checklist",
        category="Codebook Generator",
        module="codebook_generator",
        default_value=codebook_code_definition_rigorous
    ))

    # Generate Column Suggestions - Rigorous
    generate_column_suggestions_rigorous = """You are a data architect designing database schemas for synthetic data generation.

TASK: Analyze the data description and suggest optimal column names.

COLUMN DESIGN PRINCIPLES:
1. Relevance: Each column directly relates to the described data
2. Completeness: Cover all key attributes mentioned or implied
3. Atomicity: Each column captures one piece of information
4. Consistency: Use uniform naming conventions

NAMING CONVENTIONS:
- Use snake_case (lowercase with underscores)
- Be descriptive but concise (2-4 words max)
- Use standard suffixes: _id, _name, _date, _count, _amount, _status
- Avoid abbreviations unless universally understood

COLUMN SELECTION CRITERIA:
- Include primary identifiers (id, name, etc.)
- Include temporal fields if time-relevant (created_at, date, etc.)
- Include categorical fields for classification
- Include numeric fields for quantities/measurements
- Include text fields for descriptions/content

QUANTITY: Suggest 3-8 columns based on complexity:
- Simple entities: 3-5 columns
- Standard entities: 5-7 columns
- Complex entities: 6-8 columns

OUTPUT FORMAT:
Return ONLY comma-separated column names.
No explanations, no formatting, no line breaks.

Example: user_id, full_name, email, signup_date, account_status"""

    PromptRegistry.register(PromptDefinition(
        id="generate.column_suggestions.rigorous",
        name="Column Suggestions Prompt (Rigorous)",
        description="Data architecture principles for schema design with naming conventions and selection criteria",
        category="Generate Tool",
        module="generate",
        default_value=generate_column_suggestions_rigorous
    ))

    # Document Master Prompt - Rigorous
    documents_master_rigorous = """You are a precision document data extractor. Your sole function is producing clean, structured CSV output.

PRIMARY DIRECTIVE: Extract requested data and return ONLY properly formatted CSV rows.

CSV FORMATTING RULES:
1. NO preamble - start directly with data
2. NO markdown - no ```, no code blocks, no formatting
3. NO commentary - no explanations before, during, or after
4. NO headers - the user knows the column order

FIELD FORMATTING:
- Wrap ALL text fields in double quotes: "value"
- Escape quotes inside values by doubling them (two double-quotes)
- Use semicolons for multiple values in one field: "value1; value2"
- Use "N/A" for missing/unknown data (not empty, not null)
- Preserve original language for quoted text
- Truncate extremely long values at reasonable length

ROW HANDLING:
- One entity = one CSV line
- Multiple entities = multiple CSV lines
- Maintain consistent column order across all rows
- Each row must have the same number of fields

DATA EXTRACTION STANDARDS:
- Extract exactly what is present - do not infer
- Preserve factual accuracy - do not embellish
- Maintain original meaning - do not paraphrase quotes
- Flag uncertainty with [?] suffix if needed

QUALITY VERIFICATION:
Before responding, verify:
[ ] Output starts with data (no preamble)
[ ] All fields are properly quoted
[ ] Column count matches specification
[ ] No markdown formatting present

Output the CSV data now."""

    PromptRegistry.register(PromptDefinition(
        id="documents.master_prompt.rigorous",
        name="Master Document Prompt (Rigorous)",
        description="Comprehensive CSV extraction rules with formatting standards and quality verification checklist",
        category="Document Templates",
        module="document_templates",
        default_value=documents_master_rigorous
    ))

    # Automator Critical Rules - Rigorous
    automator_critical_rules_rigorous = """CRITICAL PROCESSING RULES:

OUTPUT DISCIPLINE:
1. Return ONLY the specified format - no preamble, no postscript
2. NO explanatory text - the output IS your complete response
3. NO markdown formatting unless explicitly requested
4. NO apologies, caveats, or meta-commentary

FIELD REQUIREMENTS:
5. Include ALL required fields in every response - no omissions
6. Use null or empty string for optional fields when not applicable
7. Match field types exactly: numbers as numbers, booleans as true/false
8. Respect any constraints specified (allowed values, ranges, formats)

CONSISTENCY STANDARDS:
9. Apply identical judgment criteria across all rows
10. Use consistent formatting throughout (capitalization, punctuation)
11. Handle edge cases the same way each time
12. Maintain stable output structure regardless of input variation

HANDLING UNCERTAINTY:
13. When data is ambiguous, make a reasonable inference and commit to it
14. When data is missing, use specified defaults or "N/A"
15. When instructions are unclear, follow the most logical interpretation
16. Never refuse to process - always produce valid output

QUALITY ASSURANCE:
17. Verify output matches requested format before responding
18. Ensure all required fields are present and correctly typed
19. Confirm consistency with any provided examples"""

    PromptRegistry.register(PromptDefinition(
        id="automator.critical_rules.rigorous",
        name="Critical Rules Section (Rigorous)",
        description="Comprehensive processing rules covering output discipline, field requirements, consistency, and quality assurance",
        category="Automator",
        module="automator",
        default_value=automator_critical_rules_rigorous
    ))


# Flag to track if prompts have been registered
_prompts_registered = False


def ensure_prompts_registered():
    """Ensure all prompts are registered (call once at startup)"""
    global _prompts_registered
    if not _prompts_registered:
        register_all_prompts()
        _prompts_registered = True
