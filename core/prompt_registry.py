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


# Flag to track if prompts have been registered
_prompts_registered = False


def ensure_prompts_registered():
    """Ensure all prompts are registered (call once at startup)"""
    global _prompts_registered
    if not _prompts_registered:
        register_all_prompts()
        _prompts_registered = True
