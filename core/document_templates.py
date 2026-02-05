"""
Document Processing Templates
Predefined templates for common document processing tasks
"""

from typing import Dict, List, Optional
from functools import lru_cache

# Master system prompt that enforces CSV output
MASTER_SYSTEM_PROMPT = """You are a document data extractor. Your ONLY job is to output clean CSV data.

CRITICAL RULES:
1. Output ONLY CSV data - NO headers, NO markdown, NO code blocks, NO explanations
2. Each row must match the exact column format specified
3. Wrap ALL text fields in double quotes ""
4. Use semicolons to separate multiple values within a single field
5. NEVER output ```csv or ``` or any markdown formatting
6. NEVER add notes, explanations, or commentary before or after the data
7. If a field has no data, use "N/A"
8. For multiple entities (e.g., multiple people), output multiple CSV lines

The user will specify the exact columns needed. Follow their format exactly."""


# Document processing templates
DOCUMENT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Custom": {
        "prompt": "",
        "columns": "column1,column2,column3",
        "description": "Create your own custom extraction template"
    },
    "Legal Case Extraction (Finnish)": {
        "prompt": """# Role and Objective
You are a Senior Legal Data Analyst and Expert Case Clerk. Your task is to analyze legal document text and extract structured data about all persons mentioned.

# Extraction Guidelines
1. **Entity Identification**: Identify EVERY natural person mentioned in the document.
2. **Role Normalization**: Assign exactly one of: Defendant, Plaintiff, Witness, Judge, Counsel, Law Enforcement, Victim, Other
3. **Crime Specificity**: Keep the Finnish crime name followed by English translation in parentheses. If no crime applies, use "N/A".
4. **Legal Status**: Classify as: Convicted, Accused, Acquitted, Victim, Witness, or N/A
5. **Quote**: Extract a specific Finnish sentence (max 100 chars) that supports the classification.

# Output Rules
- Output ONLY CSV lines, one per person
- NO headers, NO markdown, NO code blocks, NO explanations
- Wrap ALL fields in double quotes
- If multiple people, output multiple lines

# Example Output Format:
"Seppo Kalevi Koponen","Defendant","Liikenneturvallisuuden vaarantaminen (Traffic endangerment)","Convicted","Koponen on tienkäyttäjänä tahallaan rikkonut..."
"Pia Kauppinen","Plaintiff","N/A","Victim","Kauppisen vaatimus kärsimyksen korvaamisesta..."
"Eemeli Sillanpää","Judge","N/A","N/A","hovioikeudenneuvos Eemeli Sillanpää"

Extract ALL persons now:""",
        "columns": "person_name,standardized_role,crime_charge,legal_status,evidence_quote",
        "description": "Extract person data from Finnish legal documents"
    },
    "Legal Case Extraction (English)": {
        "prompt": """# Role and Objective
You are a Senior Legal Data Analyst. Extract structured data about all persons mentioned in this legal document.

# Extraction Guidelines
1. **Entity Identification**: Identify EVERY natural person mentioned.
2. **Role Normalization**: Assign one of: Defendant, Plaintiff, Witness, Judge, Counsel, Law Enforcement, Victim, Other
3. **Crime/Charge**: The specific crime or charge. Use "N/A" if not applicable.
4. **Legal Status**: Convicted, Accused, Acquitted, Victim, Witness, or N/A
5. **Quote**: Brief supporting quote from document (max 80 chars)

# Output Rules
- Output ONLY CSV lines, one per person
- NO headers, NO markdown, NO explanations
- Wrap ALL fields in double quotes

Example:
"John Smith","Defendant","Armed Robbery","Convicted","Smith was found guilty of..."
"Jane Doe","Witness","N/A","Witness","testified that she saw..."

Extract ALL persons:""",
        "columns": "person_name,standardized_role,crime_charge,legal_status,evidence_quote",
        "description": "Extract person data from English legal documents"
    },
    "Key Information": {
        "prompt": """Extract key information from this document.
Output ONE CSV line with these exact columns.
NO headers, NO markdown, NO explanation.

Example: "Document Title","Brief 2-3 sentence summary here","Legal","Person1; Person2"

Extract now:""",
        "columns": "title,summary,category,key_entities",
        "description": "Extract title, summary, category, and key entities"
    },
    "Data Extraction": {
        "prompt": """Extract all structured data from this document.
Output ONE CSV line. Separate multiple values with semicolons.
NO headers, NO markdown, NO explanation.

Example: "John Doe; Jane Smith","2024-01-15","Helsinki","Company Inc","5000 EUR","Key fact"

Extract now:""",
        "columns": "names,dates,locations,organizations,amounts,key_facts",
        "description": "Extract names, dates, locations, organizations, and amounts"
    },
    "Classification": {
        "prompt": """Classify this document.
Output ONE CSV line.
NO headers, NO markdown.

Example: "Legal","Court Decision","criminal; appeal","high"

Classify now:""",
        "columns": "category,subcategory,tags,confidence",
        "description": "Classify document by category, subcategory, and tags"
    },
    "Summarization": {
        "prompt": """Summarize this document.
Output ONE CSV line.
NO headers, NO markdown.

Example: "Case Title","Full comprehensive summary of the document here","Point1; Point2; Point3"

Summarize now:""",
        "columns": "title,summary,key_points",
        "description": "Generate title, summary, and key points"
    }
}


def get_template_names() -> List[str]:
    """Get list of all template names."""
    return list(DOCUMENT_TEMPLATES.keys())


def get_template(name: str) -> Optional[Dict[str, str]]:
    """
    Get a template by name.

    Args:
        name: Template name

    Returns:
        Template dict with 'prompt', 'columns', 'description' or None
    """
    return DOCUMENT_TEMPLATES.get(name)


def get_template_prompt(name: str) -> str:
    """Get the prompt for a template."""
    template = get_template(name)
    if template:
        return template.get("prompt", "")
    return ""


def get_template_columns(name: str) -> str:
    """Get the columns for a template."""
    template = get_template(name)
    if template:
        return template.get("columns", "column1,column2,column3")
    return "column1,column2,column3"


def get_template_description(name: str) -> str:
    """Get the description for a template."""
    template = get_template(name)
    if template:
        return template.get("description", "")
    return ""


def get_effective_master_prompt() -> str:
    """
    Get the effective master system prompt, respecting any overrides.

    Returns:
        The master system prompt (from override or default)
    """
    try:
        from core.prompt_registry import get_effective_prompt, ensure_prompts_registered
        ensure_prompts_registered()
        return get_effective_prompt("documents.master_prompt")
    except ImportError:
        # Fallback if prompt registry not available
        return MASTER_SYSTEM_PROMPT


def create_full_system_prompt(user_prompt: str) -> str:
    """
    Combine master system prompt with user's specific prompt.

    Args:
        user_prompt: User's extraction instructions

    Returns:
        Full system prompt for LLM
    """
    master = get_effective_master_prompt()
    return master + "\n\n" + user_prompt
