# System Prompts Settings

Handai allows you to customize the AI system prompts used by various tools. Changes can be temporary (session only) or permanent (saved to database).

## Overview

System prompts are the instructions given to AI models that define how they should behave and respond. By customizing these prompts, you can:

- Tailor AI behavior to your specific use case
- Add domain-specific instructions
- Modify output formats
- Experiment with different prompting strategies

## Accessing System Prompts

Navigate to **Settings** > **System Prompts** tab to view and edit all available prompts.

## Override Priority

When a tool requests a system prompt, the following priority order applies:

1. **Session Override** (highest priority) - Temporary changes that last only for the current session
2. **Permanent Override** - Saved to database, persists across sessions
3. **Default Value** - The hardcoded default in the source code

## Available Prompts

Each tool has two prompt variants:
- **Standard** - Concise, minimal prompts for general use
- **Rigorous** - Detailed, methodologically-grounded prompts with explicit frameworks and quality standards

### Qualitative Coder

| Prompt ID | Name | Description |
|-----------|------|-------------|
| `qualitative.default_prompt` | Default Coding Prompt | Concise prompt for qualitative coding tasks |
| `qualitative.default_prompt.rigorous` | Default Coding Prompt (Rigorous) | Detailed prompt with analytical framework and quality standards |

### Consensus Coder

| Prompt ID | Name | Description |
|-----------|------|-------------|
| `consensus.worker_prompt` | Worker Model Prompt | Basic prompt for worker models |
| `consensus.worker_prompt.rigorous` | Worker Model Prompt (Rigorous) | Emphasizes independent coding and inter-rater reliability |
| `consensus.judge_prompt` | Judge Model Prompt | Basic prompt for synthesizing worker responses |
| `consensus.judge_prompt.rigorous` | Judge Model Prompt (Rigorous) | Detailed adjudication rules and decision criteria |

### Codebook Generator

| Prompt ID | Name | Description |
|-----------|------|-------------|
| `codebook.theme_discovery` | Theme Discovery Prompt | Basic theme discovery |
| `codebook.theme_discovery.rigorous` | Theme Discovery Prompt (Rigorous) | Braun & Clarke framework for systematic thematic analysis |
| `codebook.theme_consolidation` | Theme Consolidation Prompt | Basic theme merging |
| `codebook.theme_consolidation.rigorous` | Theme Consolidation Prompt (Rigorous) | Explicit criteria for merging vs preserving themes |
| `codebook.code_definition` | Code Definition Prompt | Basic code definition |
| `codebook.code_definition.rigorous` | Code Definition Prompt (Rigorous) | Comprehensive codebook with inclusion/exclusion criteria |

### Generate Tool

| Prompt ID | Name | Description |
|-----------|------|-------------|
| `generate.column_suggestions` | Column Suggestions Prompt | Basic column name suggestions |
| `generate.column_suggestions.rigorous` | Column Suggestions Prompt (Rigorous) | Data architecture principles with naming conventions |

### Document Templates

| Prompt ID | Name | Description |
|-----------|------|-------------|
| `documents.master_prompt` | Master Document Prompt | Basic CSV extraction rules |
| `documents.master_prompt.rigorous` | Master Document Prompt (Rigorous) | Comprehensive formatting rules with quality checklist |

### Automator

| Prompt ID | Name | Description |
|-----------|------|-------------|
| `automator.critical_rules` | Critical Rules Section | Basic processing rules |
| `automator.critical_rules.rigorous` | Critical Rules Section (Rigorous) | Comprehensive rules covering output discipline, consistency, and QA |

## When to Use Rigorous Prompts

The rigorous variants are recommended when:

- **High-stakes research** - Academic papers, regulatory submissions
- **Multi-coder studies** - When inter-rater reliability matters
- **Complex data** - Nuanced content requiring careful analysis
- **Quality audits** - When outputs need to be defensible
- **Training coders** - To establish consistent coding practices

The standard prompts are suitable for:

- **Exploratory analysis** - Initial data exploration
- **Simple tasks** - Straightforward categorization
- **Fast iteration** - When speed matters more than rigor
- **Personal projects** - Lower stakes analysis

## Using the UI

### Status Indicators

Each prompt displays a status badge:

- :green_circle: **Using Default** - No overrides, using the hardcoded default
- :orange_circle: **Session Override** - Temporary override active (will be lost on restart)
- :blue_circle: **Permanent Override** - Database override active (persists across sessions)

### Actions

| Button | Description |
|--------|-------------|
| **Save Temporary** | Save changes for this session only. Changes are lost when you restart the app. |
| **Save Permanent** | Save changes to the database. Changes persist across sessions. |
| **Reset to Default** | Remove all overrides and revert to the default value. |
| **View Default** | Display the original default value for reference. |

## Best Practices

1. **Test with Temporary First** - Use session overrides to experiment before making permanent changes
2. **Document Your Changes** - Keep notes about why you modified a prompt
3. **Start Small** - Make incremental changes rather than rewriting entire prompts
4. **Check Output Quality** - Verify that modified prompts produce the expected results

## Database Storage

Permanent overrides are stored in the `system_prompt_overrides` table:

```sql
CREATE TABLE system_prompt_overrides (
    prompt_id TEXT PRIMARY KEY,
    custom_value TEXT NOT NULL,
    is_enabled INTEGER DEFAULT 1,
    updated_at TEXT NOT NULL
)
```

## Programmatic Access

You can also access the prompt registry programmatically:

```python
from core.prompt_registry import (
    get_effective_prompt,
    set_temporary_override,
    set_permanent_override,
    reset_to_default,
    get_prompt_status
)

# Get the current effective prompt (respects overrides)
prompt = get_effective_prompt("qualitative.default_prompt")

# Check current status
status = get_prompt_status("qualitative.default_prompt")  # "default", "session", or "permanent"

# Set a temporary override
set_temporary_override("qualitative.default_prompt", "Your custom prompt here")

# Set a permanent override
set_permanent_override("qualitative.default_prompt", "Your custom prompt here")

# Reset to default
reset_to_default("qualitative.default_prompt")
```

## Troubleshooting

### Changes Not Taking Effect

1. Ensure you clicked "Save Temporary" or "Save Permanent"
2. Check the status badge to confirm the override is active
3. Refresh the page and verify the change persisted

### Reverting Changes

Click "Reset to Default" to remove all overrides for a prompt. This removes both session and permanent overrides.

### Database Issues

If permanent overrides aren't persisting, check that the database migration has run. The `system_prompt_overrides` table should exist in your database.
