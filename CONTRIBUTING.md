# Contributing to Handai

Notes and guidelines for developers working on this project.

## Project Overview

Handai is a Streamlit-based AI data transformation and generation tool. It uses a modular tool-based architecture where each feature is implemented as a self-contained tool.

## Tech Stack

- **Frontend**: Streamlit (Python)
- **Database**: SQLite with custom ORM (`database/`)
- **AI Providers**: Multiple (OpenAI, Anthropic, Google, Groq, etc.) via `core/providers.py`
- **Desktop**: Electron wrapper (optional, in `desktop/`)

## Project Structure

```
handai/
├── app.py                 # Main entry point, page routing
├── config.py              # Global configuration
├── requirements.txt       # Python dependencies
│
├── tools/                 # Tool implementations (core feature modules)
│   ├── base.py           # BaseTool abstract class - READ THIS FIRST
│   ├── registry.py       # Tool registration system
│   ├── transform.py      # Transform existing data
│   ├── generate.py       # Generate synthetic data
│   ├── manual_coder.py   # Manual qualitative coding
│   └── ...               # Other tools
│
├── pages/                 # Streamlit page wrappers (thin layer)
│   ├── __init__.py       # Page exports
│   ├── home.py           # Home/landing page
│   ├── transform.py      # Wraps TransformTool
│   ├── manual_coder.py   # Wraps ManualCoderTool
│   └── ...               # Other page wrappers
│
├── core/                  # Core utilities
│   ├── providers.py      # AI provider implementations
│   ├── sample_data.py    # Sample datasets for testing
│   ├── templates.py      # Prompt templates
│   └── prompt_registry.py
│
├── ui/                    # Reusable UI components
│   ├── components/       # Shared components
│   │   ├── download_buttons.py
│   │   ├── provider_selector.py
│   │   ├── model_selector.py
│   │   └── ...
│   └── state.py          # UI state management
│
├── database/              # Database layer
│   ├── db.py             # Database connection
│   ├── models.py         # Data models
│   └── migrations.py     # Schema migrations
│
└── errors/               # Error handling
```

## Creating a New Tool

### 1. Create the Tool Class

Create `tools/your_tool.py`:

```python
from typing import Dict, Any, Optional, Callable
import streamlit as st
from .base import BaseTool, ToolConfig, ToolResult

class YourTool(BaseTool):
    # Required metadata
    id = "your_tool"           # Unique identifier (snake_case)
    name = "Your Tool"         # Display name
    description = "What it does"
    icon = ":material/icon:"   # Material icon
    category = "Processing"    # Or "Analysis", "Generation", etc.

    def render_config(self) -> ToolConfig:
        """Render configuration UI, return ToolConfig"""
        # Use st.* to render UI
        # Return ToolConfig(is_valid=True/False, config_data={...})
        pass

    async def execute(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> ToolResult:
        """Execute the tool, return ToolResult"""
        # Do the work
        # Return ToolResult(success=True/False, data=..., stats={...})
        pass

    def render_results(self, result: ToolResult):
        """Optional: Custom result display"""
        # Override for custom result rendering
        pass
```

### 2. Register the Tool

In `tools/registry.py`, add to `register_default_tools()`:

```python
from .your_tool import YourTool

if not ToolRegistry.tool_exists("your_tool"):
    ToolRegistry.register(YourTool())
```

Also add import in `tools/__init__.py`:

```python
from .your_tool import YourTool
```

### 3. Create the Page Wrapper

Create `pages/your_tool.py`:

```python
import streamlit as st
import asyncio
from tools.registry import ToolRegistry, register_default_tools

def render():
    register_default_tools()
    tool = ToolRegistry.get_tool("your_tool")

    st.title(f"{tool.icon} {tool.name}")
    st.caption(tool.description)

    config = tool.render_config()

    if config.is_valid:
        if st.button("Run", type="primary"):
            result = asyncio.run(tool.execute(config.config_data))
            tool.render_results(result)
    else:
        if config.error_message:
            st.warning(config.error_message)
```

### 4. Add to Navigation

In `app.py`:

```python
from pages import your_tool

page_your_tool = st.Page(your_tool.render, title="Your Tool", icon=":material/icon:", url_path="your-tool")

pages = {
    "Main": [..., page_your_tool],
}
```

## Key Patterns & Conventions

### Session State

Use prefixed keys to avoid conflicts:

```python
# Good - prefixed with tool abbreviation
st.session_state["mc_current_row"]  # Manual Coder
st.session_state["aic_text_cols"]   # AI Coder
st.session_state["tf_data"]         # Transform

# Bad - generic names
st.session_state["data"]            # Will conflict!
```

### Streamlit Fragments

Use `@st.fragment` for parts that need fast updates without full page rerun:

```python
@st.fragment
def coding_interface():
    # This section reruns independently
    # Good for interactive elements that change frequently
    pass
```

### Auto-save Pattern

For tools with user work that shouldn't be lost:

```python
AUTOSAVE_FILE = Path(".tool_saves/autosave.json")

def _save_progress(self):
    """Save after every change"""
    AUTOSAVE_FILE.parent.mkdir(exist_ok=True)
    with open(AUTOSAVE_FILE, "w") as f:
        json.dump(data, f)

def _load_progress(self):
    """Load on init if file exists"""
    if AUTOSAVE_FILE.exists():
        with open(AUTOSAVE_FILE) as f:
            return json.load(f)
```

### UI Components

Reuse components from `ui/components/`:

```python
from ui.components.download_buttons import render_download_buttons
from ui.components.provider_selector import render_provider_selector

# In your render method
render_download_buttons(df, filename_prefix="results")
```

## Streamlit Tips

### Layouts

```python
# Columns
col1, col2 = st.columns([3, 1])  # 3:1 ratio

# Expander (collapsed by default)
with st.expander("Details", expanded=False):
    st.write("...")

# Container for grouping
with st.container():
    st.write("...")
```

### Callbacks vs Direct Updates

```python
# Use on_click for button callbacks
def handle_click():
    st.session_state["value"] += 1

st.button("Click", on_click=handle_click)

# Use st.rerun() sparingly - only when layout needs to change
if condition_changed:
    st.rerun()
```

### Keys

Every interactive element needs a unique key:

```python
# Dynamic keys for elements in loops
for i, item in enumerate(items):
    st.button(f"Button {i}", key=f"btn_{i}")
```

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_tools.py -v
```

## Common Pitfalls

1. **Don't use generic session state keys** - They will conflict with other tools
2. **Don't call `st.rerun()` in loops** - Causes infinite reruns
3. **Don't forget keys on interactive elements** - Causes duplicate widget errors
4. **Don't put `st.set_page_config()` after other st calls** - Must be first
5. **Don't block the main thread** - Use `async/await` for long operations

## Git Workflow

```bash
# Commit message format
Tool: short description

# Examples
Manual Coder: add autosave functionality
Generate: fix column detection
Transform: improve error handling
```

## Files to Ignore

Already in `.gitignore`:
- `.manual_coder_saves/` - Autosave data
- `handai_data.db` - Local database
- `.venv/` - Virtual environment
- `__pycache__/` - Python cache

## Getting Help

- Check existing tools for patterns (especially `manual_coder.py` for complex UI)
- Look at `ui/components/` for reusable UI pieces
- Reference Streamlit docs: https://docs.streamlit.io
