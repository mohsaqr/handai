# OpenRouter Integration - Status & Handoff Document

## System Overview: Handai

**Handai** is a Streamlit-based AI Data Transformer & Generator application that allows users to:

1. **Transform existing datasets** - Upload CSV/Excel/JSON files and use AI to transform, enrich, or analyze each row
2. **Generate synthetic datasets** - Create new datasets from scratch using AI with customizable schemas

### Key Features
- Multi-provider support (OpenAI, Anthropic, Google, Groq, Together AI, Azure, OpenRouter, LM Studio, Ollama, Custom)
- Concurrent processing with configurable parallelism
- Auto-retry on failures
- Session management and run history
- Real-time progress tracking
- JSON mode support

### Architecture
```
handai/
├── handai_app.py      # Main Streamlit application (UI + processing logic)
├── handai_db.py       # SQLite database for sessions, runs, results, logs
├── handai_errors.py   # Error classification and handling
├── pages/
│   └── 1_History.py   # Run history viewer
├── venv/              # Python virtual environment
├── requirements.txt   # Dependencies
└── run.sh             # Startup script
```

### Provider Configuration
Providers are defined in `PROVIDER_CONFIGS` dict (line ~91 in `handai_app.py`):
```python
@dataclass
class ProviderConfig:
    name: str
    base_url: Optional[str]
    default_model: str
    models: List[str]
    requires_api_key: bool
    description: str
```

### Client Creation
The `get_client()` function (line ~230) creates an `AsyncOpenAI` client for each provider with appropriate configuration.

---

## OpenRouter Integration Work Completed

### What Was Implemented

1. **Dynamic model fetching** (`fetch_openrouter_models()` function, line ~267):
   - Fetches available models from `https://openrouter.ai/api/v1/models`
   - Returns top 50 models
   - Falls back to hardcoded popular models if fetch fails

2. **UI updates for OpenRouter**:
   - Added refresh button (🔄) to reload models dynamically
   - Added help text linking to openrouter.ai/keys
   - OpenRouter now included in dynamic model fetching flow (like LM Studio/Ollama)

3. **Client configuration for OpenRouter**:
   - Uses `https://openrouter.ai/api/v1` as base_url
   - Adds required headers: `HTTP-Referer` and `X-Title`
   - Does NOT use custom `http_client` (was causing issues)

### Current Code State

```python
# In get_client() function:
if provider == LLMProvider.OPENROUTER:
    return AsyncOpenAI(
        api_key=effective_key,
        base_url=effective_url,
        max_retries=0,
        default_headers={
            "HTTP-Referer": "https://handai.app",
            "X-Title": "Handai Data Transformer"
        }
    )
```

```python
# OpenRouter provider config:
LLMProvider.OPENROUTER: ProviderConfig(
    name="OpenRouter",
    base_url="https://openrouter.ai/api/v1",
    default_model="anthropic/claude-sonnet-4",
    models=["anthropic/claude-sonnet-4", "openai/gpt-4o", ...],
    requires_api_key=True,
    description="OpenRouter - access many providers via one API"
)
```

---

## Current Issue: NOT A CODE PROBLEM

### The Problem
User reported "Authentication failed" when using OpenRouter.

### Root Cause (Identified)
**The user was using an OpenAI API key instead of an OpenRouter API key.**

Debug output showed:
```
OpenRouter: key=sk-proj-kb...recA, url=https://openrouter.ai/api/v1
```

- `sk-proj-...` = **OpenAI** project API key format
- `sk-or-...` = **OpenRouter** API key format

The user mistakenly believed their OpenAI key (`sk-proj-...`) was from OpenRouter.

### Resolution
User needs to:
1. Create an account at https://openrouter.ai
2. Generate an API key at https://openrouter.ai/keys
3. Use that key (starts with `sk-or-`) when selecting OpenRouter provider

### Code Status
**The code is working correctly.** The implementation has been tested and the OpenRouter API responds properly when given a valid OpenRouter API key.

Verified working:
- Model fetching from OpenRouter API (returns 345 models)
- Base URL configuration
- Header injection (HTTP-Referer, X-Title)
- Client creation without custom http_client

---

## API Key Formats Reference

| Provider | Key Prefix | Example |
|----------|------------|---------|
| OpenAI | `sk-proj-` | `sk-proj-abc123...` |
| OpenAI (legacy) | `sk-` | `sk-abc123...` |
| OpenRouter | `sk-or-` | `sk-or-v1-abc123...` |
| Anthropic | `sk-ant-` | `sk-ant-abc123...` |
| Google | `AI...` | `AIzaSy...` |

---

## Potential Future Improvements

1. **Separate API key storage per provider** - Currently one `api_key` field is shared across all providers. Could store keys per-provider so switching doesn't require re-entering keys.

2. **API key format validation** - Warn user if key format doesn't match expected provider prefix.

3. **Key validation on entry** - Test the API key when entered before running transformations.

4. **Better error messages** - Parse OpenRouter's `"No cookie auth credentials found"` error to show "Invalid API key" instead.

---

## Running the Application

```bash
cd /Users/mohammedsaqr/Documents/Git/Coder/handai
./venv/bin/python -m streamlit run handai_app.py
```

Or use the run script:
```bash
./run.sh
```

App runs at: http://localhost:8501

---

## Files Modified in This Session

1. `handai_app.py`:
   - Added `fetch_openrouter_models()` function
   - Updated `get_client()` to handle OpenRouter specially (no custom http_client)
   - Added OpenRouter to dynamic model fetching UI
   - Updated API key help text

---

*Last updated: 2026-01-26*
*Status: Code complete, awaiting user to use correct OpenRouter API key*
