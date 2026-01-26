# Handai - AI Data Transformer & Generator

Transform existing datasets or generate synthetic data using multiple AI providers.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run handai_app.py
```

The app will open at http://localhost:8501

## Features

- **Transform Data**: Upload CSV/Excel/JSON and apply AI transformations to each row
- **Generate Data**: Create synthetic datasets from scratch with custom schemas
- **10 AI Providers**: OpenAI, Anthropic, Google Gemini, Groq, Together AI, Azure, OpenRouter, LM Studio, Ollama, Custom
- **Session History**: Full audit trail of all runs with logs and results
- **Auto-retry**: Automatically retries failed/empty results
- **Persistent Settings**: Your configuration is saved between sessions

## Requirements

- Python 3.9+
- An API key from your chosen provider (not needed for local models)

## Installation

### Option 1: Quick Install (Recommended)

```bash
cd handai
pip install -r requirements.txt
streamlit run handai_app.py
```

### Option 2: With Virtual Environment

```bash
cd handai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run handai_app.py
```

## Supported Providers

| Provider | API Key Required | Notes |
|----------|-----------------|-------|
| OpenAI | Yes | GPT-4o, GPT-4, GPT-3.5 |
| Anthropic | Yes | Claude Sonnet, Opus, Haiku |
| Google Gemini | Yes | Gemini 2.0, 1.5 Pro/Flash |
| Groq | Yes | Ultra-fast inference |
| Together AI | Yes | Wide model selection |
| Azure OpenAI | Yes | Enterprise deployment |
| OpenRouter | Yes | Access multiple providers |
| LM Studio | No | Local models |
| Ollama | No | Local models |
| Custom | No | Any OpenAI-compatible API |

## Usage

### Transform Mode
1. Upload a CSV, Excel, or JSON file
2. Select which columns to send to the AI
3. Write instructions for the transformation
4. Click "Test" to try on a few rows, then "Full Run"

### Generate Mode
1. Describe what data you want to generate
2. Choose schema: Free-form, Custom Fields, or Template
3. Set number of rows and variation level
4. Click "Generate"

## Files

```
handai/
├── handai_app.py      # Main application
├── handai_db.py       # Database & persistence
├── handai_errors.py   # Error handling
├── handai_data.db     # SQLite database (auto-created)
├── requirements.txt   # Dependencies
└── pages/
    └── 1_History.py   # History browser
```

## Troubleshooting

**App won't start?**
```bash
pip install --upgrade streamlit
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**Port 8501 in use?**
```bash
streamlit run handai_app.py --server.port 8502
```

## License

MIT
