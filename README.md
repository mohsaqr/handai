# Handai - AI Data Transformer & Qualitative Analysis Suite

Transform datasets, generate synthetic data, and code qualitative data using multiple AI providers.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

## Desktop App (Electron)

You can run the Streamlit app inside a desktop shell for macOS, Windows, and Linux.

```bash
cd desktop
npm install
npm run start
```

To build installers:

```bash
./desktop/scripts/download_python_standalone.sh
./desktop/scripts/bundle_python.sh
./desktop/scripts/install_python_deps.sh
cd desktop
npm install
npm run build
```

Or run everything in one go:

```bash
./desktop/scripts/build_desktop.sh
```

Notes:
- For fully self-contained installers, place bundled Python builds in `desktop/python` (see `desktop/python/README.md`).
- If no bundled Python is present, it falls back to system Python 3.9+.
- If Python isn't on PATH, set `HANDAI_PYTHON` to the python executable path.

## Features

### Data Processing Tools
- **Transform Data**: Upload CSV/Excel/JSON and apply AI transformations to each row
- **Generate Data**: Create synthetic datasets from scratch with custom schemas
- **Process Documents**: Extract structured data from PDFs, DOCX, and text files
- **Automator**: Build multi-step AI pipelines with branching logic

### Qualitative Analysis Tools
- **Qualitative Coder**: AI-powered coding of interviews, surveys, and observations
- **Consensus Coder**: Run multiple AI models in parallel with inter-rater reliability analytics
- **Codebook Generator**: Automatically generate structured codebooks from your data
- **Manual Coder**: Human coding interface with immersive mode, clickable codes, and session management

### System Features
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
streamlit run app.py
```

### Option 2: With Virtual Environment

```bash
cd handai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
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

### Manual Coder
1. Upload your dataset or use sample data
2. Define your codes (manually or via codebook upload)
3. Click through rows and apply codes with one click
4. Use Immersive Mode for distraction-free coding
5. Sessions auto-save and restore on page refresh

## Project Structure

```
handai/
├── app.py                 # Main application entry point
├── config.py              # Configuration management
├── database.py            # SQLite database & persistence
├── requirements.txt       # Dependencies
├── pages/                 # Page modules
│   ├── home.py           # Landing page
│   ├── transform.py      # Transform Data tool
│   ├── generate.py       # Generate Data tool
│   ├── process_documents.py
│   ├── qualitative.py    # Qualitative Coder
│   ├── consensus.py      # Consensus Coder
│   ├── codebook_generator.py
│   ├── manual_coder.py   # Manual Coder page
│   ├── automator.py
│   ├── history.py
│   ├── settings.py
│   └── models.py         # LLM Providers
├── tools/                 # Tool implementations
│   ├── manual_coder.py   # Manual Coder tool
│   └── ...
├── core/                  # Core modules
│   ├── llm.py            # LLM client
│   ├── sample_data.py    # Sample datasets
│   └── ...
└── ui/                    # UI components
    └── components/
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
streamlit run app.py --server.port 8502
```

## License

MIT
