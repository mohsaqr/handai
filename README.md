# Handai

AI-Powered Data Transformation & Qualitative Analysis Suite

## Features

**Data Processing**
- **Transform Data** - AI-powered transformation, enrichment, and classification of CSV/Excel data
- **Generate Data** - Create synthetic datasets with custom schemas
- **Process Documents** - Extract structured data from PDFs and Word documents
- **Automator** - Build multi-step AI pipelines

**Qualitative Analysis**
- **Qualitative Coder** - AI-powered coding of interviews and surveys
- **Consensus Coder** - Multi-model consensus with inter-rater reliability analytics
- **Manual Coder** - Human coding interface with session management
- **AI Coder** - AI-assisted coding with human review
- **Model Comparison** - Compare outputs across multiple LLMs
- **Codebook Generator** - Auto-generate codebooks from data

**Supported Providers**
- OpenAI, Anthropic, Google Gemini, Groq, Together AI, Azure, OpenRouter
- Local models: Ollama, LM Studio
- Any OpenAI-compatible API

---

## Installation

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Clone and setup
git clone https://github.com/mohsaqr/handai.git
cd handai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

### Windows

```powershell
# Install Python from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH" during installation

# Clone and setup (PowerShell)
git clone https://github.com/mohsaqr/handai.git
cd handai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone and setup
git clone https://github.com/mohsaqr/handai.git
cd handai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install python3 python3-pip git

# Clone and setup
git clone https://github.com/mohsaqr/handai.git
cd handai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## Quick Start

1. Open the app at **http://localhost:8501**
2. Go to **LLM Providers** and add your API key
3. Select a tool from the sidebar
4. Upload data or start generating

---

## Running Tests

```bash
# Activate environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=term-missing
```

---

## Troubleshooting

**"streamlit: command not found"**
```bash
# Activate the virtual environment first
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

**Module not found errors**
```bash
pip install -r requirements.txt
```

**Port 8501 already in use**
```bash
streamlit run app.py --server.port 8502
```

**Permission denied (Linux)**
```bash
# Use python3 explicitly
python3 -m venv .venv
```

---

## License

MIT
