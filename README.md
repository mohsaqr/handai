# Handai: AI Data Transformer & Generator

A powerful AI-powered tool for **transforming existing datasets** and **generating synthetic data**. Available as both a web application (Streamlit) and a desktop application (Flet).

## Features

### Data Transformation
- Upload CSV, Excel, or JSON files
- Process each row with AI to extract, classify, summarize, or translate
- Parallel processing with configurable concurrency
- Auto-retry for failed requests with exponential backoff
- Real-time progress tracking
- Export results to CSV, Excel, or JSON

### Synthetic Data Generation
- Generate realistic synthetic datasets from natural language descriptions
- Define custom schemas or use free-form generation
- Variable cycling for diverse data
- Templates for common use cases (interviews, reviews, support tickets, etc.)

### Multi-Provider Support

| Provider | Type | Models |
|----------|------|--------|
| **OpenAI** | Cloud | GPT-4o, GPT-4o-mini, GPT-4-turbo, o1, o3-mini |
| **Anthropic** | Cloud | Claude Sonnet 4, Claude Opus 4 |
| **Google** | Cloud | Gemini 2.0 Flash, Gemini 1.5 Pro |
| **Groq** | Cloud | Llama 3.3 70B, Mixtral 8x7B |
| **Together AI** | Cloud | Llama, Mistral, Qwen models |
| **OpenRouter** | Cloud | Access to 100+ models |
| **LM Studio** | Local | Any local model |
| **Ollama** | Local | Llama, Mistral, Phi, Gemma, etc. |

### Session Management
- Persistent settings across sessions
- Session history with restore capability
- Detailed logging and error tracking
- Performance statistics

## Installation

### Web Version (Streamlit)

```bash
cd web
pip install -r requirements.txt
streamlit run app.py
```

### Desktop Version (Flet)

```bash
cd desktop
pip install -r requirements.txt
python main.py
```

#### Building Desktop App

To build a standalone macOS/Windows app:

```bash
cd desktop
pip install pyinstaller
pyinstaller HandAI.spec
```

The built app will be in `desktop/dist/`.

## Quick Start

### Transform Mode
1. Select your AI provider and enter API key (if required)
2. Upload a CSV/Excel/JSON file
3. Write instructions for the AI (e.g., "Classify the sentiment of each review as positive, negative, or neutral")
4. Click "Test" to process 10 rows, or "Full Run" for the entire dataset
5. Export results

### Generate Mode
1. Select your AI provider
2. Describe the data you want (e.g., "Customer profiles with name, email, age, city")
3. Set the number of rows to generate
4. Click "Generate"
5. Export the synthetic dataset

## Configuration

### Settings (Sidebar)
- **Provider**: Choose your AI provider
- **API Key**: Enter your API key (not needed for local models)
- **Model**: Select or enter custom model name
- **Temperature**: 0 = deterministic, 2 = creative
- **Max Tokens**: Maximum response length
- **Concurrency**: Parallel requests (higher = faster but may hit rate limits)
- **Auto-retry**: Automatically retry failed/empty results

## Output Format

All AI outputs are structured as JSON for easy tabular export:
- Flat key-value pairs suitable for CSV columns
- Consistent schema across all rows
- Error handling with clear status indicators

## Project Structure

```
handai/
├── web/                    # Streamlit web application
│   ├── app.py              # Main web app
│   ├── handai_db.py        # SQLite database module
│   ├── handai_errors.py    # Error classification
│   ├── pages/
│   │   └── 1_History.py    # Session history page
│   └── requirements.txt
│
├── desktop/                # Flet desktop application
│   ├── main.py             # Main desktop app
│   ├── handai_db.py        # Database module
│   ├── handai_errors.py    # Error handling
│   ├── HandAI.spec         # PyInstaller spec
│   └── requirements.txt
│
└── README.md
```

## Use Cases

- **Data Enrichment**: Add AI-generated columns to existing datasets
- **Classification**: Categorize text data (sentiment, topics, intent)
- **Extraction**: Pull structured info from unstructured text
- **Translation**: Batch translate content across languages
- **Summarization**: Create summaries of long text entries
- **Synthetic Data**: Generate test data, training datasets, or mock data
- **Data Augmentation**: Create variations of existing data for ML training

## Author & Maintainer

**Mohammed Saqr**
Professor of Computer Science
University of Eastern Finland

🌐 Website: [www.saqr.me](https://www.saqr.me)
📧 Contact: Available via website

## License

MIT License

---

*Handai - Transform and generate data with AI, effortlessly.*
