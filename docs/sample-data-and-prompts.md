# Sample Data & Prompts Integration

This document describes the sample data infrastructure and prompt templates added to Handai for testing and demonstration purposes.

---

## Overview

All tools now include a **"Use Sample Data"** button that loads pre-built datasets or prompts for immediate testing without requiring users to prepare their own data.

---

## Sample Datasets

Located in `core/sample_data.py`, the module provides 9 comprehensive datasets:

### Qualitative Research Datasets

| Dataset | Rows | Description | Best For |
|---------|------|-------------|----------|
| `product_reviews` | 20 | E-commerce product reviews with varied sentiment | Sentiment analysis, opinion mining |
| `healthcare_interviews` | 15 | Healthcare worker interview excerpts | Thematic analysis, workplace research |
| `support_tickets` | 20 | Customer support conversations | Issue categorization, satisfaction analysis |
| `learning_experience` | 20 | Student course feedback responses | Educational research, experience mapping |
| `exit_interviews` | 15 | Employee departure interview responses | HR analytics, retention research |
| `social_media_posts` | 25 | Social media content with engagement metrics | Content analysis, trend identification |
| `research_abstracts` | 15 | Academic paper abstracts | Literature review, topic modeling |
| `patient_feedback` | 20 | Healthcare patient experience feedback | Healthcare quality research |
| `focus_group_excerpts` | 12 | Focus group discussion segments | Group dynamics, consensus research |

### API Functions

```python
from core.sample_data import get_sample_data, get_dataset_info, get_available_datasets

# Get list of all dataset names
datasets = get_available_datasets()

# Get metadata about all datasets
info = get_dataset_info()
# Returns: {"dataset_name": {"name": "...", "description": "...", "rows": N, "columns": [...]}}

# Load a specific dataset
data = get_sample_data("product_reviews")
# Returns: List of dictionaries ready for pd.DataFrame()
```

---

## Tool Integration

### Qualitative Coder

**Sample Datasets Available:**
- Product Reviews (20 reviews with sentiment)
- Healthcare Interviews (15 worker experiences)
- Support Tickets (20 customer issues)
- Learning Experience (20 student responses)
- Exit Interviews (15 employee departures)

**Usage:** Click "Use Sample Data" → Select dataset from dropdown

### Consensus Coder

**Sample Datasets Available:**
- Product Reviews (20 reviews) - Multi-coder sentiment agreement
- Healthcare Interviews (15 excerpts) - Theme consensus coding
- Focus Group Excerpts (12 segments) - Group analysis consensus

**Usage:** Click "Use Sample Data" → Select dataset from dropdown

### Codebook Generator

**Sample Datasets Available:**
- Healthcare Interviews (15 excerpts) - Clinical themes
- Learning Experience (20 responses) - Educational themes
- Exit Interviews (15 responses) - Workplace themes

**Usage:** Click "Use Sample Data" → Select dataset from dropdown

### Automator

**Sample Datasets Available:**
- Product Reviews (20 reviews) - Batch sentiment analysis
- Support Tickets (20 tickets) - Issue categorization
- Research Abstracts (15 abstracts) - Literature processing
- Patient Feedback (20 responses) - Healthcare analysis

**Usage:** Click "Use Sample Data" → Select dataset from dropdown

### Generate (Synthetic Data)

**Sample Prompts Available** (instead of datasets):

| Prompt | Description |
|--------|-------------|
| Customer Profiles | E-commerce customer data with demographics |
| Product Catalog | Electronics store inventory |
| Employee Directory | Tech company personnel records |
| Medical Records | Synthetic patient health data |
| Financial Transactions | Banking transaction records |
| Survey Responses | Market research survey data |
| Event Logs | System/application event data |
| Social Media Posts | Synthetic social content |

**Usage:** Select a sample prompt from dropdown → Edit as needed → Generate

---

## System Prompts

See [system-prompts.md](system-prompts.md) for documentation on:
- Customizable AI prompts for all tools
- Standard vs. rigorous prompt variants
- Session (temporary) and permanent overrides
- The System Prompts settings tab

### Prompt Categories

| Category | Prompts | Description |
|----------|---------|-------------|
| Qualitative | 2 | Default + rigorous coding prompts |
| Consensus | 4 | Worker + judge prompts (standard & rigorous) |
| Codebook | 6 | Theme discovery, consolidation, code definition |
| Generate | 2 | Column suggestion prompts |
| Documents | 2 | CSV output enforcement prompts |
| Automator | 2 | Critical rules prompts |

---

## Data Quality

All sample datasets are designed with:

- **Realistic content** - Reflects actual data patterns users would encounter
- **Varied complexity** - Mix of simple and nuanced examples
- **Edge cases** - Includes ambiguous cases to test AI handling
- **Balanced distribution** - Covers different categories/sentiments
- **Appropriate length** - Matches typical real-world data sizes

---

## Adding New Datasets

To add a new sample dataset:

1. Add data to `core/sample_data.py`:

```python
SAMPLE_DATASETS["my_dataset"] = {
    "name": "My Dataset",
    "description": "Description of the dataset",
    "columns": ["col1", "col2", "col3"],
    "data": [
        {"col1": "value1", "col2": "value2", "col3": "value3"},
        # ... more rows
    ]
}
```

2. Update the tool's sample selector in `tools/<tool>.py`:

```python
sample_options = {
    # ... existing options
    "my_dataset": "My Dataset (N rows description)",
}
```

---

## Files Modified

| File | Changes |
|------|---------|
| `core/sample_data.py` | New module with 9 datasets |
| `tools/qualitative.py` | Sample dataset selector (5 options) |
| `tools/consensus.py` | Sample dataset selector (3 options) |
| `tools/codebook_generator.py` | Sample dataset selector (3 options) |
| `tools/automator.py` | Sample dataset selector (4 options) |
| `tools/generate.py` | Sample prompts dropdown (8 options) |
