# Consensus Coder - Enhanced Judge Features

## Overview

The Consensus Coder tool enables multi-model consensus coding where multiple AI workers process the same data independently, and a judge model synthesizes the best answer. The enhanced judge features add detailed quality evaluation capabilities.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Feature Summary](#feature-summary)
3. [File Structure](#file-structure)
4. [Data Flow](#data-flow)
5. [Configuration Options](#configuration-options)
6. [Output Schema](#output-schema)
7. [Analytics & Visualizations](#analytics--visualizations)
8. [Implementation Details](#implementation-details)
9. [Testing](#testing)
10. [Future Enhancements (TODOs)](#future-enhancements-todos)
11. [API Reference](#api-reference)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Consensus Coder UI                          │
│                      (tools/consensus.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Data Upload │  │   Model     │  │     Options Panel           │  │
│  │   Panel     │  │  Selectors  │  │  ☑ Include Reasoning        │  │
│  │             │  │  (Workers   │  │  ☑ Enable Quality Scoring   │  │
│  │  CSV/Excel  │  │   + Judge)  │  │  ☑ Enable Disagreement      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ConsensusProcessor                             │
│                    (core/processing.py)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   For each row:                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│   │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  (parallel)             │
│   │  (LLM)   │  │  (LLM)   │  │  (LLM)   │                         │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                         │
│        │             │             │                                │
│        └─────────────┼─────────────┘                                │
│                      ▼                                              │
│              ┌──────────────┐                                       │
│              │    Judge     │                                       │
│              │    (LLM)     │                                       │
│              │              │                                       │
│              │ Enhanced     │                                       │
│              │ Evaluation:  │                                       │
│              │ • Scores     │                                       │
│              │ • Ranks      │                                       │
│              │ • Confidence │                                       │
│              │ • Disagree.  │                                       │
│              └──────────────┘                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Results & Analytics                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌────────────────────────────────────────┐  │
│  │  Results Table   │  │        Analytics Panels                │  │
│  │                  │  │  ┌─────────────────────────────────┐   │  │
│  │ • worker outputs │  │  │ Inter-Rater Agreement Analysis  │   │  │
│  │ • judge_consensus│  │  │ • Pairwise Agreement            │   │  │
│  │ • best_answer    │  │  │ • Cohen's Kappa                 │   │  │
│  │ • confidence     │  │  │ • Jaccard Index                 │   │  │
│  │ • worker scores  │  │  └─────────────────────────────────┘   │  │
│  │ • worker ranks   │  │  ┌─────────────────────────────────┐   │  │
│  │ • disagreements  │  │  │ Enhanced Judge Analytics        │   │  │
│  │                  │  │  │ • Quality Leaderboard           │   │  │
│  └──────────────────┘  │  │ • Rank Distribution             │   │  │
│                        │  │ • Confidence Stats              │   │  │
│                        │  │ • Disagreement Hotspots         │   │  │
│                        │  └─────────────────────────────────┘   │  │
│                        └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Summary

### Core Features (Existing)
| Feature | Description |
|---------|-------------|
| Multi-worker processing | 2-3 AI workers process each row independently |
| Judge synthesis | Judge model combines worker outputs into best answer |
| Consensus status | Yes/No/Partial consensus indicator |
| Basic reasoning | Judge explains decision |
| Inter-rater analytics | Pairwise agreement, Cohen's Kappa, Jaccard index |

### Enhanced Features (New)
| Feature | Description | Config Option |
|---------|-------------|---------------|
| Quality Scoring | Judge rates each worker 1-5 stars | `enable_quality_scoring` |
| Worker Rankings | Judge ranks workers 1st, 2nd, 3rd per row | `enable_quality_scoring` |
| Confidence Scores | Judge provides 0-100% confidence | Auto with either option |
| Disagreement Analysis | Detailed explanation of differences | `enable_disagreement_analysis` |
| Quality Leaderboard | Average scores visualization | Auto when scoring enabled |
| Rank Distribution | How often each worker ranked 1st/2nd/3rd | Auto when scoring enabled |
| Confidence Histogram | Distribution of judge confidence | Auto when either enabled |
| Disagreement Hotspots | Common disagreement patterns | Auto when analysis enabled |

---

## File Structure

```
handai/
├── core/
│   ├── processing.py              # ConsensusConfig, ConsensusProcessor
│   ├── comparison_analytics.py    # Quality metrics calculations
│   └── prompt_registry.py         # Enhanced judge prompt
├── tools/
│   └── consensus.py               # UI, render_config, render_results
├── tests/
│   ├── conftest.py                # Shared fixtures including consensus fixtures
│   ├── mocks/
│   │   └── mock_llm_client.py     # Mock LLM client with error injection
│   └── unit/
│       └── test_consensus.py      # 124 comprehensive tests
└── docs/
    └── CONSENSUS_ENHANCED_JUDGE.md  # This document
```

### File Responsibilities

#### `core/processing.py`
- **ConsensusConfig** dataclass with configuration options
- **ConsensusProcessor** class that orchestrates worker/judge calls
- Enhanced JSON instruction building
- Result field extraction and parsing

#### `core/comparison_analytics.py`
- **QualityMetrics** dataclass
- `calculate_average_scores()` - Per-worker score averages
- `calculate_rank_distribution()` - Rank frequency counts
- `calculate_confidence_stats()` - Statistical measures
- `extract_disagreement_patterns()` - Pattern extraction
- `render_quality_metrics()` - Streamlit visualizations

#### `tools/consensus.py`
- **ConsensusTool** class (BaseTool subclass)
- `render_config()` - UI for configuration
- `execute()` - Async processing execution
- `render_results()` - Results display and analytics
- `_render_analytics()` - Inter-rater agreement charts
- `_render_quality_analytics_from_df()` - Fallback analytics renderer

#### `core/prompt_registry.py`
- `consensus.judge_prompt.enhanced` - Full enhanced judge prompt
- Scoring guide and consensus level definitions

---

## Data Flow

### 1. Configuration Phase
```
User Input → render_config() → ToolConfig
                                   │
                                   ├── df (DataFrame)
                                   ├── worker_configs (List[Dict])
                                   ├── judge_config (Dict)
                                   ├── worker_prompt (str)
                                   ├── judge_prompt (str)
                                   ├── include_reasoning (bool)
                                   ├── enable_quality_scoring (bool)    ← NEW
                                   └── enable_disagreement_analysis (bool) ← NEW
```

### 2. Processing Phase
```
ToolConfig → ConsensusProcessor.process()
                    │
                    ├── For each row:
                    │   ├── Call workers in parallel
                    │   ├── Build judge context
                    │   ├── Build JSON instruction (enhanced if options enabled)
                    │   ├── Call judge
                    │   ├── Parse JSON response
                    │   └── Extract enhanced fields:
                    │       ├── judge_confidence
                    │       ├── worker_N_score
                    │       ├── worker_N_rank
                    │       ├── disagreement_summary
                    │       └── disagreements_raw
                    │
                    └── Return ProcessingResult
```

### 3. Results Phase
```
ProcessingResult → execute() → ToolResult
                                   │
                                   ├── data (DataFrame with new columns)
                                   ├── stats (includes enable flags)
                                   └── metadata (raw_results for analytics)
                                         │
                                         ▼
                               render_results()
                                         │
                                         ├── Results DataFrame
                                         ├── Inter-Rater Analytics
                                         ├── Enhanced Judge Analytics ← NEW
                                         │   ├── Quality Leaderboard
                                         │   ├── Rank Distribution
                                         │   ├── Confidence Stats
                                         │   └── Disagreement Hotspots
                                         ├── Result Inspector
                                         └── Download Buttons
```

---

## Configuration Options

### UI Checkboxes

```python
# In render_config()
include_reasoning = st.checkbox("Include Judge Reasoning", value=True)

st.markdown("**Enhanced Judge Features**")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    enable_quality_scoring = st.checkbox(
        "Enable Quality Scoring",
        value=False,
        help="Judge rates each worker's output (1-5 stars) and ranks them"
    )
with col_opt2:
    enable_disagreement_analysis = st.checkbox(
        "Enable Disagreement Analysis",
        value=False,
        help="Judge provides detailed explanation of where/why workers differ"
    )
```

### ConsensusConfig Dataclass

```python
@dataclass
class ConsensusConfig:
    worker_configs: List[Dict[str, Any]]
    judge_config: Dict[str, Any]
    max_concurrency: int
    auto_retry: bool
    max_retries: int
    save_path: Optional[str]
    realtime_progress: bool
    include_reasoning: bool
    enable_quality_scoring: bool = False      # NEW
    enable_disagreement_analysis: bool = False # NEW
```

---

## Output Schema

### Standard Judge Response (Existing)
```json
{
  "consensus": "Yes|No|Partial",
  "best_answer": "value1,value2,value3",
  "reasoning": "Brief explanation"
}
```

### Enhanced Judge Response (New)
```json
{
  "consensus": "Full|Majority|Partial|None",
  "confidence": 85,
  "best_answer": "value1,value2,value3",
  "reasoning": "Brief explanation",
  "worker_evaluations": {
    "worker_1": {
      "score": 5,
      "rank": 1,
      "notes": "Excellent - correct format, complete answer"
    },
    "worker_2": {
      "score": 4,
      "rank": 2,
      "notes": "Good - minor formatting issues"
    },
    "worker_3": {
      "score": 3,
      "rank": 3,
      "notes": "Acceptable - missing some details"
    }
  },
  "disagreements": [
    {
      "aspect": "sentiment classification",
      "details": "Worker 1 classified as positive, Worker 2 as neutral"
    },
    {
      "aspect": "category assignment",
      "details": "Workers disagreed on primary vs secondary category"
    }
  ]
}
```

### Result DataFrame Columns

| Column | Type | Condition | Description |
|--------|------|-----------|-------------|
| `worker_1_output` | str | Always | Worker 1's raw output |
| `worker_1_latency_s` | float | Always | Worker 1's response time |
| `worker_2_output` | str | Always | Worker 2's raw output |
| `worker_2_latency_s` | float | Always | Worker 2's response time |
| `worker_3_output` | str | If 3 workers | Worker 3's raw output |
| `worker_3_latency_s` | float | If 3 workers | Worker 3's response time |
| `judge_consensus` | str | Always | Full/Majority/Partial/None |
| `judge_best_answer` | str | Always | Synthesized best answer |
| `judge_reasoning` | str | If reasoning enabled | Judge's explanation |
| `judge_latency_s` | float | Always | Judge's response time |
| `judge_confidence` | int | If enhanced | 0-100% confidence |
| `worker_1_score` | int | If quality enabled | 1-5 quality score |
| `worker_1_rank` | int | If quality enabled | 1-N rank position |
| `worker_2_score` | int | If quality enabled | 1-5 quality score |
| `worker_2_rank` | int | If quality enabled | 1-N rank position |
| `worker_3_score` | int | If quality & 3 workers | 1-5 quality score |
| `worker_3_rank` | int | If quality & 3 workers | 1-N rank position |
| `disagreement_summary` | str | If disagreement enabled | Brief summary |

---

## Analytics & Visualizations

### 1. Model Quality Leaderboard

**Data Source**: `worker_N_score` columns
**Visualization**: Horizontal bar chart
**Metric**: Average score per worker (1-5 scale)

```
Worker 1: ████████████████████ 4.5
Worker 2: ████████████████     4.0
Worker 3: ████████████         3.5
```

### 2. Rank Distribution

**Data Source**: `worker_N_rank` columns
**Visualization**: Grouped bar chart
**Metric**: Count of 1st/2nd/3rd place finishes

```
        1st   2nd   3rd
Worker1 [60%] [30%] [10%]
Worker2 [25%] [50%] [25%]
Worker3 [15%] [20%] [65%]
```

### 3. Confidence Statistics

**Data Source**: `judge_confidence` column
**Visualization**: Metric cards
**Metrics**: Mean, Median, Min, Max, Std Dev

```
┌────────┬────────┬────────┬────────┬────────┐
│  Mean  │ Median │  Min   │  Max   │ Std Dev│
│ 78.5%  │ 82.0%  │ 45.0%  │ 100%   │ 12.3   │
└────────┴────────┴────────┴────────┴────────┘
```

### 4. Disagreement Hotspots

**Data Source**: `disagreements_raw` JSON field
**Visualization**: Data table
**Metrics**: Aspect, Occurrences, Frequency, Example

| Aspect | Occurrences | Frequency | Example |
|--------|-------------|-----------|---------|
| sentiment | 35 | 35.0% | "positive vs neutral interpretation" |
| category | 22 | 22.0% | "primary vs secondary classification" |
| confidence | 18 | 18.0% | "high vs medium certainty" |

---

## Implementation Details

### Enhanced JSON Instruction Building

```python
# In ConsensusProcessor.process_row()

use_enhanced = self.config.enable_quality_scoring or self.config.enable_disagreement_analysis

if use_enhanced:
    json_instruction = """
CRITICAL: Return a VALID JSON object only. No markdown.
Required keys:
- "consensus": "Full|Majority|Partial|None"
- "confidence": 0-100 (your confidence in the decision)
- "best_answer": CSV format only
- "reasoning": brief explanation"""

    if self.config.enable_quality_scoring:
        json_instruction += """
- "worker_evaluations": {"worker_1": {"score": 1-5, "rank": 1-N, "notes": "brief"}, ...}"""

    if self.config.enable_disagreement_analysis:
        json_instruction += """
- "disagreements": [{"aspect": "what differs", "details": "explanation"}, ...]"""
else:
    # Standard instruction
    json_instruction = """
CRITICAL: Return a VALID JSON object only. No markdown.
Keys: "consensus" (Yes/No/Partial), "best_answer" (CSV format only), "reasoning" (brief)."""
```

### Result Field Extraction

```python
# After parsing judge JSON response

if parsed:
    results["judge_consensus"] = parsed.get("consensus", "Partial")
    results["judge_best_answer"] = parsed.get("best_answer", judge_output[:500])
    results["judge_reasoning"] = parsed.get("reasoning", "N/A")

    # Enhanced fields
    if use_enhanced:
        results["judge_confidence"] = parsed.get("confidence", 50)

    if self.config.enable_quality_scoring:
        worker_evals = parsed.get("worker_evaluations", {})
        for w_name in worker_clients.keys():
            eval_data = worker_evals.get(w_name, {})
            results[f"{w_name}_score"] = eval_data.get("score", 3)
            results[f"{w_name}_rank"] = eval_data.get("rank", 0)

    if self.config.enable_disagreement_analysis:
        disagreements = parsed.get("disagreements", [])
        if disagreements:
            summary_parts = [d.get("aspect", "unknown") for d in disagreements[:3]]
            results["disagreement_summary"] = "; ".join(summary_parts)
            results["disagreements_raw"] = json.dumps(disagreements)
        else:
            results["disagreement_summary"] = ""
            results["disagreements_raw"] = "[]"
```

### Analytics Calculation

```python
# In core/comparison_analytics.py

def calculate_quality_metrics(results: List[Dict]) -> QualityMetrics:
    return QualityMetrics(
        average_scores=calculate_average_scores(results),
        rank_distribution=calculate_rank_distribution(results),
        confidence_stats=calculate_confidence_stats(results),
        disagreement_patterns=extract_disagreement_patterns(results)
    )
```

---

## Testing

### Test File: `tests/unit/test_consensus.py`

**Total Tests**: 124 tests (comprehensive coverage)

#### Test Classes

| Class | Tests | Description |
|-------|-------|-------------|
| `TestCalculateAverageScores` | 12 | Score averaging with edge cases, 3+ workers, performance |
| `TestCalculateRankDistribution` | 9 | Rank counting with ties, invalid values, 3 workers |
| `TestCalculateConfidenceStats` | 9 | Statistical calculations, boundary values, float precision |
| `TestExtractDisagreementPatterns` | 12 | Pattern extraction, Unicode, nested JSON, limits |
| `TestCalculateQualityMetrics` | 1 | Full metrics calculation |
| `TestEnhancedJudgeParsing` | 2 | Enhanced response parsing |
| `TestJSONParsingEdgeCases` | 15 | Markdown removal, leading/trailing text, escapes |
| `TestBackwardsCompatibility` | 8 | Old format support, field defaults |
| `TestConsensusConfigDataclass` | 2 | Config dataclass defaults |
| `TestQualityMetricsDataclass` | 5 | Type verification for dataclass fields |
| `TestMockJudgeResponses` | 10 | Various consensus/confidence scenarios |
| `TestErrorScenarios` | 10 | Error types (429, 401, timeout, etc.) + async retry |
| `TestPropertyBased` | 6 | Hypothesis property-based tests (graceful skip) |
| `TestPerformance` | 5 | 1000 rows, 10 workers, stress tests |
| `TestConsensusProcessorIntegration` | 10 | Integration with mock configs |
| `TestRenderQualityMetrics` | 8 | Rendering and visualization tests |

#### Test Categories

- **Unit Tests**: Analytics functions with edge cases and boundary conditions
- **JSON Parsing**: Robustness tests for judge response parsing
- **Backwards Compatibility**: Old format support and default values
- **Error Handling**: All mock error types and graceful degradation
- **Property-Based**: Hypothesis tests for invariants (requires `hypothesis`)
- **Performance**: Large datasets (1000+ rows) and stress testing
- **Integration**: ConsensusProcessor configuration tests

#### Running Tests

```bash
# Run all consensus tests
.venv/bin/python -m pytest tests/unit/test_consensus.py -v

# Run specific test class
.venv/bin/python -m pytest tests/unit/test_consensus.py::TestCalculateAverageScores -v

# Run with coverage
.venv/bin/python -m pytest tests/unit/test_consensus.py --cov=core.comparison_analytics --cov-report=term-missing

# Run property-based tests only (requires hypothesis)
.venv/bin/python -m pytest tests/unit/test_consensus.py::TestPropertyBased -v

# Run performance tests only
.venv/bin/python -m pytest tests/unit/test_consensus.py::TestPerformance -v
```

#### Test Dependencies

Install testing dependencies:
```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov hypothesis
```

#### Test Coverage

| Module | Coverage |
|--------|----------|
| `core/comparison_analytics.py` | >95% |
| `core/processing.py` (consensus sections) | >80% |

| Function | Coverage |
|----------|----------|
| `calculate_average_scores` | 100% |
| `calculate_rank_distribution` | 100% |
| `calculate_confidence_stats` | 100% |
| `extract_disagreement_patterns` | 100% |
| `calculate_quality_metrics` | 100% |

---

## Future Enhancements (TODOs)

### High Priority

- [ ] **Confidence Histogram Visualization**
  - Add plotly histogram for confidence distribution
  - Show distribution buckets (0-20%, 20-40%, etc.)
  - File: `core/comparison_analytics.py`

- [ ] **Export Enhanced Analytics**
  - Add JSON export of quality metrics
  - Include in download options
  - File: `tools/consensus.py`

- [ ] **Persistent Prompt Selection**
  - Allow selecting enhanced vs standard judge prompt from UI
  - Store preference in session state
  - File: `tools/consensus.py`

### Medium Priority

- [ ] **Worker Performance Trends**
  - Track scores/ranks over multiple runs
  - Show improvement or degradation
  - Requires: Session-level storage

- [ ] **Disagreement Categorization**
  - Auto-categorize disagreements (format, content, interpretation)
  - Add category filters to hotspots table
  - File: `core/comparison_analytics.py`

- [ ] **Confidence Thresholds**
  - Flag rows below confidence threshold
  - Allow re-processing low-confidence rows
  - File: `tools/consensus.py`

- [ ] **Worker Notes Display**
  - Show judge's notes for each worker in expandable UI
  - Currently stored but not displayed
  - File: `tools/consensus.py`

### Low Priority

- [ ] **Custom Scoring Rubric**
  - Allow users to define custom scoring criteria
  - Inject into judge prompt
  - File: `core/prompt_registry.py`

- [ ] **Agreement vs Quality Correlation**
  - Analyze if high agreement correlates with high quality
  - Add correlation chart
  - File: `core/comparison_analytics.py`

- [ ] **Model Cost Tracking**
  - Track token usage per worker/judge
  - Calculate cost-per-quality metrics
  - Requires: Token counting integration

- [ ] **Batch Comparison**
  - Compare quality metrics across different runs
  - A/B testing for different worker models
  - Requires: Run history storage

### Technical Debt

- [ ] **Refactor analytics rendering**
  - Consolidate `render_quality_metrics` and `_render_quality_analytics_from_df`
  - Remove duplication
  - File: `tools/consensus.py`, `core/comparison_analytics.py`

- [x] **Add integration tests** (Completed v1.1.0)
  - ~~Test full processing pipeline with mock LLM~~
  - Test UI rendering with Streamlit testing
  - File: `tests/unit/test_consensus.py`

- [ ] **Type hints completion**
  - Add full type hints to all new functions
  - Add mypy configuration
  - Files: All modified files

---

## API Reference

### ConsensusConfig

```python
@dataclass
class ConsensusConfig:
    """Configuration for a consensus processing run"""

    worker_configs: List[Dict[str, Any]]
        # List of worker configurations, each containing:
        # - provider_enum: LLMProvider
        # - api_key: str
        # - base_url: Optional[str]
        # - model: str

    judge_config: Dict[str, Any]
        # Judge configuration (same structure as worker)

    max_concurrency: int
        # Maximum parallel requests

    auto_retry: bool
        # Whether to retry failed requests

    max_retries: int
        # Maximum retry attempts

    save_path: Optional[str]
        # Path for saving partial results

    realtime_progress: bool
        # Whether to show real-time progress

    include_reasoning: bool
        # Whether to include judge reasoning in output

    enable_quality_scoring: bool = False
        # Whether to enable quality scoring (1-5) and ranking

    enable_disagreement_analysis: bool = False
        # Whether to enable disagreement analysis
```

### QualityMetrics

```python
@dataclass
class QualityMetrics:
    """Quality scoring metrics from enhanced judge evaluation"""

    average_scores: Dict[str, float]
        # Worker name -> average score (1-5)

    rank_distribution: Dict[str, Dict[int, int]]
        # Worker name -> {rank: count}

    confidence_stats: Dict[str, float]
        # Keys: mean, median, min, max, std

    disagreement_patterns: List[Dict]
        # List of {aspect, count, percentage, examples}
```

### Analytics Functions

```python
def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average quality score per worker."""

def calculate_rank_distribution(results: List[Dict]) -> Dict[str, Dict[int, int]]:
    """Calculate how often each worker ranked 1st, 2nd, 3rd."""

def calculate_confidence_stats(results: List[Dict]) -> Dict[str, float]:
    """Calculate confidence score statistics."""

def extract_disagreement_patterns(results: List[Dict]) -> List[Dict]:
    """Extract common disagreement patterns from results."""

def calculate_quality_metrics(results: List[Dict]) -> QualityMetrics:
    """Calculate all quality metrics from enhanced judge results."""

def render_quality_metrics(metrics: QualityMetrics, num_workers: int) -> None:
    """Render quality metrics visualizations in Streamlit."""
```

---

## Changelog

### v1.1.0 (2026-02-09)
- Expanded test suite from 19 to 124 comprehensive tests
- Added JSON parsing edge case tests (markdown, unicode, escapes)
- Added backwards compatibility tests for old response formats
- Added error scenario tests (rate limit, auth, timeout, etc.)
- Added property-based tests using Hypothesis
- Added performance/stress tests (1000+ rows, 10 workers)
- Added integration tests for ConsensusProcessor
- Added consensus-specific fixtures to conftest.py
- Updated requirements.txt with testing dependencies

### v1.0.0 (2024-02-08)
- Initial implementation of enhanced judge features
- Added quality scoring (1-5 stars)
- Added worker rankings
- Added confidence scores (0-100%)
- Added disagreement analysis
- Added analytics visualizations:
  - Model Quality Leaderboard
  - Rank Distribution
  - Confidence Statistics
  - Disagreement Hotspots
- Added 19 unit tests
- Full backwards compatibility with existing consensus runs

---

## Contributing

When contributing to the enhanced judge features:

1. **Adding new analytics**: Add functions to `core/comparison_analytics.py`
2. **Modifying judge prompt**: Update `core/prompt_registry.py`
3. **Changing UI**: Modify `tools/consensus.py`
4. **Processing logic**: Update `core/processing.py`
5. **Always add tests**: Add to `tests/unit/test_consensus.py`

### Code Style

- Use type hints for all function parameters and returns
- Add docstrings with Args and Returns sections
- Follow existing patterns for Streamlit components
- Use plotly for visualizations with ImportError fallbacks
