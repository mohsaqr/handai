"""
Model Comparison Analytics
Agreement metrics and analysis for multi-model comparison results
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ModelAgreementMetrics:
    """Complete agreement metrics for model comparison"""
    pairwise_agreement: pd.DataFrame  # Matrix of exact match percentages
    jaccard_similarity: pd.DataFrame  # Matrix of token-level Jaccard similarity
    average_pairwise_agreement: float  # Overall agreement score
    per_model_consistency: Dict[str, float]  # How often each model agrees with majority


def calculate_pairwise_agreement(outputs: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Calculate pairwise exact match agreement between models.

    Args:
        outputs: Dict mapping model name to list of outputs (one per row)

    Returns:
        DataFrame with pairwise agreement percentages
    """
    model_names = list(outputs.keys())
    n_models = len(model_names)

    if n_models < 2:
        return pd.DataFrame()

    n_rows = len(outputs[model_names[0]])

    # Build agreement matrix
    matrix = np.zeros((n_models, n_models))

    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i == j:
                matrix[i][j] = 100.0
            else:
                # Calculate exact match percentage
                matches = sum(
                    1 for k in range(n_rows)
                    if _normalize_output(outputs[model_i][k]) == _normalize_output(outputs[model_j][k])
                )
                matrix[i][j] = (matches / n_rows * 100) if n_rows > 0 else 0.0

    return pd.DataFrame(matrix, index=model_names, columns=model_names)


def calculate_jaccard_similarity(outputs: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Calculate pairwise Jaccard similarity (token-level) between models.

    Args:
        outputs: Dict mapping model name to list of outputs (one per row)

    Returns:
        DataFrame with pairwise Jaccard similarity percentages
    """
    model_names = list(outputs.keys())
    n_models = len(model_names)

    if n_models < 2:
        return pd.DataFrame()

    n_rows = len(outputs[model_names[0]])

    # Build similarity matrix
    matrix = np.zeros((n_models, n_models))

    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i == j:
                matrix[i][j] = 100.0
            else:
                # Calculate average Jaccard similarity across all rows
                similarities = []
                for k in range(n_rows):
                    tokens_i = _tokenize(outputs[model_i][k])
                    tokens_j = _tokenize(outputs[model_j][k])

                    if len(tokens_i) == 0 and len(tokens_j) == 0:
                        similarities.append(1.0)
                    elif len(tokens_i) == 0 or len(tokens_j) == 0:
                        similarities.append(0.0)
                    else:
                        intersection = len(tokens_i & tokens_j)
                        union = len(tokens_i | tokens_j)
                        similarities.append(intersection / union if union > 0 else 0.0)

                matrix[i][j] = (np.mean(similarities) * 100) if similarities else 0.0

    return pd.DataFrame(matrix, index=model_names, columns=model_names)


def calculate_per_model_consistency(outputs: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate how often each model agrees with the majority answer.

    Args:
        outputs: Dict mapping model name to list of outputs (one per row)

    Returns:
        Dict mapping model name to consistency percentage
    """
    model_names = list(outputs.keys())
    n_models = len(model_names)

    if n_models < 2:
        return {name: 100.0 for name in model_names}

    n_rows = len(outputs[model_names[0]])
    consistency = {name: 0 for name in model_names}

    for row_idx in range(n_rows):
        # Get all outputs for this row
        row_outputs = [_normalize_output(outputs[name][row_idx]) for name in model_names]

        # Find the majority answer
        from collections import Counter
        counts = Counter(row_outputs)
        majority_output, majority_count = counts.most_common(1)[0]

        # Only count as majority if more than half agree
        if majority_count > n_models / 2:
            for name in model_names:
                if _normalize_output(outputs[name][row_idx]) == majority_output:
                    consistency[name] += 1

    # Convert to percentages
    return {
        name: (count / n_rows * 100) if n_rows > 0 else 0.0
        for name, count in consistency.items()
    }


def calculate_all_agreement_metrics(outputs: Dict[str, List[str]]) -> ModelAgreementMetrics:
    """
    Calculate all agreement metrics for model comparison.

    Args:
        outputs: Dict mapping model name to list of outputs (one per row)

    Returns:
        ModelAgreementMetrics with all calculated metrics
    """
    pairwise = calculate_pairwise_agreement(outputs)
    jaccard = calculate_jaccard_similarity(outputs)
    consistency = calculate_per_model_consistency(outputs)

    # Calculate average pairwise agreement (excluding diagonal)
    if pairwise.empty:
        avg_pairwise = 0.0
    else:
        n = len(pairwise)
        # Get upper triangle values (excluding diagonal)
        upper_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_vals.append(pairwise.iloc[i, j])
        avg_pairwise = np.mean(upper_vals) if upper_vals else 0.0

    return ModelAgreementMetrics(
        pairwise_agreement=pairwise,
        jaccard_similarity=jaccard,
        average_pairwise_agreement=avg_pairwise,
        per_model_consistency=consistency
    )


def _normalize_output(text: Optional[str]) -> str:
    """Normalize output for comparison"""
    if text is None:
        return ""
    # Lowercase, strip whitespace, normalize spaces
    normalized = str(text).lower().strip()
    normalized = " ".join(normalized.split())
    return normalized


def _tokenize(text: Optional[str]) -> set:
    """Tokenize text into a set of tokens for Jaccard similarity"""
    if text is None:
        return set()

    # Simple tokenization: split on whitespace and punctuation
    import re
    text = str(text).lower()
    # Split on non-alphanumeric characters
    tokens = re.split(r'[^a-z0-9]+', text)
    # Filter empty tokens
    return {t for t in tokens if t}


def render_agreement_metrics(metrics: ModelAgreementMetrics) -> None:
    """
    Render agreement metrics in Streamlit.

    Args:
        metrics: Calculated agreement metrics
    """
    import streamlit as st

    st.subheader("Model Agreement Analysis")

    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Average Pairwise Agreement",
            f"{metrics.average_pairwise_agreement:.1f}%",
            help="Average exact match percentage between all model pairs"
        )

    with col2:
        # Show highest/lowest consistency models
        if metrics.per_model_consistency:
            best_model = max(metrics.per_model_consistency, key=metrics.per_model_consistency.get)
            st.metric(
                "Most Consistent Model",
                best_model,
                f"{metrics.per_model_consistency[best_model]:.1f}%",
                help="Model that most often agrees with the majority"
            )

    # Pairwise Agreement Matrix
    st.markdown("### Pairwise Exact Match Agreement")
    if not metrics.pairwise_agreement.empty:
        _render_heatmap(
            metrics.pairwise_agreement,
            "Exact match percentage between model pairs"
        )
    else:
        st.info("Not enough models for pairwise comparison")

    # Jaccard Similarity Matrix
    st.markdown("### Token-Level Similarity (Jaccard)")
    if not metrics.jaccard_similarity.empty:
        _render_heatmap(
            metrics.jaccard_similarity,
            "Average Jaccard similarity between model pairs"
        )
    else:
        st.info("Not enough models for similarity comparison")

    # Per-Model Consistency
    st.markdown("### Per-Model Majority Agreement")
    if metrics.per_model_consistency:
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Bar(
                x=list(metrics.per_model_consistency.keys()),
                y=list(metrics.per_model_consistency.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(metrics.per_model_consistency)],
                text=[f"{v:.1f}%" for v in metrics.per_model_consistency.values()],
                textposition='auto',
            )])
            fig.update_layout(
                title="How Often Each Model Agrees with Majority",
                xaxis_title="Model",
                yaxis_title="Agreement %",
                yaxis_range=[0, 100],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback without plotly
            for name, pct in metrics.per_model_consistency.items():
                st.metric(name, f"{pct:.1f}%")


def _render_heatmap(df: pd.DataFrame, title: str) -> None:
    """Render a heatmap for a matrix DataFrame"""
    import streamlit as st

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale='RdYlGn',
            zmin=0,
            zmax=100,
            text=[[f"{val:.1f}%" for val in row] for row in df.values],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Models: %{x} vs %{y}<br>Agreement: %{z:.1f}%<extra></extra>"
        ))

        fig.update_layout(
            title=title,
            height=300 + len(df) * 30,
            xaxis_title="",
            yaxis_title=""
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback: show as styled dataframe
        st.dataframe(
            df.style.format("{:.1f}%").background_gradient(cmap='RdYlGn', vmin=0, vmax=100),
            use_container_width=True
        )


# ==========================================
# Enhanced Judge Analytics
# ==========================================

@dataclass
class QualityMetrics:
    """Quality scoring metrics from enhanced judge evaluation"""
    average_scores: Dict[str, float]  # Worker name -> average score
    rank_distribution: Dict[str, Dict[int, int]]  # Worker -> {rank: count}
    confidence_stats: Dict[str, float]  # mean, median, min, max, std
    disagreement_patterns: List[Dict]  # Common disagreement aspects


def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate average quality score per worker.

    Args:
        results: List of result dicts containing worker_N_score fields

    Returns:
        Dict mapping worker name to average score (1-5)
    """
    worker_scores: Dict[str, List[float]] = {}

    for r in results:
        for key, value in r.items():
            if key.endswith("_score") and key.startswith("worker_"):
                worker_name = key.replace("_score", "")
                if worker_name not in worker_scores:
                    worker_scores[worker_name] = []
                try:
                    score = float(value)
                    if 1 <= score <= 5:
                        worker_scores[worker_name].append(score)
                except (ValueError, TypeError):
                    pass

    return {
        name: np.mean(scores) if scores else 0.0
        for name, scores in worker_scores.items()
    }


def calculate_rank_distribution(results: List[Dict]) -> Dict[str, Dict[int, int]]:
    """
    Calculate how often each worker ranked 1st, 2nd, 3rd, etc.

    Args:
        results: List of result dicts containing worker_N_rank fields

    Returns:
        Dict mapping worker name to {rank: count} distribution
    """
    rank_counts: Dict[str, Dict[int, int]] = {}

    for r in results:
        for key, value in r.items():
            if key.endswith("_rank") and key.startswith("worker_"):
                worker_name = key.replace("_rank", "")
                if worker_name not in rank_counts:
                    rank_counts[worker_name] = {}
                try:
                    rank = int(value)
                    if rank > 0:  # Valid rank
                        rank_counts[worker_name][rank] = rank_counts[worker_name].get(rank, 0) + 1
                except (ValueError, TypeError):
                    pass

    return rank_counts


def calculate_confidence_stats(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate confidence score statistics.

    Args:
        results: List of result dicts containing judge_confidence field

    Returns:
        Dict with mean, median, min, max, std confidence values
    """
    confidences = []
    for r in results:
        conf = r.get("judge_confidence")
        if conf is not None:
            try:
                conf_val = float(conf)
                if 0 <= conf_val <= 100:
                    confidences.append(conf_val)
            except (ValueError, TypeError):
                pass

    if not confidences:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}

    return {
        "mean": float(np.mean(confidences)),
        "median": float(np.median(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "std": float(np.std(confidences)),
    }


def extract_disagreement_patterns(results: List[Dict]) -> List[Dict]:
    """
    Extract common disagreement patterns from results.

    Args:
        results: List of result dicts containing disagreements_raw JSON

    Returns:
        List of {aspect, count, examples} dicts sorted by frequency
    """
    import json
    from collections import Counter

    aspect_counter: Counter = Counter()
    aspect_examples: Dict[str, List[str]] = {}

    for r in results:
        raw = r.get("disagreements_raw", "[]")
        try:
            disagreements = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(disagreements, list):
                continue

            for d in disagreements:
                aspect = d.get("aspect", "unknown")
                details = d.get("details", "")

                aspect_counter[aspect] += 1

                if aspect not in aspect_examples:
                    aspect_examples[aspect] = []
                if details and len(aspect_examples[aspect]) < 3:
                    aspect_examples[aspect].append(details)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build result sorted by frequency
    patterns = []
    total = len(results)
    for aspect, count in aspect_counter.most_common():
        patterns.append({
            "aspect": aspect,
            "count": count,
            "percentage": (count / total * 100) if total > 0 else 0,
            "examples": aspect_examples.get(aspect, [])
        })

    return patterns


def calculate_quality_metrics(results: List[Dict]) -> QualityMetrics:
    """
    Calculate all quality metrics from enhanced judge results.

    Args:
        results: List of result dicts from consensus processing

    Returns:
        QualityMetrics with all calculated values
    """
    return QualityMetrics(
        average_scores=calculate_average_scores(results),
        rank_distribution=calculate_rank_distribution(results),
        confidence_stats=calculate_confidence_stats(results),
        disagreement_patterns=extract_disagreement_patterns(results)
    )


def render_quality_metrics(metrics: QualityMetrics, num_workers: int) -> None:
    """
    Render quality metrics visualizations in Streamlit.

    Args:
        metrics: Calculated quality metrics
        num_workers: Number of workers in the consensus run
    """
    import streamlit as st

    # Model Quality Leaderboard
    if metrics.average_scores:
        st.markdown("### Model Quality Leaderboard")
        try:
            import plotly.graph_objects as go

            # Sort by score descending
            sorted_workers = sorted(
                metrics.average_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            worker_names = [w[0].replace("_", " ").title() for w in sorted_workers]
            scores = [w[1] for w in sorted_workers]

            fig = go.Figure(data=[go.Bar(
                x=worker_names,
                y=scores,
                marker_color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'][:len(scores)],
                text=[f"{s:.2f}" for s in scores],
                textposition='auto',
            )])
            fig.update_layout(
                xaxis_title="Worker",
                yaxis_title="Average Score (1-5)",
                yaxis_range=[0, 5.5],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for name, score in metrics.average_scores.items():
                st.metric(name.replace("_", " ").title(), f"{score:.2f} / 5")

    # Rank Distribution
    if metrics.rank_distribution:
        st.markdown("### Rank Distribution")
        try:
            import plotly.graph_objects as go

            workers = list(metrics.rank_distribution.keys())
            max_rank = num_workers

            fig = go.Figure()
            colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E']

            for rank in range(1, max_rank + 1):
                counts = [
                    metrics.rank_distribution.get(w, {}).get(rank, 0)
                    for w in workers
                ]
                rank_label = f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}"
                fig.add_trace(go.Bar(
                    name=rank_label,
                    x=[w.replace("_", " ").title() for w in workers],
                    y=counts,
                    marker_color=colors[rank - 1] if rank <= len(colors) else colors[-1],
                ))

            fig.update_layout(
                barmode='group',
                xaxis_title="Worker",
                yaxis_title="Count",
                height=300,
                legend_title="Rank"
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for worker, ranks in metrics.rank_distribution.items():
                st.write(f"**{worker.replace('_', ' ').title()}**: {ranks}")

    # Confidence Distribution
    if metrics.confidence_stats.get("mean", 0) > 0:
        st.markdown("### Judge Confidence Statistics")
        cols = st.columns(5)
        cols[0].metric("Mean", f"{metrics.confidence_stats['mean']:.1f}%")
        cols[1].metric("Median", f"{metrics.confidence_stats['median']:.1f}%")
        cols[2].metric("Min", f"{metrics.confidence_stats['min']:.1f}%")
        cols[3].metric("Max", f"{metrics.confidence_stats['max']:.1f}%")
        cols[4].metric("Std Dev", f"{metrics.confidence_stats['std']:.1f}")

    # Disagreement Hotspots
    if metrics.disagreement_patterns:
        st.markdown("### Disagreement Hotspots")
        hotspot_data = []
        for p in metrics.disagreement_patterns[:10]:  # Top 10
            hotspot_data.append({
                "Aspect": p["aspect"],
                "Occurrences": p["count"],
                "Frequency": f"{p['percentage']:.1f}%",
                "Example": p["examples"][0] if p["examples"] else "-"
            })
        if hotspot_data:
            st.dataframe(pd.DataFrame(hotspot_data), use_container_width=True)
