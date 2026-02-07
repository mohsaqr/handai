"""
AI Coder Analytics
Inter-rater reliability calculations for AI vs Human coding comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class AgreementMetrics:
    """Container for agreement metrics between AI and human coders"""
    exact_agreement: float  # Percentage of rows with exact code match
    cohens_kappa: float  # Agreement corrected for chance
    jaccard_index: float  # Set-based similarity (intersection/union)
    per_code_precision: Dict[str, float]  # TP / (TP + FP) for each code
    per_code_recall: Dict[str, float]  # TP / (TP + FN) for each code
    per_code_f1: Dict[str, float]  # Harmonic mean of precision and recall
    confusion_matrix: pd.DataFrame  # AI code vs Human code counts
    total_rows: int
    coded_rows: int
    ai_suggested_rows: int


def calculate_exact_agreement(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    total_rows: int
) -> float:
    """
    Calculate percentage of rows where codes match exactly.

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        total_rows: Total number of rows in dataset

    Returns:
        Percentage of exact matches (0-100)
    """
    if total_rows == 0:
        return 0.0

    matches = 0
    for row_idx in range(total_rows):
        human = set(human_codes.get(row_idx, []))
        ai_data = ai_suggestions.get(row_idx, {})
        ai = set(ai_data.get("codes", []) if ai_data else [])

        if human == ai:
            matches += 1

    return (matches / total_rows) * 100


def calculate_cohens_kappa(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    all_codes: List[str],
    total_rows: int
) -> float:
    """
    Calculate Cohen's Kappa for multi-label classification.

    Uses a binary approach per code and averages.

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        all_codes: List of all possible codes
        total_rows: Total number of rows

    Returns:
        Cohen's Kappa coefficient (-1 to 1)
    """
    if total_rows == 0 or not all_codes:
        return 0.0

    kappa_scores = []

    for code in all_codes:
        # Build binary arrays for this code
        human_binary = []
        ai_binary = []

        for row_idx in range(total_rows):
            human_has = code in human_codes.get(row_idx, [])
            ai_data = ai_suggestions.get(row_idx, {})
            ai_has = code in ai_data.get("codes", []) if ai_data else False

            human_binary.append(1 if human_has else 0)
            ai_binary.append(1 if ai_has else 0)

        # Calculate observed agreement
        agreement = sum(1 for h, a in zip(human_binary, ai_binary) if h == a)
        p_o = agreement / total_rows

        # Calculate expected agreement by chance
        human_pos = sum(human_binary) / total_rows
        ai_pos = sum(ai_binary) / total_rows
        p_e = (human_pos * ai_pos) + ((1 - human_pos) * (1 - ai_pos))

        # Calculate kappa for this code
        if p_e == 1.0:
            kappa = 1.0 if p_o == 1.0 else 0.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        kappa_scores.append(kappa)

    return np.mean(kappa_scores) if kappa_scores else 0.0


def calculate_jaccard_index(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    total_rows: int
) -> float:
    """
    Calculate average Jaccard similarity index across all rows.

    Jaccard = |intersection| / |union|

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        total_rows: Total number of rows

    Returns:
        Average Jaccard index (0-1)
    """
    if total_rows == 0:
        return 0.0

    jaccard_scores = []

    for row_idx in range(total_rows):
        human = set(human_codes.get(row_idx, []))
        ai_data = ai_suggestions.get(row_idx, {})
        ai = set(ai_data.get("codes", []) if ai_data else [])

        if len(human) == 0 and len(ai) == 0:
            jaccard_scores.append(1.0)  # Both empty = perfect agreement
        elif len(human) == 0 or len(ai) == 0:
            jaccard_scores.append(0.0)  # One empty = no agreement
        else:
            intersection = len(human & ai)
            union = len(human | ai)
            jaccard_scores.append(intersection / union)

    return np.mean(jaccard_scores)


def calculate_per_code_metrics(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    all_codes: List[str],
    total_rows: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calculate precision, recall, and F1 for each code.

    Treating human codes as ground truth:
    - Precision: How many AI suggestions were correct?
    - Recall: How many human codes did AI find?
    - F1: Harmonic mean of precision and recall

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        all_codes: List of all possible codes
        total_rows: Total number of rows

    Returns:
        Tuple of (precision_dict, recall_dict, f1_dict)
    """
    precision = {}
    recall = {}
    f1 = {}

    for code in all_codes:
        tp = 0  # True positives (AI suggested and human coded)
        fp = 0  # False positives (AI suggested but human didn't code)
        fn = 0  # False negatives (human coded but AI didn't suggest)

        for row_idx in range(total_rows):
            human_has = code in human_codes.get(row_idx, [])
            ai_data = ai_suggestions.get(row_idx, {})
            ai_has = code in ai_data.get("codes", []) if ai_data else False

            if human_has and ai_has:
                tp += 1
            elif ai_has and not human_has:
                fp += 1
            elif human_has and not ai_has:
                fn += 1

        # Calculate metrics
        precision[code] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[code] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[code] + recall[code] > 0:
            f1[code] = 2 * (precision[code] * recall[code]) / (precision[code] + recall[code])
        else:
            f1[code] = 0.0

    return precision, recall, f1


def build_confusion_matrix(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    all_codes: List[str],
    total_rows: int
) -> pd.DataFrame:
    """
    Build a confusion matrix comparing AI suggestions vs human codes.

    Rows = AI suggested codes
    Columns = Human applied codes

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        all_codes: List of all possible codes
        total_rows: Total number of rows

    Returns:
        DataFrame with confusion matrix
    """
    # Initialize matrix
    matrix = {code: {c: 0 for c in all_codes} for code in all_codes}

    for row_idx in range(total_rows):
        human = set(human_codes.get(row_idx, []))
        ai_data = ai_suggestions.get(row_idx, {})
        ai = set(ai_data.get("codes", []) if ai_data else [])

        # Count co-occurrences
        for ai_code in ai:
            if ai_code in matrix:
                for human_code in human:
                    if human_code in matrix[ai_code]:
                        matrix[ai_code][human_code] += 1

    # Convert to DataFrame
    df = pd.DataFrame(matrix).T  # Rows = AI, Columns = Human
    df.index.name = "AI Suggested"
    df.columns.name = "Human Applied"

    return df


def calculate_all_metrics(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    all_codes: List[str],
    total_rows: int
) -> AgreementMetrics:
    """
    Calculate all inter-rater reliability metrics.

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        all_codes: List of all possible codes
        total_rows: Total number of rows

    Returns:
        AgreementMetrics dataclass with all calculated metrics
    """
    # Count coded and suggested rows
    coded_rows = sum(1 for codes in human_codes.values() if codes)
    ai_suggested_rows = sum(1 for row_idx in range(total_rows)
                           if ai_suggestions.get(row_idx, {}).get("codes"))

    # Calculate all metrics
    exact_agreement = calculate_exact_agreement(human_codes, ai_suggestions, total_rows)
    cohens_kappa = calculate_cohens_kappa(human_codes, ai_suggestions, all_codes, total_rows)
    jaccard_index = calculate_jaccard_index(human_codes, ai_suggestions, total_rows)
    precision, recall, f1 = calculate_per_code_metrics(human_codes, ai_suggestions, all_codes, total_rows)
    confusion_matrix = build_confusion_matrix(human_codes, ai_suggestions, all_codes, total_rows)

    return AgreementMetrics(
        exact_agreement=exact_agreement,
        cohens_kappa=cohens_kappa,
        jaccard_index=jaccard_index,
        per_code_precision=precision,
        per_code_recall=recall,
        per_code_f1=f1,
        confusion_matrix=confusion_matrix,
        total_rows=total_rows,
        coded_rows=coded_rows,
        ai_suggested_rows=ai_suggested_rows
    )


def get_disagreement_analysis(
    human_codes: Dict[int, List[str]],
    ai_suggestions: Dict[int, dict],
    total_rows: int
) -> List[Dict]:
    """
    Get detailed analysis of disagreements between AI and human.

    Args:
        human_codes: {row_idx: [code1, code2, ...]}
        ai_suggestions: {row_idx: {"codes": [code1, code2], "confidence": {...}}}
        total_rows: Total number of rows

    Returns:
        List of disagreement records with row_idx, human_codes, ai_codes, type
    """
    disagreements = []

    for row_idx in range(total_rows):
        human = set(human_codes.get(row_idx, []))
        ai_data = ai_suggestions.get(row_idx, {})
        ai = set(ai_data.get("codes", []) if ai_data else [])

        if human != ai:
            # Determine type of disagreement
            only_human = human - ai
            only_ai = ai - human

            if only_human and only_ai:
                dis_type = "both_different"
            elif only_human:
                dis_type = "ai_missed"
            elif only_ai:
                dis_type = "ai_extra"
            else:
                dis_type = "unknown"

            disagreements.append({
                "row_idx": row_idx,
                "human_codes": list(human),
                "ai_codes": list(ai),
                "ai_confidence": ai_data.get("confidence", {}),
                "ai_reasoning": ai_data.get("reasoning", ""),
                "only_in_human": list(only_human),
                "only_in_ai": list(only_ai),
                "type": dis_type
            })

    return disagreements


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Cohen's Kappa value using Landis & Koch guidelines.

    Args:
        kappa: Cohen's Kappa coefficient

    Returns:
        Interpretation string
    """
    if kappa < 0:
        return "Poor (less than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"
