"""Answer quality metrics for QA evaluation.

Implements:
- Exact Match (EM)
- F1 Score (token-level)
- BLEU Score
- ROUGE-L Score
- Retrieval Metrics (Recall@k, MRR, NDCG@k)
- Comprehensive Evaluation with component breakdown
"""

import re
import string
import numpy as np
from typing import Tuple, Dict
from collections import Counter


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    - Convert to lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Collapse whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def exact_match(predicted: str, gold: str) -> float:
    """Exact Match (EM) score.

    Returns 1.0 if normalized answers match exactly, 0.0 otherwise.

    Args:
        predicted: Model-generated answer
        gold: Ground truth answer

    Returns:
        EM score (0.0 or 1.0)
    """
    return float(normalize_answer(predicted) == normalize_answer(gold))


def f1_score(predicted: str, gold: str) -> float:
    """Token-level F1 score.

    Computes F1 based on token overlap between normalized answers.

    Args:
        predicted: Model-generated answer
        gold: Ground truth answer

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def bleu_score(predicted: str, gold: str, max_n: int = 4) -> float:
    """BLEU score for answer evaluation.

    Computes smoothed BLEU score adapted for short answer evaluation.
    Uses n-gram precision with brevity penalty.

    Args:
        predicted: Model-generated answer
        gold: Ground truth answer
        max_n: Maximum n-gram order (default 4)

    Returns:
        BLEU score between 0.0 and 1.0
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(len(pred_tokens) == len(gold_tokens))

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)
        )
        gold_ngrams = Counter(
            tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens) - n + 1)
        )

        if not pred_ngrams or not gold_ngrams:
            precisions.append(0.0)
            continue

        matches = sum((pred_ngrams & gold_ngrams).values())
        precision = matches / sum(pred_ngrams.values())
        precisions.append(precision)

    # Geometric mean with smoothing
    if any(p == 0 for p in precisions):
        return 0.0

    geo_mean = 1.0
    for p in precisions:
        geo_mean *= p
    geo_mean = geo_mean ** (1.0 / len(precisions))

    # Brevity penalty
    ratio = len(pred_tokens) / max(len(gold_tokens), 1)
    if ratio > 1.0:
        brevity = 1.0
    else:
        brevity = max(0.0, 1.0 - 1.0 / ratio) if ratio > 0 else 0.0

    bleu = geo_mean * (1.0 - brevity)
    return bleu


def rouge_l(predicted: str, gold: str) -> float:
    """ROUGE-L score (Longest Common Subsequence).

    Computes F-score based on longest common subsequence between
    normalized predicted and gold answers.

    Args:
        predicted: Model-generated answer
        gold: Ground truth answer

    Returns:
        ROUGE-L F-score between 0.0 and 1.0
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(len(pred_tokens) == len(gold_tokens))

    # Compute LCS using dynamic programming
    m, n = len(pred_tokens), len(gold_tokens)
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                lcs_table[i][j] = lcs_table[i-1][j-1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])

    lcs_len = lcs_table[m][n]

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    rouge_l = (2 * precision * recall) / (precision + recall)

    return rouge_l


def compute_answer_metrics(predicted: str, gold: str) -> Dict[str, float]:
    """Compute all answer quality metrics.

    Args:
        predicted: Model-generated answer
        gold: Ground truth answer

    Returns:
        Dictionary containing:
            - em: Exact Match (0.0 or 1.0)
            - f1: Token-level F1 score
            - bleu: BLEU score
            - rouge_l: ROUGE-L score
    """
    return {
        "em": exact_match(predicted, gold),
        "f1": f1_score(predicted, gold),
        "bleu": bleu_score(predicted, gold),
        "rouge_l": rouge_l(predicted, gold),
    }


def micro_average_metrics(results: list[Dict[str, float]]) -> Dict[str, float]:
    """Compute micro-average of metrics across multiple questions.

    Args:
        results: List of metric dictionaries from compute_answer_metrics()

    Returns:
        Dictionary with averaged metrics
    """
    if not results:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0}

    metrics = {
        "em": sum(r["em"] for r in results) / len(results),
        "f1": sum(r["f1"] for r in results) / len(results),
        "bleu": sum(r["bleu"] for r in results) / len(results),
        "rouge_l": sum(r["rouge_l"] for r in results) / len(results),
    }

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as readable string.

    Args:
        metrics: Dictionary from compute_answer_metrics() or micro_average_metrics()

    Returns:
        Formatted string representation
    """
    return (
        f"EM: {metrics['em']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"BLEU: {metrics['bleu']:.4f} | "
        f"ROUGE-L: {metrics['rouge_l']:.4f}"
    )


# ── Retrieval-Level Metrics ─────────────────────────────────────────────────

def recall_at_k(retrieved_chunk_ids: list, gold_chunk_id: int, k: int = 5) -> float:
    """Recall@k: Is gold chunk in top-k retrieved?

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in rank order
        gold_chunk_id: Ground truth chunk ID containing answer
        k: Top-k threshold (default 5)

    Returns:
        1.0 if gold in top-k, 0.0 otherwise
    """
    if gold_chunk_id is None:
        return 0.0
    return float(gold_chunk_id in retrieved_chunk_ids[:k])


def mean_reciprocal_rank(retrieved_chunk_ids: list, gold_chunk_id: int) -> float:
    """Mean Reciprocal Rank (MRR): Position-weighted ranking metric.

    Returns 1/rank for position of gold chunk, or 0 if not retrieved.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in rank order
        gold_chunk_id: Ground truth chunk ID

    Returns:
        1/rank if found, 0.0 if not in retrieved list
    """
    if gold_chunk_id is None:
        return 0.0
    if gold_chunk_id in retrieved_chunk_ids:
        rank = retrieved_chunk_ids.index(gold_chunk_id) + 1
        return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_chunk_ids: list, gold_chunk_id: int, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain@k.

    Rewards retrieving gold chunk at higher positions more heavily.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in rank order
        gold_chunk_id: Ground truth chunk ID
        k: Top-k threshold (default 5)

    Returns:
        NDCG@k score between 0.0 and 1.0
    """
    if gold_chunk_id is None:
        return 0.0

    # DCG: discount log (rank)
    if gold_chunk_id in retrieved_chunk_ids[:k]:
        rank = retrieved_chunk_ids[:k].index(gold_chunk_id) + 1
        dcg = 1.0 / (1.0 + np.log2(rank))
    else:
        dcg = 0.0

    # IDCG: ideal DCG (gold at position 1)
    idcg = 1.0

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_chunk_ids: list, gold_chunk_id: int, ks: list = None
) -> Dict[str, float]:
    """Compute all retrieval-level metrics.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in rank order
        gold_chunk_id: Ground truth chunk ID
        ks: List of k values for recall/NDCG (default [5, 10])

    Returns:
        Dictionary with retrieval metrics
    """
    if ks is None:
        ks = [5, 10]

    metrics = {
        "mrr": mean_reciprocal_rank(retrieved_chunk_ids, gold_chunk_id),
    }

    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_chunk_ids, gold_chunk_id, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_chunk_ids, gold_chunk_id, k)

    return metrics


# ── Comprehensive Evaluation ────────────────────────────────────────────────

def comprehensive_evaluation(
    qa_results: list, gold_chunks: Dict[str, int] = None
) -> Dict:
    """Comprehensive evaluation across all metrics and components.

    Args:
        qa_results: List of QA result dicts, each with:
            - question: str
            - predicted_answer: str
            - gold_answer: str
            - confidence: float
            - retrieval_rank: int (optional, position of answer chunk)
            - retrieved_chunk_ids: list (optional, for retrieval metrics)
            - gold_chunk_id: int (optional, for retrieval evaluation)
            - metadata: dict with "qtype", "graph_nodes", etc.

        gold_chunks: Optional dict mapping question to gold chunk ID
            {question_text: chunk_id}

    Returns:
        Comprehensive evaluation report with answer, retrieval, and component metrics
    """
    if not qa_results:
        return {"error": "No results to evaluate"}

    # ── Answer-level metrics ────────────────────────────────────────────────
    answer_metrics_list = []
    retrieval_metrics_list = []
    failure_types = {}
    confidences = []
    retrieval_positions = []

    for result in qa_results:
        # Compute answer metrics
        metrics = compute_answer_metrics(
            result["predicted_answer"], result["gold_answer"]
        )
        answer_metrics_list.append(metrics)

        # Retrieval metrics (if available)
        if "retrieved_chunk_ids" in result and "gold_chunk_id" in result:
            ret_metrics = compute_retrieval_metrics(
                result["retrieved_chunk_ids"], result["gold_chunk_id"]
            )
            retrieval_metrics_list.append(ret_metrics)

        # Failure classification
        if metrics["em"] == 0.0:
            retrieval_rank = result.get("retrieval_rank")
            if retrieval_rank is None or retrieval_rank > 5:
                ftype = "retrieval"
            elif metrics["f1"] < 0.3:
                ftype = "extraction"
            else:
                ftype = "reasoning"
        else:
            ftype = "none"

        failure_types[ftype] = failure_types.get(ftype, 0) + 1

        # Confidence stats
        confidences.append(result.get("confidence", 0.5))
        if "retrieval_rank" in result and result["retrieval_rank"] is not None:
            retrieval_positions.append(result["retrieval_rank"])

    # Aggregate answer metrics
    avg_answer_metrics = micro_average_metrics(answer_metrics_list)

    # Aggregate retrieval metrics
    avg_retrieval_metrics = {}
    if retrieval_metrics_list:
        for key in retrieval_metrics_list[0].keys():
            avg_retrieval_metrics[key] = (
                sum(m[key] for m in retrieval_metrics_list) / len(retrieval_metrics_list)
            )

    # Per-question-type analysis
    question_types = {}
    for result in qa_results:
        qtype = result.get("metadata", {}).get("qtype", "unknown")
        if qtype not in question_types:
            question_types[qtype] = {
                "count": 0,
                "em": 0.0,
                "f1": 0.0,
                "bleu": 0.0,
                "rouge_l": 0.0,
                "avg_confidence": 0.0,
            }
        idx = qa_results.index(result)
        if idx < len(answer_metrics_list):
            metrics = answer_metrics_list[idx]
            question_types[qtype]["count"] += 1
            question_types[qtype]["em"] += metrics["em"]
            question_types[qtype]["f1"] += metrics["f1"]
            question_types[qtype]["bleu"] += metrics["bleu"]
            question_types[qtype]["rouge_l"] += metrics["rouge_l"]
            question_types[qtype]["avg_confidence"] += result.get("confidence", 0.5)

    # Normalize per-type metrics
    for qtype in question_types:
        count = question_types[qtype]["count"]
        if count > 0:
            question_types[qtype]["em"] /= count
            question_types[qtype]["f1"] /= count
            question_types[qtype]["bleu"] /= count
            question_types[qtype]["rouge_l"] /= count
            question_types[qtype]["avg_confidence"] /= count

    # Component quality analysis
    graph_usage = sum(
        1 for r in qa_results if r.get("metadata", {}).get("graph_nodes", 0) > 0
    )
    graph_success = sum(
        1
        for i, r in enumerate(qa_results)
        if r.get("metadata", {}).get("graph_nodes", 0) > 0
        and (i < len(answer_metrics_list) and answer_metrics_list[i]["em"] > 0)
    )

    component_quality = {
        "retrieval_quality": (
            avg_retrieval_metrics if avg_retrieval_metrics else None
        ),
        "extraction_quality": {
            "avg_f1": sum(m["f1"] for m in answer_metrics_list) / len(answer_metrics_list) if answer_metrics_list else 0,
            "exact_matches": sum(m["em"] for m in answer_metrics_list),
            "partial_matches": sum(1 for m in answer_metrics_list if 0 < m["f1"] < 1),
        },
        "reasoning_quality": {
            "graph_usage_rate": graph_usage / len(qa_results) if qa_results else 0,
            "graph_success_rate": graph_success / graph_usage if graph_usage > 0 else 0,
        },
    }

    # Confidence calibration analysis
    calibration = analyze_confidence_calibration(qa_results, answer_metrics_list)

    return {
        "total_questions": len(qa_results),
        "answer_metrics": avg_answer_metrics,
        "formatted_metrics": format_metrics(avg_answer_metrics),
        "retrieval_metrics": avg_retrieval_metrics,
        "failure_analysis": failure_types,
        "confidence_stats": {
            "mean": sum(confidences) / len(confidences) if confidences else 0.0,
            "min": min(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0,
            "std": (
                (
                    sum((x - (sum(confidences) / len(confidences))) ** 2 for x in confidences)
                    / len(confidences)
                )
                ** 0.5
                if confidences
                else 0.0
            ),
        },
        "retrieval_stats": {
            "avg_rank": sum(retrieval_positions) / len(retrieval_positions)
            if retrieval_positions
            else None,
        },
        "per_question_type": question_types,
        "component_quality": component_quality,
        "confidence_calibration": calibration,
    }


def analyze_confidence_calibration(qa_results: list, answer_metrics_list: list) -> Dict:
    """Analyze whether model confidence matches actual accuracy.

    Args:
        qa_results: List of QA results
        answer_metrics_list: List of metric dicts from compute_answer_metrics()

    Returns:
        Calibration analysis with confidence bins
    """
    if not qa_results or not answer_metrics_list:
        return {}

    # Pair confidence with accuracy (EM)
    paired = [
        (qa_results[i].get("confidence", 0.5), answer_metrics_list[i]["em"])
        for i in range(len(qa_results))
        if i < len(answer_metrics_list)
    ]

    # Bin by confidence
    bins = {
        "0.0-0.2": [],
        "0.2-0.4": [],
        "0.4-0.6": [],
        "0.6-0.8": [],
        "0.8-1.0": [],
    }

    bin_ranges = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.01),
    ]

    for conf, accuracy in paired:
        for i, (lo, hi) in enumerate(bin_ranges):
            if lo <= conf < hi:
                bin_key = list(bins.keys())[i]
                bins[bin_key].append(accuracy)
                break

    # Compute stats per bin
    calibration = {}
    for bin_key, accuracies in bins.items():
        if accuracies:
            avg_conf = float(bin_key.split("-")[0])  # Use bin center as proxy
            actual_accuracy = sum(accuracies) / len(accuracies)
            calibration[bin_key] = {
                "confidence": avg_conf,
                "actual_accuracy": actual_accuracy,
                "count": len(accuracies),
                "calibrated": abs(actual_accuracy - avg_conf) < 0.1,  # Within 10%
            }

    return calibration
