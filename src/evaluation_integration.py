"""Example integration of answer quality metrics into QA pipeline.

This module shows how to evaluate QA results using the metrics defined in evaluation.py
"""

from typing import Optional
from src.evaluation import (
    compute_answer_metrics, micro_average_metrics, format_metrics,
    comprehensive_evaluation
)


def evaluate_qa_result(
    question: str,
    predicted_answer: str,
    gold_answer: str,
    confidence: float,
    retrieval_rank: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Evaluate a single QA result with metrics and confidence.

    Args:
        question: Input question
        predicted_answer: Model-generated answer
        gold_answer: Ground truth answer
        confidence: Model confidence (combined score from graph + cross-encoder)
        retrieval_rank: Rank of answer chunk in retrieval (for ablation analysis)
        metadata: Additional metadata (reasoning type, graph size, etc.)

    Returns:
        Evaluation result with metrics and provenance information
    """
    metrics = compute_answer_metrics(predicted_answer, gold_answer)

    result = {
        "question": question,
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "confidence": confidence,
        "metrics": metrics,
        "em": metrics["em"],
        "f1": metrics["f1"],
        "bleu": metrics["bleu"],
        "rouge_l": metrics["rouge_l"],
        "retrieval_rank": retrieval_rank,
        "metadata": metadata or {},
    }

    # Determine failure type if not exact match
    if metrics["em"] == 0.0:
        if retrieval_rank is None or retrieval_rank > 5:
            result["failure_type"] = "retrieval"
        elif metrics["f1"] < 0.3:
            result["failure_type"] = "extraction"
        else:
            result["failure_type"] = "reasoning"
    else:
        result["failure_type"] = "none"

    return result


def batch_evaluate(qa_results: list[dict]) -> dict:
    """Evaluate a batch of QA results.

    Args:
        qa_results: List of dictionaries, each with keys:
            - question
            - predicted_answer
            - gold_answer
            - confidence
            - retrieval_rank (optional)
            - metadata (optional)

    Returns:
        Comprehensive evaluation report with:
            - Aggregate metrics (micro-average)
            - Per-question-type analysis
            - Failure analysis
            - Confidence statistics
    """
    if not qa_results:
        return {"error": "No results to evaluate"}

    # Evaluate each result
    evaluated = []
    for result in qa_results:
        evaluated.append(evaluate_qa_result(
            question=result["question"],
            predicted_answer=result["predicted_answer"],
            gold_answer=result["gold_answer"],
            confidence=result.get("confidence", 0.5),
            retrieval_rank=result.get("retrieval_rank"),
            metadata=result.get("metadata"),
        ))

    # Aggregate metrics
    metrics_list = [r["metrics"] for r in evaluated]
    avg_metrics = micro_average_metrics(metrics_list)

    # Count failures by type
    failure_types = {}
    for r in evaluated:
        ftype = r["failure_type"]
        failure_types[ftype] = failure_types.get(ftype, 0) + 1

    # Confidence statistics
    confidences = [r["confidence"] for r in evaluated]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Retrieval analysis (if available)
    retrieval_positions = [r["retrieval_rank"] for r in evaluated if r["retrieval_rank"] is not None]
    avg_retrieval_rank = sum(retrieval_positions) / len(retrieval_positions) if retrieval_positions else None

    # Question type distribution
    question_types = {}
    for r in evaluated:
        qtype = r["metadata"].get("qtype", "unknown")
        if qtype not in question_types:
            question_types[qtype] = {"count": 0, "em": 0, "f1": 0}
        question_types[qtype]["count"] += 1
        question_types[qtype]["em"] += r["em"]
        question_types[qtype]["f1"] += r["f1"]

    # Normalize per-question-type metrics
    for qtype in question_types:
        count = question_types[qtype]["count"]
        if count > 0:
            question_types[qtype]["em"] /= count
            question_types[qtype]["f1"] /= count

    return {
        "total_questions": len(evaluated),
        "aggregate_metrics": avg_metrics,
        "formatted_metrics": format_metrics(avg_metrics),
        "failure_analysis": failure_types,
        "confidence_stats": {
            "mean": avg_confidence,
            "min": min(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0,
        },
        "retrieval_stats": {
            "avg_rank": avg_retrieval_rank,
        } if avg_retrieval_rank else {},
        "per_question_type": question_types,
        "detailed_results": evaluated,
    }


def print_evaluation_report(report: dict) -> None:
    """Print formatted evaluation report.

    Args:
        report: Dictionary from batch_evaluate()
    """
    print("\n" + "="*70)
    print("QA EVALUATION REPORT")
    print("="*70)

    print(f"\nTotal Questions: {report['total_questions']}")
    print(f"\nAggregate Metrics:")
    print(f"  {report['formatted_metrics']}")

    print(f"\nFailure Analysis:")
    for ftype, count in report['failure_analysis'].items():
        pct = 100.0 * count / report['total_questions']
        print(f"  {ftype}: {count} ({pct:.1f}%)")

    print(f"\nConfidence Statistics:")
    conf_stats = report['confidence_stats']
    print(f"  Mean: {conf_stats['mean']:.4f}")
    print(f"  Min: {conf_stats['min']:.4f}")
    print(f"  Max: {conf_stats['max']:.4f}")

    if report['retrieval_stats']:
        ret_stats = report['retrieval_stats']
        if ret_stats.get('avg_rank'):
            print(f"\nRetrieval Statistics:")
            print(f"  Avg Answer Rank: {ret_stats['avg_rank']:.1f}")

    if report['per_question_type']:
        print(f"\nPer-Question-Type Analysis:")
        for qtype, stats in report['per_question_type'].items():
            print(f"  {qtype}: {stats['count']} questions")
            print(f"    EM: {stats['em']:.4f}, F1: {stats['f1']:.4f}")

    print("\n" + "="*70 + "\n")


def print_comprehensive_report(report: dict) -> None:
    """Print comprehensive evaluation report with retrieval and component metrics.

    Args:
        report: Dictionary from comprehensive_evaluation()
    """
    print("\n" + "="*80)
    print(" "*20 + "COMPREHENSIVE QA EVALUATION REPORT")
    print("="*80)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total Questions: {report['total_questions']}")

    # Answer metrics
    if report['answer_metrics']:
        print(f"\n📈 ANSWER QUALITY METRICS")
        metrics = report['answer_metrics']
        print(f"  {report['formatted_metrics']}")

    # Retrieval metrics
    if report['retrieval_metrics']:
        print(f"\n🔍 RETRIEVAL METRICS")
        ret = report['retrieval_metrics']
        for key, value in ret.items():
            print(f"  {key}: {value:.4f}")

    # Component quality
    if report['component_quality']:
        comp = report['component_quality']
        print(f"\n⚙️  COMPONENT QUALITY ANALYSIS")

        if comp.get('retrieval_quality'):
            print(f"  Retrieval Quality:")
            rq = comp['retrieval_quality']
            for key, val in rq.items():
                if 'recall' in key or 'ndcg' in key:
                    print(f"    {key}: {val:.4f}")

        if comp.get('extraction_quality'):
            print(f"  Extraction Quality:")
            eq = comp['extraction_quality']
            print(f"    Avg F1: {eq['avg_f1']:.4f}")
            print(f"    Exact Matches: {eq['exact_matches']}/{report['total_questions']}")
            print(f"    Partial Matches: {eq['partial_matches']}")

        if comp.get('reasoning_quality'):
            print(f"  Reasoning Quality:")
            gq = comp['reasoning_quality']
            print(f"    Graph Usage Rate: {gq['graph_usage_rate']:.2%}")
            if gq['graph_success_rate'] > 0:
                print(f"    Graph Success Rate: {gq['graph_success_rate']:.2%}")

    # Failure analysis
    print(f"\n❌ FAILURE ANALYSIS")
    for ftype, count in report['failure_analysis'].items():
        pct = 100.0 * count / report['total_questions']
        print(f"  {ftype}: {count} ({pct:.1f}%)")

    # Confidence stats
    if report['confidence_stats']:
        print(f"\n📍 CONFIDENCE STATISTICS")
        conf = report['confidence_stats']
        print(f"  Mean: {conf['mean']:.4f}")
        print(f"  Std Dev: {conf.get('std', 0):.4f}")
        print(f"  Range: [{conf['min']:.4f}, {conf['max']:.4f}]")

    # Confidence calibration
    if report.get('confidence_calibration'):
        print(f"\n🎯 CONFIDENCE CALIBRATION")
        calib = report['confidence_calibration']
        for bin_key, stats in calib.items():
            status = "✓ GOOD" if stats['calibrated'] else "✗ BAD"
            print(f"  {bin_key}: Conf={stats['confidence']:.2f}, "
                  f"Actual={stats['actual_accuracy']:.4f}, "
                  f"N={stats['count']} {status}")

    # Per-question-type breakdown
    if report['per_question_type']:
        print(f"\n📋 PER-QUESTION-TYPE ANALYSIS")
        for qtype, stats in report['per_question_type'].items():
            print(f"  {qtype.upper()}: {stats['count']} questions")
            print(f"    EM={stats['em']:.4f}, F1={stats['f1']:.4f}, "
                  f"BLEU={stats['bleu']:.4f}, ROUGE-L={stats['rouge_l']:.4f}")
            print(f"    Avg Confidence: {stats['avg_confidence']:.4f}")

    print("\n" + "="*80 + "\n")

