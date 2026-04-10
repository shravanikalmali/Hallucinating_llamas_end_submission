#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Narrative QA Pipeline

Evaluates the QA system across books from qaps.csv:
1. Loads questions and ground truth answers
2. Runs full retrieval + answering pipeline
3. Collects predictions with metadata
4. Computes multiple evaluation metrics (EM, F1, ROUGE-L, Recall@k)
5. Aggregates results by book and question type
6. Saves predictions and metrics to JSON files

Usage:
    python3 evaluate_metrics.py [--num-books 200] [--output-dir ./results] [--test-set-only]
"""

import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import retrieve_and_answer
from src.config import find_data_directory
from src.evaluation import (
    compute_answer_metrics, micro_average_metrics, format_metrics,
    comprehensive_evaluation
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def parse_qaps_csv(csv_file: Path, test_set_only: bool = True) -> Dict[str, List[Dict]]:
    """Parse qaps.csv and group questions by book_id.

    Args:
        csv_file: Path to qaps.csv
        test_set_only: If True, only load 'test' set questions

    Returns:
        Dict mapping book_id → list of question dicts
    """
    books_data = defaultdict(list)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if test_set_only and row.get('set') != 'test':
                continue

            book_id = row['document_id']
            books_data[book_id].append({
                'question': row['question'],
                'ground_truth': row['answer1'],  # Use answer1 as primary truth
                'answer2': row.get('answer2'),
            })

    return dict(books_data)


def get_unique_books(books_data: Dict, processed_dir: Path) -> List[str]:
    """Get unique book IDs from questions data that exist on disk."""
    all_books = sorted(list(books_data.keys()))
    existing_books = []
    for bid in all_books:
        if (processed_dir / bid).is_dir():
            existing_books.append(bid)
    return existing_books


def book_is_indexed(book_id: str, processed_dir: Path) -> bool:
    """Check if book has all required indexing files."""
    book_path = processed_dir / book_id
    required_files = [
        'chunks.jsonl',
        'embeddings.npy',
        f'{book_id}.faiss',
        'filtered_chunk_entities.json',
        'graph.graphml',
    ]

    for fname in required_files:
        if not (book_path / fname).exists():
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 2. QUESTION TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_question_type(question: str) -> str:
    """Detect question type from opening word.

    Returns: 'who', 'where', 'when', 'why', 'what', or 'other'
    """
    question_lower = question.lower().strip()

    # Remove common prefixes
    for prefix in ["why does", "why did", "why do", "why is", "why was"]:
        if question_lower.startswith(prefix):
            return "why"

    for prefix in ["who is", "who was", "who does", "who did", "who gets", "who becomes"]:
        if question_lower.startswith(prefix):
            return "who"

    for prefix in ["where does", "where did", "where is", "where was"]:
        if question_lower.startswith(prefix):
            return "where"

    for prefix in ["when does", "when did", "when is", "when was"]:
        if question_lower.startswith(prefix):
            return "when"

    for prefix in ["what is", "what was", "what does", "what did", "what can"]:
        if question_lower.startswith(prefix):
            return "what"

    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_qa_on_question(book_id: str, question: str, verbose: bool = False) -> Dict:
    """Run QA pipeline on a single question.

    Args:
        book_id: Book ID to query
        question: Question text
        verbose: Print debug info

    Returns:
        Result dict with answer, reasoning, confidence, etc.
    """
    try:
        result = retrieve_and_answer(
            book_id,
            question,
            verbose=verbose,
            attempt=1
        )
        return result
    except Exception as e:
        # Return error result
        return {
            "answer": "",
            "error": str(e),
            "reasoning": "error",
            "answer_score": 0.0,
            "cross_encoder_score": 0.0,
            "graph_confidence": 0.0,
        }


def evaluate_on_book(book_id: str, questions_data: List[Dict],
                     processed_dir: Path = None) -> Tuple[List[Dict], Dict]:
    """Evaluate QA pipeline on all questions for a single book.

    Args:
        book_id: Book ID to evaluate
        questions_data: List of question dicts (with 'question', 'ground_truth' keys)
        processed_dir: Path to processed data directory

    Returns:
        (predictions_list, book_metrics_dict)
    """
    predictions = []
    metrics_list = []

    for q_idx, q_data in enumerate(questions_data):
        question = q_data['question']
        ground_truth = q_data['ground_truth']
        question_type = detect_question_type(question)

        # Run QA pipeline
        result = run_qa_on_question(book_id, question, verbose=False)

        # Extract prediction
        predicted_answer = result.get('answer', '')

        # Compute metrics
        if predicted_answer and not result.get('error'):
            metrics = compute_answer_metrics(predicted_answer, ground_truth)
        else:
            metrics = {'em': 0.0, 'f1': 0.0, 'bleu': 0.0, 'rouge_l': 0.0}

        # Build prediction record
        prediction_record = {
            'book_id': book_id,
            'question': question,
            'question_type': question_type,
            'prediction': predicted_answer,
            'ground_truth': ground_truth,
            'reasoning': result.get('reasoning', 'unknown'),
            'confidence': result.get('answer_score', 0.0),
            'cross_encoder_score': result.get('cross_encoder_score', 0.0),
            'graph_confidence': result.get('graph_confidence', 0.0),
            'retrieved_chunk_ids': result.get('stages', {}).get('final_expanded_ids', []),
            'source_chunk_id': result.get('source_chunk_id'),
            'metrics': metrics,
        }

        predictions.append(prediction_record)
        metrics_list.append(metrics)

    # Compute per-book aggregate metrics
    if metrics_list:
        book_metrics = micro_average_metrics(metrics_list)
        book_metrics['num_questions'] = len(metrics_list)
        book_metrics['num_correct'] = sum(1 for m in metrics_list if m['em'] > 0)
    else:
        book_metrics = {
            'em': 0.0, 'f1': 0.0, 'bleu': 0.0, 'rouge_l': 0.0,
            'num_questions': 0, 'num_correct': 0
        }

    return predictions, book_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_predictions(all_predictions: List[Dict]) -> Dict:
    """Aggregate metrics across all predictions.

    Args:
        all_predictions: List of all prediction records

    Returns:
        Dict with overall, per-book, and per-question-type metrics
    """
    # Extract metrics lists
    metrics_list = [p['metrics'] for p in all_predictions]

    # Overall metrics
    overall_metrics = micro_average_metrics(metrics_list) if metrics_list else {}

    # Per-book metrics
    by_book = defaultdict(list)
    for pred in all_predictions:
        book_id = pred['book_id']
        by_book[book_id].append(pred['metrics'])

    per_book_metrics = {}
    for book_id, book_metrics_list in by_book.items():
        per_book_metrics[book_id] = micro_average_metrics(book_metrics_list)
        per_book_metrics[book_id]['num_questions'] = len(book_metrics_list)
        per_book_metrics[book_id]['num_correct'] = sum(
            1 for m in book_metrics_list if m['em'] > 0
        )

    # Per-question-type metrics
    by_qtype = defaultdict(list)
    for pred in all_predictions:
        qtype = pred.get('question_type', 'other')
        by_qtype[qtype].append(pred['metrics'])

    per_qtype_metrics = {}
    for qtype, qtype_metrics_list in by_qtype.items():
        per_qtype_metrics[qtype] = micro_average_metrics(qtype_metrics_list)
        per_qtype_metrics[qtype]['num_questions'] = len(qtype_metrics_list)
        per_qtype_metrics[qtype]['num_correct'] = sum(
            1 for m in qtype_metrics_list if m['em'] > 0
        )

    # Reasoning strategy breakdown
    by_reasoning = defaultdict(list)
    for pred in all_predictions:
        reasoning = pred.get('reasoning', 'unknown')
        by_reasoning[reasoning].append(pred['metrics'])

    per_reasoning_metrics = {}
    for reasoning, reason_metrics_list in by_reasoning.items():
        per_reasoning_metrics[reasoning] = micro_average_metrics(reason_metrics_list)
        per_reasoning_metrics[reasoning]['num_questions'] = len(reason_metrics_list)
        per_reasoning_metrics[reasoning]['num_correct'] = sum(
            1 for m in reason_metrics_list if m['em'] > 0
        )

    return {
        'overall': overall_metrics,
        'per_book': per_book_metrics,
        'per_question_type': per_qtype_metrics,
        'per_reasoning': per_reasoning_metrics,
        'num_total_questions': len(all_predictions),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. FILE I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(predictions: List[Dict], output_file: Path) -> None:
    """Save all predictions to JSON file.

    Args:
        predictions: List of prediction dicts
        output_file: Output JSON file path
    """
    # Remove large retrieved_chunk_ids to keep file size manageable
    for pred in predictions:
        if 'retrieved_chunk_ids' in pred:
            pred['retrieved_chunk_ids'] = pred['retrieved_chunk_ids'][:20]  # Keep top-20

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Predictions saved: {output_file}")


def save_metrics(metrics: Dict, output_file: Path) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Aggregated metrics dict
        output_file: Output JSON file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Metrics saved: {output_file}")


def print_metrics_summary(metrics: Dict) -> None:
    """Pretty-print metrics summary to console.

    Args:
        metrics: Aggregated metrics dict
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    overall = metrics['overall']
    print(f"\n📊 OVERALL METRICS (n={metrics['num_total_questions']} questions)")
    print(f"  Exact Match (EM):  {overall.get('em', 0):.4f}")
    print(f"  F1 Score:          {overall.get('f1', 0):.4f}")
    print(f"  ROUGE-L:           {overall.get('rouge_l', 0):.4f}")
    print(f"  BLEU:              {overall.get('bleu', 0):.4f}")

    # Per question type
    print(f"\n❓ PER QUESTION TYPE")
    for qtype in sorted(metrics['per_question_type'].keys()):
        qtype_metrics = metrics['per_question_type'][qtype]
        n = qtype_metrics.get('num_questions', 0)
        em = qtype_metrics.get('em', 0)
        f1 = qtype_metrics.get('f1', 0)
        print(f"  {qtype:8s} (n={n:3d}): EM={em:.4f}  F1={f1:.4f}")

    # Per reasoning strategy
    print(f"\n🧠 PER REASONING STRATEGY")
    for reasoning in sorted(metrics['per_reasoning'].keys()):
        reason_metrics = metrics['per_reasoning'][reasoning]
        n = reason_metrics.get('num_questions', 0)
        em = reason_metrics.get('em', 0)
        f1 = reason_metrics.get('f1', 0)
        print(f"  {reasoning:15s} (n={n:3d}): EM={em:.4f}  F1={f1:.4f}")

    # Top books by EM
    print(f"\n📚 TOP 5 BOOKS (by EM)")
    sorted_books = sorted(
        metrics['per_book'].items(),
        key=lambda x: x[1].get('em', 0),
        reverse=True
    )[:5]
    for book_id, book_metrics in sorted_books:
        em = book_metrics.get('em', 0)
        n = book_metrics.get('num_questions', 0)
        print(f"  {book_id[:30]:30s}: EM={em:.4f} (n={n})")

    print("\n" + "=" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main(num_books: int = 200, output_dir: Path = None, test_set_only: bool = True,
         verbose: bool = False):
    """Main evaluation loop.

    Args:
        num_books: Number of books to evaluate (default 200)
        output_dir: Directory to save output files
        test_set_only: Only evaluate on test set questions
        verbose: Print debug info
    """
    # Setup paths
    repo_root = Path(__file__).parent.parent
    qaps_file = repo_root / 'qaps.csv'
    processed_dir = find_data_directory()

    if output_dir is None:
        output_dir = repo_root / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 NARRATIVE QA EVALUATION")
    print("=" * 80)
    print(f"📁 QAPS File:      {qaps_file}")
    print(f"📁 Processed Data:  {processed_dir}")
    print(f"📁 Output Dir:      {output_dir}")
    print(f"📖 Num Books:       {num_books}")
    print(f"🔬 Test Set Only:   {test_set_only}")
    print("=" * 80 + "\n")

    # Load questions from qaps.csv
    print("📚 Loading QAPS data...")
    books_data = parse_qaps_csv(qaps_file, test_set_only=test_set_only)
    unique_books = get_unique_books(books_data, processed_dir)
    print(f"   Found {len(unique_books)} books, {sum(len(v) for v in books_data.values())} questions")

    # Limit to num_books
    selected_books = unique_books[:num_books]
    print(f"   Evaluating first {len(selected_books)} books\n")

    # Main evaluation loop
    all_predictions = []
    book_results = {}
    skipped_books = []

    for idx, book_id in enumerate(selected_books, 1):
        print(f"[{idx}/{len(selected_books)}] {book_id[:30]}...", end=" ", flush=True)

        # Check if book is indexed
        if not book_is_indexed(book_id, processed_dir):
            print(f"⚠️  SKIP (missing index files)")
            skipped_books.append(book_id)
            continue

        # Get questions for this book
        questions_data = books_data.get(book_id, [])
        if not questions_data:
            print(f"⚠️  SKIP (no questions)")
            skipped_books.append(book_id)
            continue

        # Evaluate this book
        try:
            preds, book_metrics = evaluate_on_book(
                book_id, questions_data, processed_dir
            )
            all_predictions.extend(preds)
            book_results[book_id] = book_metrics

            em = book_metrics.get('em', 0)
            f1 = book_metrics.get('f1', 0)
            n = book_metrics.get('num_questions', 0)
            print(f"✅ EM={em:.4f}, F1={f1:.4f} (n={n})")

        except Exception as e:
            print(f"❌ ERROR: {str(e)[:50]}")
            skipped_books.append(book_id)
            continue

    print(f"\n{'='*80}")
    print(f"✅ Successfully evaluated {len(all_predictions)} predictions")
    print(f"⏭️  Skipped {len(skipped_books)} books")
    print(f"{'='*80}\n")

    # Aggregate metrics
    print("📊 Aggregating metrics...")
    metrics = aggregate_predictions(all_predictions)

    # Save outputs
    predictions_file = output_dir / 'predictions.json'
    metrics_file = output_dir / 'metrics.json'

    save_predictions(all_predictions, predictions_file)
    save_metrics(metrics, metrics_file)

    # Print summary
    print_metrics_summary(metrics)

    # Save summary to text file
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Books evaluated: {len(selected_books)}\n")
        f.write(f"Books skipped: {len(skipped_books)}\n")
        f.write(f"Total predictions: {len(all_predictions)}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  EM: {metrics['overall'].get('em', 0):.4f}\n")
        f.write(f"  F1: {metrics['overall'].get('f1', 0):.4f}\n")
        f.write(f"  ROUGE-L: {metrics['overall'].get('rouge_l', 0):.4f}\n")

    print(f"💾 Summary saved: {summary_file}")
    print(f"\n✨ Evaluation complete! Results saved to {output_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Narrative QA pipeline on QAPS dataset'
    )
    parser.add_argument(
        '--num-books',
        type=int,
        default=200,
        help='Number of books to evaluate (default: 200)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (default: ./results/TIMESTAMP)'
    )
    parser.add_argument(
        '--test-set-only',
        action='store_true',
        default=False,
        help='Only evaluate test set questions (default: False)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print verbose debug output'
    )

    args = parser.parse_args()

    main(
        num_books=args.num_books,
        output_dir=args.output_dir,
        test_set_only=args.test_set_only,
        verbose=args.verbose
    )
