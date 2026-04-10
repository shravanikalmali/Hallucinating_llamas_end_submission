#!/usr/bin/env python3
"""
Variant 1: Dense Only - FAISS embedding-based retrieval only

No BM25, no MMR, no graph features.

Usage:
    python3 tests/eval_dense_only.py [--num-books 200] [--output-dir ./results]
"""

import sys
import csv
import json
import re
import string
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import find_data_directory
from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval_config import set_config


# ══════════════════════════════════════════════════════════════════════════════
# SHARED METRICS (Copied for independence)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 score."""
    from collections import Counter
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


def exact_match(predicted: str, gold1: str, gold2: Optional[str] = None) -> bool:
    """Check if prediction is exact match with either ground truth."""
    pred_norm = normalize_answer(predicted)
    if pred_norm == normalize_answer(gold1):
        return True
    if gold2 and pred_norm == normalize_answer(gold2):
        return True
    return False


def rouge_l(predicted: str, gold: str) -> float:
    """Compute ROUGE-L (longest common subsequence)."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(gold_tokens) if gold_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def compute_recall_at_k(retrieved_chunk_ids: List[int], relevant_chunk_ids: List[int], k: int) -> bool:
    """Check if any relevant chunk is in top-k retrieved chunks."""
    if not relevant_chunk_ids:
        return True
    top_k_ids = set(retrieved_chunk_ids[:k])
    return any(cid in top_k_ids for cid in relevant_chunk_ids)


def identify_relevant_chunks(book_id: str, question: str, processed_dir: Path) -> List[int]:
    """Identify which chunks contain the answer (simple heuristic)."""
    try:
        chunks_file = processed_dir / book_id / 'chunks.jsonl'
        if not chunks_file.exists():
            return []
        keywords = set(normalize_answer(question).split())
        keywords.discard('')
        relevant_chunk_ids = []
        with open(chunks_file, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                chunk_text = normalize_answer(chunk['text'])
                chunk_keywords = set(chunk_text.split())
                overlap = len(keywords & chunk_keywords)
                if overlap >= 2:
                    relevant_chunk_ids.append(chunk['chunk_id'])
        return relevant_chunk_ids
    except Exception as e:
        print(f"    ⚠️  Error: {str(e)[:50]}")
        return []


def parse_qaps_csv(csv_file: Path, limit_books: int = 200) -> Dict[str, List[Dict]]:
    """Parse qaps.csv and group by book_id."""
    books_data = defaultdict(list)
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if False:  # row.get('set') != 'test':
                continue
            book_id = row['document_id']
            if len(books_data[book_id]) < 10: books_data[book_id].append({
                'question': row['question'],
                'ground_truth_1': row['answer1'],
                'ground_truth_2': row.get('answer2'),
            })
    books_dict = {k: v for k, v in list(books_data.items())[:limit_books]}
    return books_dict


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dense_only(num_books: int = 200, output_dir: Path = None, processed_dir: Path = None) -> Dict:
    """Evaluate Dense Only variant."""
    repo_root = Path(__file__).parent.parent
    qaps_file = repo_root / 'qaps.csv'

    if processed_dir is None:
        processed_dir = find_data_directory()

    if output_dir is None:
        output_dir = repo_root / 'variant_results' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"🔬 VARIANT: Dense Only")
    print(f"{'='*80}")
    print(f"Config: use_dense=True, use_bm25=False, use_mmr=False, use_graph=False")
    print(f"Books: {num_books}")
    print(f"{'='*80}\n")

    # Set configuration
    set_config({
        'use_dense': True,
        'use_bm25': False,
        'use_mmr': False,
        'use_graph_expansion': False,
        'use_graph_reasoning': False,
        'use_neighbors': False,
        'use_hybrid_llm': False,
    })

    # Load questions
    books_data = parse_qaps_csv(qaps_file, limit_books=2000)
    with open("/tmp/valid_100_books.txt", "r") as f: target_books = [l.strip() for l in f]; unique_books = [b for b in target_books if b in books_data][:num_books]
    total_questions = sum(len(qs) for qs in books_data.values())

    print(f"📚 Loaded {len(unique_books)} books with {total_questions} questions\n")

    all_predictions = []
    all_em = []
    all_f1 = []
    all_rouge_l = []
    all_recall_at_5 = []
    all_recall_at_10 = []

    # Evaluate
    for book_idx, book_id in enumerate(unique_books, 1):
        questions = books_data[book_id]
        print(f"[{book_idx}/{len(unique_books)}] {book_id[:40]:40s}", end=" | ", flush=True)

        for q_data in questions:
            question = q_data['question']
            gt1 = q_data['ground_truth_1']
            gt2 = q_data['ground_truth_2']

            try:
                result = retrieve_and_answer(book_id=book_id, question=question, processed_dir=processed_dir)

                if not result or 'answer' not in result:
                    prediction = ""
                    retrieved_chunks = []
                else:
                    prediction = result['answer']
                    retrieval_log = result.get('retrieval_log', {})
                    retrieved_chunks = retrieval_log.get('retrieved_chunk_ids', [])

                em = exact_match(prediction, gt1, gt2)
                f1 = token_f1(prediction, gt1)
                rouge = rouge_l(prediction, gt1)

                relevant_chunks = identify_relevant_chunks(book_id, question, processed_dir)
                recall_5 = compute_recall_at_k(retrieved_chunks, relevant_chunks, 5)
                recall_10 = compute_recall_at_k(retrieved_chunks, relevant_chunks, 10)

                all_predictions.append({
                    'book_id': book_id,
                    'question': question,
                    'ground_truth_1': gt1,
                    'ground_truth_2': gt2,
                    'prediction': prediction,
                    'em': em,
                    'f1': f1,
                    'rouge_l': rouge,
                    'recall_at_5': recall_5,
                    'recall_at_10': recall_10,
                })

                all_em.append(1.0 if em else 0.0)
                all_f1.append(f1)
                all_rouge_l.append(rouge)
                all_recall_at_5.append(1.0 if recall_5 else 0.0)
                all_recall_at_10.append(1.0 if recall_10 else 0.0)

            except Exception as e:
                print(f"\n    ⚠️  Error: {str(e)[:50]}")
                all_em.append(0.0)
                all_f1.append(0.0)
                all_rouge_l.append(0.0)
                all_recall_at_5.append(0.0)
                all_recall_at_10.append(0.0)

        print("✅")

    # Compute metrics
    metrics = {
        'variant': 'Dense Only',
        'config': {
            'use_dense': True,
            'use_bm25': False,
            'use_mmr': False,
            'use_graph': False,
        },
        'overall': {
            'EM': float(np.mean(all_em)) if all_em else 0.0,
            'F1': float(np.mean(all_f1)) if all_f1 else 0.0,
            'ROUGE-L': float(np.mean(all_rouge_l)) if all_rouge_l else 0.0,
            'Recall@5': float(np.mean(all_recall_at_5)) if all_recall_at_5 else 0.0,
            'Recall@10': float(np.mean(all_recall_at_10)) if all_recall_at_10 else 0.0,
        },
        'num_questions': len(all_predictions),
        'timestamp': datetime.now().isoformat(),
    }

    # Save results
    pred_file = output_dir / 'predictions_dense_only.json'
    with open(pred_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    metrics_file = output_dir / 'metrics_dense_only.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Predictions: {pred_file}")
    print(f"✅ Metrics: {metrics_file}")

    print(f"\n{'='*80}")
    print(f"📊 RESULTS: Dense Only")
    print(f"{'='*80}")
    print(f"EM:       {metrics['overall']['EM']:.3f}")
    print(f"F1:       {metrics['overall']['F1']:.3f}")
    print(f"ROUGE-L:  {metrics['overall']['ROUGE-L']:.3f}")
    print(f"Recall@5: {metrics['overall']['Recall@5']:.3f}")
    print(f"Recall@10:{metrics['overall']['Recall@10']:.3f}")
    print(f"{'='*80}\n")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Dense Only variant')
    parser.add_argument('--num-books', type=int, default=200, help='Number of books (default: 200)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()

    evaluate_dense_only(num_books=args.num_books, output_dir=args.output_dir)
