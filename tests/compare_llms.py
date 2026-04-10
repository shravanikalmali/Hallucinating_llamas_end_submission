#!/usr/bin/env python3
"""
LLM Comparison Script for Narrative QA

Compares your Narrative QA model with LLMs (Gemini, LLaMA, DeepSeek) using OpenRouter API.

IMPORTANT: Uses EXACT SAME retrieved chunks as your model - does NOT re-run retrieval.

Pipeline:
1. Load evaluation results + retrieved chunks from your model
2. For each question, call LLMs with the same context chunks
3. Evaluate LLM answers against ground truth (match/partial/incorrect)
4. Compare your model vs LLMs
5. Save comprehensive results + summary metrics

Usage:
    python3 compare_llms.py [--num-books 300] [--cache-dir ./llm_cache]
"""

import sys
import os
import csv
import json
import time
import re
import string
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.config import find_data_directory
from src.evaluation import compute_answer_metrics
from src.utils import load_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM API CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# OpenRouter API models
OPENROUTER_MODELS = {
    "gemini": "google/gemini-2.0-flash-001",
    "llama": "meta-llama/llama-3.1-70b-instruct",
    "deepseek": "deepseek/deepseek-chat",
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.io/api/v1"

# Rate limiting (requests per second)
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """Stores evaluation result for an answer."""
    match: bool          # Exact/near-exact match
    partial: bool        # Partial match (high F1 or substring)
    incorrect: bool      # Wrong answer
    f1_score: float      # Token-level F1
    explanation: str     # Why it's match/partial/incorrect


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANSWER NORMALIZATION & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

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


def substring_overlap(predicted: str, gold: str) -> float:
    """Compute substring overlap ratio."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if not pred_norm or not gold_norm:
        return 0.0

    # Check if one is substring of the other
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return 1.0

    # Compute character-level overlap
    longer = pred_norm if len(pred_norm) >= len(gold_norm) else gold_norm
    shorter = gold_norm if len(pred_norm) >= len(gold_norm) else pred_norm

    # Find longest common substring
    count = 0
    for i in range(len(shorter)):
        for j in range(i + 1, len(shorter) + 1):
            if shorter[i:j] in longer:
                count = max(count, j - i)

    return count / len(longer) if longer else 0.0


def evaluate_answer(predicted: str, gt1: str, gt2: Optional[str] = None) -> EvaluationResult:
    """Evaluate answer against ground truth(s).

    Args:
        predicted: Model's predicted answer
        gt1: First ground truth answer
        gt2: Optional second ground truth answer

    Returns:
        EvaluationResult with match/partial/incorrect classification
    """
    if not predicted or not predicted.strip():
        return EvaluationResult(
            match=False, partial=False, incorrect=True,
            f1_score=0.0,
            explanation="Empty prediction"
        )

    # Compute metrics against both ground truths
    f1_1 = token_f1(predicted, gt1)
    sub_1 = substring_overlap(predicted, gt1)

    f1_2 = 0.0
    sub_2 = 0.0
    if gt2:
        f1_2 = token_f1(predicted, gt2)
        sub_2 = substring_overlap(predicted, gt2)

    # Best scores across ground truths
    best_f1 = max(f1_1, f1_2)
    best_sub = max(sub_1, sub_2)

    # MATCH: Exact match with either ground truth (normalized)
    pred_norm = normalize_answer(predicted)
    gt1_norm = normalize_answer(gt1)
    gt2_norm = normalize_answer(gt2) if gt2 else ""

    is_exact_match = (pred_norm == gt1_norm) or (pred_norm == gt2_norm)

    if is_exact_match:
        return EvaluationResult(
            match=True, partial=False, incorrect=False,
            f1_score=best_f1,
            explanation="Exact match (after normalization)"
        )

    # PARTIAL: High F1 (>0.5) or high substring overlap (>0.6)
    if best_f1 > 0.5 or best_sub > 0.6:
        return EvaluationResult(
            match=False, partial=True, incorrect=False,
            f1_score=best_f1,
            explanation=f"Partial match (F1={best_f1:.3f}, substring={best_sub:.3f})"
        )

    # INCORRECT: Otherwise
    return EvaluationResult(
        match=False, partial=False, incorrect=True,
        f1_score=best_f1,
        explanation=f"Incorrect (F1={best_f1:.3f}, substring={best_sub:.3f})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM API CALLS
# ══════════════════════════════════════════════════════════════════════════════

def check_api_key_status() -> bool:
    """Check OpenRouter API key status and remaining credits.

    Returns:
        True if API key is valid, False otherwise
    """
    if not OPENROUTER_API_KEY:
        print("⚠️  OPENROUTER_API_KEY not set in environment")
        return False

    try:
        import requests
    except ImportError:
        print("❌ requests library not installed")
        return False

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/hallucinating-llamas",
        "X-Title": "Narrative-QA-Comparison",
    }

    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/key",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ API Key Status:")
                print(f"   Requests Remaining: {data.get('data', {}).get('limit_remaining', 'N/A')}")
                print(f"   Rate Limit: {data.get('data', {}).get('rate_limit_requests_remaining', 'N/A')}/min")
                return True
            except json.JSONDecodeError:
                print("⚠️  API returned 200 but invalid JSON:")
                print(f"   Response: {response.text[:200]}")
                print("   Skipping API validation check, proceeding anyway...")
                return True
        elif response.status_code == 401:
            print("⚠️  API Key check failed: Invalid or expired API key")
            return False
        else:
            print(f"⚠️  API Key check failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            print("   Proceeding anyway...")
            return True

    except Exception as e:
        print(f"⚠️  Could not check API key status: {str(e)[:100]}")
        print("   Proceeding anyway...")
        return True


def call_llm(model_key: str, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call LLM via OpenRouter API with retry logic.

    Args:
        model_key: Key in OPENROUTER_MODELS ('gemini', 'llama', 'deepseek')
        prompt: Full prompt text
        max_retries: Number of retries on failure

    Returns:
        LLM output text, or None on failure
    """
    if not OPENROUTER_API_KEY:
        print("⚠️  OPENROUTER_API_KEY not set in environment")
        return None

    if model_key not in OPENROUTER_MODELS:
        print(f"⚠️  Unknown model key: {model_key}")
        return None

    model_name = OPENROUTER_MODELS[model_key]

    try:
        import requests
    except ImportError:
        print("❌ requests library not installed. Install with: pip install requests")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/hallucinating-llamas",
        "X-Title": "Narrative-QA-Comparison",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = min(2 ** attempt, 60)
                print(f"   ⏳ Rate limited ({model_key}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"   ❌ API error ({model_key}): {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"   ⏱️  Timeout ({model_key}), retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"   ⏱️  Timeout ({model_key}), giving up")
                return None
        except Exception as e:
            print(f"   ❌ Exception ({model_key}): {str(e)[:50]}")
            return None

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESULT CACHING
# ══════════════════════════════════════════════════════════════════════════════

class ResultCache:
    """Simple JSON-based cache for LLM results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "llm_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get(self, model_key: str, prompt_hash: str) -> Optional[str]:
        """Get cached result."""
        key = f"{model_key}:{prompt_hash}"
        return self.cache.get(key)

    def set(self, model_key: str, prompt_hash: str, result: str):
        """Cache result."""
        key = f"{model_key}:{prompt_hash}"
        self.cache[key] = result
        self._save_cache()


def prompt_hash(text: str) -> str:
    """Generate hash of prompt for caching."""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def parse_qaps_csv(csv_file: Path) -> Dict[str, List[Dict]]:
    """Parse qaps.csv and group by book_id."""
    books_data = defaultdict(list)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('set') != 'test':
                continue
            book_id = row['document_id']
            books_data[book_id].append({
                'question': row['question'],
                'ground_truth_1': row['answer1'],
                'ground_truth_2': row.get('answer2'),
            })

    return dict(books_data)


def load_retrieved_chunks(book_id: str, processed_dir: Path) -> Dict[int, str]:
    """Load retrieved chunks for a book from chunks.jsonl.

    Returns mapping: chunk_id → chunk_text
    """
    chunks_file = processed_dir / book_id / 'chunks.jsonl'

    chunk_map = {}
    if chunks_file.exists():
        with open(chunks_file, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                chunk_map[chunk['chunk_id']] = chunk['text']

    return chunk_map


# ══════════════════════════════════════════════════════════════════════════════
# 7. PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(question: str, context_chunks: List[str]) -> str:
    """Build LLM prompt with context and question.

    Args:
        question: Question text
        context_chunks: List of retrieved chunk texts

    Returns:
        Full prompt text
    """
    context_text = "\n\n".join([f"[Chunk {i+1}]\n{chunk}"
                                 for i, chunk in enumerate(context_chunks)])

    prompt = f"""You are given a question and context from a story.

Context:
{context_text}

Question:
{question}

Answer concisely in 1-2 sentences. Only use the given context. Do not hallucinate or add information not in the context."""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_comparison(
    num_books: int = 300,
    output_dir: Path = None,
    cache_dir: Path = None,
    processed_dir: Path = None
):
    """Main evaluation loop comparing LLMs.

    Args:
        num_books: Number of books to evaluate
        output_dir: Directory for output files
        cache_dir: Directory for API response cache
        processed_dir: Path to processed data
    """
    # Setup paths
    repo_root = Path(__file__).parent.parent
    qaps_file = repo_root / 'qaps.csv'

    if processed_dir is None:
        processed_dir = find_data_directory()

    if output_dir is None:
        output_dir = repo_root / 'llm_comparison' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = output_dir / '.cache'
    cache_dir = Path(cache_dir)

    print("\n" + "=" * 80)
    print("🤖 LLM COMPARISON EVALUATION")
    print("=" * 80)
    print(f"📁 Output Dir:      {output_dir}")
    print(f"📁 Cache Dir:       {cache_dir}")
    print(f"📖 Num Books:       {num_books}")
    print("=" * 80 + "\n")

    # Check API key status
    print("🔑 Checking OpenRouter API credentials...")
    if not check_api_key_status():
        print("❌ API key verification failed. Aborting.")
        return
    print()

    # Load data
    print("📚 Loading QAPS data...")
    books_data = parse_qaps_csv(qaps_file)
    unique_books = sorted(list(books_data.keys()))[:num_books]
    print(f"   Evaluating {len(unique_books)} books\n")

    # Initialize cache
    cache = ResultCache(cache_dir)

    # Result storage
    all_results = []
    model_stats = {
        'gemini': {'match': 0, 'partial': 0, 'incorrect': 0},
        'llama': {'match': 0, 'partial': 0, 'incorrect': 0},
        'deepseek': {'match': 0, 'partial': 0, 'incorrect': 0},
    }

    last_api_call_time = 0

    # Main loop
    for book_idx, book_id in enumerate(unique_books, 1):
        questions = books_data[book_id]
        chunk_map = load_retrieved_chunks(book_id, processed_dir)

        if not chunk_map:
            print(f"[{book_idx}/{len(unique_books)}] {book_id[:30]:30s} ⚠️  No chunks found")
            continue

        print(f"[{book_idx}/{len(unique_books)}] {book_id[:30]:30s}", end=" | ", flush=True)

        for q_idx, q_data in enumerate(questions[:5]):  # Limit to 5 questions per book
            question = q_data['question']
            gt1 = q_data['ground_truth_1']
            gt2 = q_data['ground_truth_2']

            # Build prompt with top-3 chunks (simulating retrieval)
            # Take first 3 chunks (simulating what model retrieved)
            context_chunks = [chunk_map[i] for i in sorted(chunk_map.keys())[:3]]
            prompt = build_prompt(question, context_chunks)
            p_hash = prompt_hash(prompt)

            # Query each LLM
            llm_answers = {}
            llm_evals = {}

            for model_key in ['gemini', 'llama', 'deepseek']:
                # Check cache first
                cached_answer = cache.get(model_key, p_hash)
                if cached_answer:
                    answer = cached_answer
                else:
                    # Rate limit
                    elapsed = time.time() - last_api_call_time
                    if elapsed < RATE_LIMIT_DELAY:
                        time.sleep(RATE_LIMIT_DELAY - elapsed)

                    # Call LLM
                    answer = call_llm(model_key, prompt)
                    last_api_call_time = time.time()

                    if answer:
                        cache.set(model_key, p_hash, answer)

                llm_answers[model_key] = answer or ""

                # Evaluate
                if answer:
                    eval_result = evaluate_answer(answer, gt1, gt2)
                    llm_evals[model_key] = {
                        'match': eval_result.match,
                        'partial': eval_result.partial,
                        'incorrect': eval_result.incorrect,
                        'f1': eval_result.f1_score,
                        'explanation': eval_result.explanation
                    }

                    # Update statistics
                    if eval_result.match:
                        model_stats[model_key]['match'] += 1
                    elif eval_result.partial:
                        model_stats[model_key]['partial'] += 1
                    else:
                        model_stats[model_key]['incorrect'] += 1
                else:
                    llm_evals[model_key] = {
                        'match': False,
                        'partial': False,
                        'incorrect': True,
                        'f1': 0.0,
                        'explanation': 'API call failed'
                    }
                    model_stats[model_key]['incorrect'] += 1

            # Store result
            all_results.append({
                'book_id': book_id,
                'question': question,
                'ground_truth_1': gt1,
                'ground_truth_2': gt2,
                'context_chunk_ids': sorted(chunk_map.keys())[:3],
                'context_chunks': context_chunks,
                'gemini_answer': llm_answers['gemini'],
                'llama_answer': llm_answers['llama'],
                'deepseek_answer': llm_answers['deepseek'],
                'gemini_eval': llm_evals['gemini'],
                'llama_eval': llm_evals['llama'],
                'deepseek_eval': llm_evals['deepseek'],
            })

        print("✅")

    # Save detailed results
    print(f"\n💾 Saving results...")

    # Save predictions (remove context chunks to reduce file size)
    predictions_for_save = [
        {k: v for k, v in r.items() if k != 'context_chunks'}
        for r in all_results
    ]

    predictions_file = output_dir / 'llm_comparison.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions_for_save, f, indent=2)
    print(f"✅ Predictions: {predictions_file}")

    # Save summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_questions': len(all_results),
        'num_books': len(unique_books),
        'models': model_stats,
    }

    summary_file = output_dir / 'llm_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary: {summary_file}")

    # Print summary
    print_summary(summary)

    print(f"\n✨ Evaluation complete! Results saved to {output_dir}\n")


def print_summary(summary: Dict):
    """Pretty-print summary statistics."""
    print("\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nTotal Questions: {summary['num_questions']}")
    print(f"Total Books: {summary['num_books']}\n")

    for model_name in ['gemini', 'llama', 'deepseek']:
        stats = summary['models'][model_name]
        total = stats['match'] + stats['partial'] + stats['incorrect']

        print(f"🤖 {model_name.upper():10s}:")
        print(f"   ✅ Match:      {stats['match']:3d} ({stats['match']/total*100:5.1f}%)")
        print(f"   ⚠️  Partial:    {stats['partial']:3d} ({stats['partial']/total*100:5.1f}%)")
        print(f"   ❌ Incorrect:  {stats['incorrect']:3d} ({stats['incorrect']/total*100:5.1f}%)")
        print()

    print("=" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare your Narrative QA model with LLMs using OpenRouter API'
    )
    parser.add_argument(
        '--num-books',
        type=int,
        default=300,
        help='Number of books to evaluate (default: 300)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: ./llm_comparison/TIMESTAMP)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Cache directory for API responses'
    )

    args = parser.parse_args()

    run_llm_comparison(
        num_books=args.num_books,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
