#!/usr/bin/env python3
"""
Ablation Experiment Runner

Systematically evaluates different retrieval configurations to determine
which components contribute the most to retrieval quality.

Usage:
    python experiments/run_experiments.py \\
        --book <BOOK_ID> --questions-file <PATH> \\
        [--processed-dir <DIR>] [--output-dir experiments/results]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval_config import RETRIEVAL_CONFIG, set_config, reset_config, config_summary
from src.retrieval.pipeline import retrieve_and_answer
from src.config import PROCESSED_DIR


# ── Ablation Configurations ───────────────────────────────────────────────────

ABLATIONS = [
    ("dense_only", {
        "use_dense": True, "use_bm25": False, "use_mmr": False,
        "use_neighbors": False, "use_graph_expansion": False,
        "use_graph_reasoning": False, "use_cross_encoder": False,
    }),
    ("dense+bm25", {
        "use_dense": True, "use_bm25": True, "use_mmr": False,
        "use_neighbors": False, "use_graph_expansion": False,
        "use_graph_reasoning": False, "use_cross_encoder": False,
    }),
    ("dense+bm25+mmr", {
        "use_dense": True, "use_bm25": True, "use_mmr": True,
        "use_neighbors": False, "use_graph_expansion": False,
        "use_graph_reasoning": False, "use_cross_encoder": False,
    }),
    ("dense+bm25+mmr+neighbors", {
        "use_dense": True, "use_bm25": True, "use_mmr": True,
        "use_neighbors": True, "use_graph_expansion": False,
        "use_graph_reasoning": False, "use_cross_encoder": False,
    }),
    ("dense+bm25+mmr+neighbors+graph_exp", {
        "use_dense": True, "use_bm25": True, "use_mmr": True,
        "use_neighbors": True, "use_graph_expansion": True,
        "use_graph_reasoning": False, "use_cross_encoder": False,
    }),
    ("full_no_crossenc", {
        "use_dense": True, "use_bm25": True, "use_mmr": True,
        "use_neighbors": True, "use_graph_expansion": True,
        "use_graph_reasoning": True, "use_cross_encoder": False,
    }),
    ("full", {
        "use_dense": True, "use_bm25": True, "use_mmr": True,
        "use_neighbors": True, "use_graph_expansion": True,
        "use_graph_reasoning": True, "use_cross_encoder": True,
    }),
    ("bm25_only", {
        "use_dense": False, "use_bm25": True, "use_mmr": False,
        "use_neighbors": False, "use_graph_expansion": False,
        "use_graph_reasoning": False, "use_cross_encoder": False,
    }),
]


def load_questions(path: str) -> list[dict]:
    """Load questions from a JSONL file."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_ablation(
    name: str, config: dict, book_id: str, questions: list[dict],
    processed_dir: Path, verbose: bool = False,
) -> dict:
    """Run a single ablation configuration and return metrics."""
    reset_config()
    set_config(config)

    print(f"\n{'━' * 60}")
    print(f"  Ablation: {name}")
    print(f"  Config:   {config_summary()}")
    print(f"{'━' * 60}")

    results = []
    recall_at_5, recall_at_10 = 0, 0
    failures = {"retrieval": 0, "extraction": 0, "none": 0}

    for i, q_item in enumerate(questions, 1):
        q_text = q_item["question"]
        gold = q_item.get("answer")

        try:
            result = retrieve_and_answer(
                book_id=book_id, question=q_text, processed_dir=processed_dir,
                gold_answer=gold, verbose=verbose,
            )
            results.append(result)

            log = result.get("retrieval_log", {})
            if log.get("contains_answer") is not None:
                rank = log.get("rank_of_answer_chunk", -1)
                if 1 <= rank <= 5:
                    recall_at_5 += 1
                if 1 <= rank <= 10:
                    recall_at_10 += 1
                ft = log.get("failure_type")
                failures[ft if ft else "none"] += 1

            print(f"  [{i}/{len(questions)}] {q_text[:60]}… → "
                  f"{'✓' if log.get('contains_answer', '?') else '✗'}")
        except Exception as e:
            print(f"  [{i}/{len(questions)}] ERROR: {e}")
            failures["retrieval"] += 1

    n = len(questions)
    metrics = {
        "name": name,
        "config": config,
        "num_questions": n,
        "recall_at_5": recall_at_5 / n if n else 0,
        "recall_at_10": recall_at_10 / n if n else 0,
        "failures": failures,
        "raw_recall_5": recall_at_5,
        "raw_recall_10": recall_at_10,
    }

    print(f"\n  Results: Recall@5={metrics['recall_at_5']:.2%}, "
          f"Recall@10={metrics['recall_at_10']:.2%}")
    print(f"  Failures: {failures}")

    return {"metrics": metrics, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--book", required=True, help="Book ID")
    parser.add_argument("--questions-file", required=True, help="JSONL with questions + answers")
    parser.add_argument("--processed-dir", help="Processed data directory")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--configs", nargs="*", help="Run only specific ablation names")
    args = parser.parse_args()

    processed = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR

    if args.dry_run:
        print("DRY RUN — Ablation configurations:\n")
        for name, config in ABLATIONS:
            reset_config()
            set_config(config)
            print(f"  {name:40s} → {config_summary()}")
        return

    questions = load_questions(args.questions_file)
    print(f"Loaded {len(questions)} questions from {args.questions_file}")

    ablations_to_run = ABLATIONS
    if args.configs:
        ablations_to_run = [(n, c) for n, c in ABLATIONS if n in args.configs]

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_metrics = []
    for name, config in ablations_to_run:
        result = run_ablation(name, config, args.book, questions, processed, args.verbose)
        all_metrics.append(result["metrics"])

        # Save individual results
        detail_file = out_dir / f"{name}_{timestamp}.jsonl"
        with open(detail_file, "w", encoding="utf-8") as f:
            for r in result["results"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary
    summary_file = out_dir / f"ablation_summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n\n{'═' * 70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Config':<40s} {'Recall@5':>10s} {'Recall@10':>10s}")
    print(f"  {'─' * 40} {'─' * 10} {'─' * 10}")
    for m in all_metrics:
        print(f"  {m['name']:<40s} {m['recall_at_5']:>9.1%} {m['recall_at_10']:>10.1%}")
    print(f"{'═' * 70}")
    print(f"\n  Summary saved → {summary_file}")
    print(f"  Details saved → {out_dir}/")


if __name__ == "__main__":
    main()
