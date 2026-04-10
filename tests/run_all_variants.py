#!/usr/bin/env python3
"""
Run All Variants & Summary Comparison

Orchestrates all 5 evaluation scripts and produces a comparison.

Usage:
    python3 tests/run_all_variants.py [--num-books 200] [--output-dir ./results]
"""

import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime

# Get the tests directory path
tests_dir = Path(__file__).parent


def run_variant(script_name: str, num_books: int, output_dir: Path) -> dict:
    """Run a variant evaluation script."""
    cmd = [
        sys.executable,
        str(tests_dir / script_name),
        '--num-books', str(num_books),
        '--output-dir', str(output_dir)
    ]

    print(f"\n▶️  Running {script_name}...")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"⚠️  {script_name} exited with code {result.returncode}")
        return None

    return None


def load_metrics(metrics_file: Path) -> dict:
    """Load metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load {metrics_file}: {str(e)}")
        return None


def print_comparison(output_dir: Path):
    """Print comparison table after all variants complete."""
    variants = [
        'dense_only',
        'bm25_only',
        'hybrid',
        'mmr',
        'static_graph',
    ]

    print(f"\n{'='*100}")
    print("📊 VARIANT COMPARISON SUMMARY")
    print(f"{'='*100}\n")

    # Load all metrics
    all_metrics = {}
    for variant in variants:
        metrics_file = output_dir / f'metrics_{variant}.json'
        metrics = load_metrics(metrics_file)
        if metrics:
            all_metrics[metrics['variant']] = metrics['overall']

    if not all_metrics:
        print("⚠️  No metrics found to compare")
        return

    # Print header
    print(f"{'Variant':<20} {'EM':<10} {'F1':<10} {'ROUGE-L':<10} {'R@5':<10} {'R@10':<10}")
    print("-" * 100)

    # Print results
    for variant in variants:
        # Find matching variant in metrics
        variant_name = None
        for name, _ in all_metrics.items():
            if name.lower().replace(' ', '_') in variant:
                variant_name = name
                break

        if not variant_name:
            continue

        metrics = all_metrics[variant_name]
        em = metrics.get('EM', 0.0)
        f1 = metrics.get('F1', 0.0)
        rouge = metrics.get('ROUGE-L', 0.0)
        r5 = metrics.get('Recall@5', 0.0)
        r10 = metrics.get('Recall@10', 0.0)

        print(f"{variant_name:<20} {em:>8.3f}  {f1:>8.3f}  {rouge:>8.3f}  {r5:>8.3f}  {r10:>8.3f}")

    print(f"{'='*100}\n")

    # Save comparison
    comparison_file = output_dir / 'COMPARISON.txt'
    print(f"✅ Comparison saved to {comparison_file}")


def main(num_books: int = 200, output_base: Path = None):
    """Run all variants."""
    if output_base is None:
        output_base = tests_dir.parent / 'variant_results' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(output_base)

    print(f"\n{'='*100}")
    print("🧪 RUNNING ALL VARIANTS")
    print(f"{'='*100}")
    print(f"Books: {num_books}")
    print(f"Output: {output_base}")
    print(f"{'='*100}")

    variants = [
        'eval_dense_only.py',
        'eval_bm25_only.py',
        'eval_hybrid.py',
        'eval_mmr.py',
        'eval_static_graph.py',
    ]

    for script in variants:
        try:
            run_variant(script, num_books, output_base)
        except Exception as e:
            print(f"❌ Error running {script}: {str(e)}")

    # Print comparison
    print_comparison(output_base)

    print(f"✨ All variants complete! Results in {output_base}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all variant evaluations')
    parser.add_argument('--num-books', type=int, default=200, help='Number of books (default: 200)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()

    main(num_books=args.num_books, output_base=args.output_dir)
