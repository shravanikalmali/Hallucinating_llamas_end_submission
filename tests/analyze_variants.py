#!/usr/bin/env python3
"""
Analyze Variant Results - Compare and visualize results

Usage:
    python3 tests/analyze_variants.py --results-dir variant_results/YYYYMMDD_HHMMSS

Outputs:
    - comparison_table.txt: Formatted comparison table
    - ablation_analysis.txt: Ablation impact analysis
    - detailed_comparison.json: Full comparison data
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_metrics(results_dir: Path) -> dict:
    """Load all metrics_*.json files and organize."""
    metrics_files = sorted(results_dir.glob('metrics_*.json'))

    all_data = {}
    for f in metrics_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            variant_name = data['variant']
            all_data[variant_name] = data

    return all_data


def print_comparison_table(metrics_data: dict):
    """Print formatted comparison table."""
    if not metrics_data:
        print("No metrics data found")
        return

    print("\n" + "=" * 100)
    print("📊 VARIANT COMPARISON TABLE")
    print("=" * 100)

    # Header
    variants = list(metrics_data.keys())
    metric_names = ['EM', 'F1', 'ROUGE-L', 'Recall@5', 'Recall@10']

    print(f"{'Variant':<20} ", end="")
    for m in metric_names:
        print(f"{m:>12} ", end="")
    print()
    print("-" * 100)

    # Data rows
    for variant in variants:
        metrics = metrics_data[variant]['overall']
        print(f"{variant:<20} ", end="")
        for m in metric_names:
            val = metrics.get(m, 0.0)
            print(f"{val:>11.3f}% ", end="")
        print()

    print("=" * 100 + "\n")


def compute_ablation_impact(metrics_data: dict):
    """Compute impact of each component."""
    print("\n" + "=" * 100)
    print("🔬 ABLATION ANALYSIS")
    print("=" * 100 + "\n")

    def get_em(variant_name):
        return metrics_data.get(variant_name, {}).get('overall', {}).get('EM', 0.0)

    # Baseline: Dense Only
    dense_em = get_em('Dense Only')
    bm25_em = get_em('BM25 Only')
    hybrid_em = get_em('Hybrid')
    mmr_em = get_em('MMR')
    graph_em = get_em('Static Graph')

    print(f"Baseline (Dense Only):                    {dense_em:>6.1f}%")

    if bm25_em > 0:
        bm25_delta = bm25_em - dense_em
        print(f"BM25 Only (keyword search):              {bm25_em:>6.1f}% (Δ {bm25_delta:+6.1f}%)")

    if hybrid_em > 0:
        hybrid_delta = hybrid_em - dense_em
        print(f"Hybrid (Dense+BM25):                     {hybrid_em:>6.1f}% (Δ {hybrid_delta:+6.1f}%)")

    if mmr_em > 0:
        mmr_delta = mmr_em - hybrid_em if hybrid_em > 0 else 0
        print(f"MMR (+ similarity reranking):            {mmr_em:>6.1f}% (Δ {mmr_delta:+6.1f}%)")

    if graph_em > 0:
        graph_delta = graph_em - mmr_em if mmr_em > 0 else 0
        print(f"Static Graph (+ context expansion):      {graph_em:>6.1f}% (Δ {graph_delta:+6.1f}%)")

    print("\n" + "=" * 100 + "\n")


def save_detailed_comparison(metrics_data: dict, output_file: Path):
    """Save detailed comparison to JSON."""
    comparison = {
        'timestamp': str(Path(output_file).parent),
        'variants': metrics_data,
        'summary': {}
    }

    # Add summary
    for variant, data in metrics_data.items():
        comparison['summary'][variant] = data['overall']

    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Saved detailed comparison: {output_file}")


def main(results_dir: Path):
    """Main analysis function."""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        sys.exit(1)

    # Load metrics
    metrics_data = load_metrics(results_dir)

    if not metrics_data:
        print(f"❌ No metrics_*.json files found in {results_dir}")
        sys.exit(1)

    print(f"\n✅ Loaded metrics for {len(metrics_data)} variants")

    # Print comparison
    print_comparison_table(metrics_data)

    # Ablation analysis
    compute_ablation_impact(metrics_data)

    # Save detailed comparison
    output_file = results_dir / 'detailed_comparison.json'
    save_detailed_comparison(metrics_data, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze variant evaluation results')
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Path to results directory (e.g., variant_results/YYYYMMDD_HHMMSS)'
    )

    args = parser.parse_args()
    main(args.results_dir)
