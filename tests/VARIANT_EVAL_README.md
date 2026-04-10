# Variant Evaluation System

This directory contains standalone evaluation scripts for 5 model variants of your Narrative QA pipeline.

## Overview

Each variant gets its own evaluation script with:
- ✅ Same 200 books (configurable)
- ✅ Same questions and ground truth answers
- ✅ Same evaluation metrics
- ✅ Independent, self-contained execution

## Variants

| File | Variant | Dense | BM25 | MMR | Graph | Description |
|------|---------|-------|------|-----|-------|-------------|
| `eval_dense_only.py` | **Dense Only** | ✓ | ✗ | ✗ | ✗ | FAISS embedding-based retrieval only |
| `eval_bm25_only.py` | **BM25 Only** | ✗ | ✓ | ✗ | ✗ | Keyword-based retrieval only |
| `eval_hybrid.py` | **Hybrid** | ✓ | ✓ | ✗ | ✗ | Dense + BM25 combined scores |
| `eval_mmr.py` | **MMR** | ✓ | ✓ | ✓ | ✗ | Hybrid + MMR reranking for diversity |
| `eval_static_graph.py` | **Static Graph** | ✓ | ✓ | ✓ | ✓ | Full pipeline with graph features |

## Metrics

All variants are evaluated on the same 5 metrics:

- **EM** (Exact Match): % of answers that exactly match ground truth (after normalization)
- **F1** (Token-level): Average F1 score based on token overlap
- **ROUGE-L** (LCS-based): Longest common subsequence-based F1
- **Recall@5**: % of questions where answer chunk is in top-5 retrieved
- **Recall@10**: % of questions where answer chunk is in top-10 retrieved

## Quick Start

### Run a single variant

```bash
# Dense Only
python3 tests/eval_dense_only.py --num-books 200

# BM25 Only
python3 tests/eval_bm25_only.py --num-books 200

# Hybrid
python3 tests/eval_hybrid.py --num-books 200

# MMR
python3 tests/eval_mmr.py --num-books 200

# Static Graph
python3 tests/eval_static_graph.py --num-books 200
```

### Run all variants at once

```bash
python3 tests/run_all_variants.py --num-books 200
```

### Run on smaller dataset (for testing)

```bash
# Quick test on 10 books
python3 tests/eval_dense_only.py --num-books 10

# Or all variants on 10 books
python3 tests/run_all_variants.py --num-books 10
```

### Specify output directory

```bash
python3 tests/eval_dense_only.py --num-books 200 --output-dir ./my_results
```

## Output Format

Each variant produces two files:

### 1. Predictions (`predictions_{variant_name}.json`)

```json
[
  {
    "book_id": "children-of-the-new-forest",
    "question": "Who did the children try to rescue?",
    "ground_truth_1": "Lady Birkenhall",
    "ground_truth_2": "their cousin",
    "prediction": "Lady Birkenhall",
    "em": true,
    "f1": 1.0,
    "rouge_l": 1.0,
    "recall_at_5": true,
    "recall_at_10": true
  }
]
```

### 2. Metrics Summary (`metrics_{variant_name}.json`)

```json
{
  "variant": "Dense Only",
  "config": {
    "use_dense": true,
    "use_bm25": false,
    "use_mmr": false,
    "use_graph": false
  },
  "overall": {
    "EM": 0.425,
    "F1": 0.612,
    "ROUGE-L": 0.598,
    "Recall@5": 0.721,
    "Recall@10": 0.812
  },
  "num_questions": 842,
  "timestamp": "2026-04-10T14:32:45.123456"
}
```

## File Structure

```
tests/
├── eval_dense_only.py          # Variant 1: Dense Only
├── eval_bm25_only.py           # Variant 2: BM25 Only
├── eval_hybrid.py              # Variant 3: Hybrid
├── eval_mmr.py                 # Variant 4: MMR
├── eval_static_graph.py        # Variant 5: Static Graph
├── run_all_variants.py         # 🚀 Master runner (runs all 5)
├── analyze_variants.py         # Post-hoc comparison & analysis
├── compare_llms.py             # Compare with LLMs
├── evaluate_metrics.py         # Full pipeline evaluation
├── VARIANT_EVAL_README.md      # This file
└── variant_results/
    └── YYYYMMDD_HHMMSS/        # Timestamped results
        ├── predictions_dense_only.json
        ├── metrics_dense_only.json
        ├── predictions_bm25_only.json
        ├── metrics_bm25_only.json
        ├── predictions_hybrid.json
        ├── metrics_hybrid.json
        ├── predictions_mmr.json
        ├── metrics_mmr.json
        ├── predictions_static_graph.json
        └── metrics_static_graph.json
```

## Implementation Details

### Metric Functions (Embedded in Each Script)

Each evaluation script includes:

- `normalize_answer()` - Standardize answers for comparison
- `token_f1()` - Compute token-level F1 scores
- `rouge_l()` - Compute ROUGE-L using LCS
- `exact_match()` - Check exact match with either ground truth
- `compute_recall_at_k()` - Check if relevant chunks are in top-k retrieved
- `identify_relevant_chunks()` - Find chunks related to question
- `parse_qaps_csv()` - Load test questions
- `evaluate_*()` - Main evaluation pipeline per variant

### Configuration

Each variant sets the `RETRIEVAL_CONFIG` independently:

```python
# Dense Only
set_config({
    'use_dense': True,
    'use_bm25': False,
    'use_mmr': False,
    'use_graph_expansion': False,
    'use_graph_reasoning': False,
    'use_neighbors': False,
})

# Static Graph (all features enabled)
set_config({
    'use_dense': True,
    'use_bm25': True,
    'use_mmr': True,
    'use_graph_expansion': True,       # All graph features ON
    'use_graph_reasoning': True,
    'use_neighbors': True,
})
```

## Analysis & Comparison

### After Running All Variants

The `run_all_variants.py` script automatically prints a comparison table:

```
Variant           EM       F1       ROUGE-L  R@5      R@10
Dense Only        0.425    0.612    0.598    0.721    0.812
BM25 Only         0.312    0.501    0.489    0.634    0.745
Hybrid            0.521    0.698    0.685    0.812    0.891
MMR               0.534    0.712    0.701    0.825    0.903
Static Graph      0.567    0.745    0.733    0.839    0.915
```

### Post-Hoc Analysis

Use `analyze_variants.py` to load and compare results later:

```bash
python3 tests/analyze_variants.py --results-dir variant_results/YYYYMMDD_HHMMSS
```

This generates:
- Detailed comparison table
- Ablation impact analysis
- `detailed_comparison.json` with full data

## Ablation Analysis

The variant progression tests ablation:

1. **Dense Only** → Baseline embedding-only retrieval
2. **BM25 Only** → Baseline keyword-only retrieval
3. **Hybrid** → Impact of combining dense + BM25
4. **MMR** → Impact of diversity reranking
5. **Static Graph** → Impact of graph context (total)

Compare EM scores to see which components matter:

```python
# Example: Calculate improvements
dense = 0.425
bm25 = 0.312
hybrid = 0.521      # Hybrid improvement over Dense
mmr = 0.534         # MMR improvement over Hybrid
graph = 0.567       # Graph improvement over MMR

print(f"Dense+BM25:  {(hybrid-dense)/dense*100:.1f}% improvement")
print(f"+MMR:        {(mmr-hybrid)/hybrid*100:.1f}% improvement")
print(f"+Graph:      {(graph-mmr)/mmr*100:.1f}% improvement")
```

## Notes

- **First run may take 20-60 minutes** (depending on num_books and hardware)
- Output files are timestamped to prevent overwrites
- Progress printed as: `[book_i/total_books] book_id | ✅`
- All variants use the **same 200 books** (or whatever `--num-books` specified)
- Metrics are computed consistently across all variants
- Each script is completely self-contained (can run independently)
- Errors are logged but don't abort evaluation

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

Make sure you're running from the project root:
```bash
cd /path/to/Hallucinating_llamas_end_submission
python3 tests/eval_dense_only.py
```

### "FAISS index not found"

Ensure processed data exists for your books:
```bash
ls processed_data/children-of-the-new-forest/
# Should have: chunks.jsonl, embeddings.npy, graph.graphml, etc.
```

### Slow performance

- Reduce `--num-books` for testing (e.g., 5-20)
- Check disk I/O (FAISS loads embeddings from disk each time)
- Static Graph variant takes longer due to graph operations

## Integration with Paper

Results can be used for:

| Section | Use |
|---------|-----|
| Ablation Study | Compare EM, F1 across variants |
| Statistical Analysis | Use recall@k to show retrieval quality |
| Error Analysis | Analyze predictions_*.json for failure patterns |
| Comparison with LLMs | Combine with compare_llms.py results |
