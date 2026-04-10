"""
# Answer Quality Metrics - Implementation Guide

## What was implemented

### Point 13: Answer Quality Metrics ✅

Created comprehensive metrics for QA evaluation:

**File: `src/evaluation.py`**
- **Exact Match (EM)**: Binary score (1.0 if exact match, 0.0 otherwise)
- **F1 Score**: Token-level F1 based on precision and recall of tokens
- **BLEU Score**: N-gram based metric adapted for short answers
- **ROUGE-L**: Longest Common Subsequence (LCS) based metric

All metrics use normalized answers (lowercase, remove punctuation, articles).

**Functions:**
- `normalize_answer()` - Normalize for fair comparison
- `exact_match()` - EM score
- `f1_score()` - Token-level F1
- `bleu_score()` - BLEU with n-gram precision
- `rouge_l()` - LCS-based ROUGE-L
- `compute_answer_metrics()` - Compute all 4 metrics for one QA pair
- `micro_average_metrics()` - Average metrics across multiple questions
- `format_metrics()` - Pretty-print metrics

### Point 14: Structured Answer Generation with Confidence ✅

**Already Implemented** in previous edits to `src/retrieval/reasoning.py`:
- Answer result includes: `answer_score` (combined confidence)
- Stores: `cross_encoder_score` (model prediction quality)
- Stores: `graph_confidence` (reasoning structure quality)
- Formula: `combined = 0.6 * cross_encoder_score + 0.4 * graph_confidence`

**Answer Structure:**
```python
{
    "answer": "extracted sentence",
    "answer_score": 0.75,  # Combined confidence 0.6*model + 0.4*graph
    "cross_encoder_score": 0.8,  # Raw model score
    "graph_confidence": 0.65,  # Reasoning confidence
    "source_chunk_id": 42,
    "source_segment": 5,
    "context": "full chunk text",
    "reasoning": "graph_guided" or "graph" or "fallback",
    "graph_target": "entity",
    "graph_nodes": 15,
    "graph_edges": 28,
}
```

### Integration: `src/evaluation_integration.py`

Provides evaluation harness for using metrics:

**Functions:**
- `evaluate_qa_result()` - Evaluate single QA with metrics + confidence
- `batch_evaluate()` - Evaluate multiple QA results, compute aggregates
- `print_evaluation_report()` - Format results for human readability

**Failure Type Classification:**
- `retrieval`: Answer chunk ranked > 5
- `extraction`: Retrieved correct chunk but extracted wrong sentence (F1 < 0.3)
- `reasoning`: Other failures
- `none`: No failure (EM = 1.0)

---

## Usage Examples

### Example 1: Single QA Evaluation

```python
from src.evaluation_integration import evaluate_qa_result

result = evaluate_qa_result(
    question="Who directed Inception?",
    predicted_answer="Christopher Nolan directed Inception",
    gold_answer="Christopher Nolan",
    confidence=0.78,
    retrieval_rank=1,
    metadata={"qtype": "who", "graph_nodes": 25}
)

print(result)
# {
#     "em": 0.0,  # Partial match
#     "f1": 0.67,  # 2 out of 3 tokens match
#     "bleu": 0.52,
#     "rouge_l": 0.67,
#     "failure_type": "extraction"
# }
```

### Example 2: Batch Evaluation

```python
from src.evaluation_integration import batch_evaluate, print_evaluation_report

qa_results = [
    {
        "question": "Who directed Inception?",
        "predicted_answer": "Christopher Nolan",
        "gold_answer": "Christopher Nolan",
        "confidence": 0.82,
        "retrieval_rank": 1,
        "metadata": {"qtype": "who"}
    },
    {
        "question": "Where was Inception filmed?",
        "predicted_answer": "United Kingdom",
        "gold_answer": "Various locations including UK",
        "confidence": 0.65,
        "retrieval_rank": 3,
        "metadata": {"qtype": "where"}
    },
]

report = batch_evaluate(qa_results)
print_evaluation_report(report)

# Output:
# Total Questions: 2
# Aggregate Metrics: EM: 0.5000 | F1: 0.7500 | BLEU: 0.6500 | ROUGE-L: 0.7200
# Failure Analysis: none: 1 (50.0%), extraction: 1 (50.0%)
# Confidence Statistics: Mean: 0.74, Min: 0.65, Max: 0.82
# Per-Question-Type Analysis:
#   who: 1 questions, EM: 1.0000, F1: 1.0000
#   where: 1 questions, EM: 0.0000, F1: 0.5000
```

### Example 3: Direct Metric Computation

```python
from src.evaluation import compute_answer_metrics, format_metrics

predicted = "The Eiffel Tower is in Paris, France"
gold = "Paris"

metrics = compute_answer_metrics(predicted, gold)
print(format_metrics(metrics))
# Output: EM: 0.0000 | F1: 0.3333 | BLEU: 0.2500 | ROUGE-L: 0.3333
```

---

## Integration with Retrieval Pipeline

The answer result structure already includes confidence scoring:

```python
# In src/retrieval/reasoning.py (already implemented)
answer_result = {
    ...
    "answer_score": combined_confidence,  # 0.6*cross_encoder + 0.4*graph
    "cross_encoder_score": cross_encoder_score,
    "graph_confidence": graph_confidence,
}

# Now evaluable with:
from src.evaluation_integration import evaluate_qa_result

result = evaluate_qa_result(
    question=question,
    predicted_answer=answer_result["answer"],
    gold_answer=gold_answer,
    confidence=answer_result["answer_score"],
    metadata={"qtype": qtype, "graph_nodes": answer_result.get("graph_nodes", 0)}
)
```

---

## Metrics Interpretation

| Metric | Interpretation | Range | Ideal |
|--------|---|-------|-------|
| EM | Exact string match after normalization | [0.0, 1.0] | 1.0 |
| F1 | Token-level precision/recall balance | [0.0, 1.0] | 1.0 |
| BLEU | N-gram precision with brevity penalty | [0.0, 1.0] | 1.0 |
| ROUGE-L | LCS-based overlap | [0.0, 1.0] | 1.0 |

**Note:** All metrics are normalized (lowercase, punctuation removed, articles stripped).

---

## For Ablation Studies

Store in evaluation:
```python
per_config_results = {
    "config_1": batch_evaluate(results_1),
    "config_2": batch_evaluate(results_2),
    "config_3": batch_evaluate(results_3),
}

for config, report in per_config_results.items():
    print(f"{config}: {report['formatted_metrics']}")
```

This enables systematic comparison of retrieval + reasoning configurations.
"""
