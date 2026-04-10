import sys
from pathlib import Path

# Fix paths
ROOT = Path("/home/shravanikalmali/Desktop/inlp_final/Hallucinating_llamas_end_submission")
sys.path.insert(0, str(ROOT))

from retrieval_tuning.full_eval_engine import run_full_baseline

configs = [
    (1.0, 0.0, 0.0, "DENSE ONLY"),
    (0.0, 1.0, 0.0, "BM25 ONLY"),
    (0.0, 0.0, 1.0, "ENTITY ONLY"),
    (0.5, 0.5, 0.0, "DENSE + BM25"),
    (0.5, 0.0, 0.5, "DENSE + ENTITY"),
    (0.0, 0.5, 0.5, "BM25 + ENTITY"),
    (0.4, 0.4, 0.2, "ORIGINAL MIX")
]

print("Starting FULL PIPELINE Comparative Evaluation...")
for d, b, e, label in configs:
    run_full_baseline(d, b, e, label)
print("\nFull Pipeline Comparison Complete.")
