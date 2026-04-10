import sys, numpy as np
from pathlib import Path

# Fix paths
ROOT = Path("/home/shravanikalmali/Desktop/inlp_final/Hallucinating_llamas_end_submission")
sys.path.insert(0, str(ROOT))

from retrieval_tuning.eval_engine import run_eval

configs = [
    (1.0, 0.0, 0.0, "DENSE ONLY"),
    (0.0, 1.0, 0.0, "BM25 ONLY"),
    (0.0, 0.0, 1.0, "ENTITY ONLY"),
    (0.5, 0.5, 0.0, "DENSE + BM25"),
    (0.5, 0.0, 0.5, "DENSE + ENTITY"),
    (0.0, 0.5, 0.5, "BM25 + ENTITY"),
    (0.4, 0.4, 0.2, "ALL THREE (Original)")
]

print("Starting Comparative Evaluation of Stage 1 (Retrieval)...")
for d, b, e, label in configs:
    run_eval(d, b, e, label)
print("\nEvaluation Complete.")
