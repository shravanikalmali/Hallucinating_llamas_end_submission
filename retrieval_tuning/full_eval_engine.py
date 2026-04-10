import sys, json, numpy as np
from pathlib import Path

# Fix paths
ROOT = Path("/home/shravanikalmali/Desktop/inlp_final/Hallucinating_llamas_end_submission")
sys.path.insert(0, str(ROOT))

from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval.hybrid_search import load_book_index, get_model
import src.retrieval.hybrid_search as hs

DATA_DIR = Path("/home/shravanikalmali/Desktop/inlp_final/drive-download-20260408T131838Z-3-001")
BOOK_ID = "598ad8540bb3851e7347cda52466e566cda0cccf"
TEST_CASES = [
    ("Who frees Conan?", "Tascela"),
    ("What is the name of the person Valeria first encounters?", "Techotl"),
    ("Which brotherhood is Valeria from?", "Red Brotherhood"),
    ("What did Tecuhltli steal?", "bride"),
]

def run_full_baseline(dense_w, bm25_w, ent_w, label):
    print(f"\n{'='*30}\n  FULL PIPELINE BASELINE: {label}\n  Weights: D={dense_w}, B={bm25_w}, E={ent_w}\n{'='*30}")
    
    # Patch weights
    orig_d, orig_b, orig_e = hs.DENSE_WEIGHT, hs.BM25_WEIGHT, hs.ENTITY_WEIGHT
    hs.DENSE_WEIGHT, hs.BM25_WEIGHT, hs.ENTITY_WEIGHT = dense_w, bm25_w, ent_w
    
    for q, gold in TEST_CASES:
        print(f"\nQ: {q}")
        print(f"Gold: {gold}")
        
        # Run entire pipeline
        result = retrieve_and_answer(
            BOOK_ID, q, 
            processed_dir=DATA_DIR,
            verbose=False
        )
        
        pred = result.get("answer", "No Answer Found")
        score = result.get("answer_score", 0.0)
        reasoning = result.get("reasoning", "unknown")
        
        match = "✅" if gold.lower() in pred.lower() or pred.lower() in gold.lower() else "❌"
        print(f"{match} PRED: {pred}")
        print(f"  Confidence: {score:.4f} | Strategy: {reasoning}")
    
    # Restore
    hs.DENSE_WEIGHT, hs.BM25_WEIGHT, hs.ENTITY_WEIGHT = orig_d, orig_b, orig_e
