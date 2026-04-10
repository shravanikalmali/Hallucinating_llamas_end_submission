import sys, numpy as np
from pathlib import Path

# Fix paths
ROOT = Path("/home/shravanikalmali/Desktop/inlp_final/Hallucinating_llamas_end_submission")
sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_search import (
    load_book_index, embed_question, get_model, hybrid_retrieval
)

DATA_DIR = Path("/home/shravanikalmali/Desktop/inlp_final/drive-download-20260408T131838Z-3-001")
BOOK_ID = "598ad8540bb3851e7347cda52466e566cda0cccf"

TEST_CASES = [
    ("Who frees Conan?", "Tascela"),
    ("What is the name of the person Valeria first encounters?", "Techotl"),
    ("Which brotherhood is Valeria from?", "Red Brotherhood"),
    ("What type of fruit is growing nearby?", "poisons"),
    ("What did Tecuhltli steal?", "bride"),
]

def run_eval(dense_w, bm25_w, ent_w, label):
    print(f"\n{'='*20} {label} {'='*20}")
    print(f"Weights: Dense={dense_w}, BM25={bm25_w}, Entity={ent_w}")
    
    index, chunks = load_book_index(BOOK_ID, DATA_DIR)
    model = get_model()
    
    # We patch the variables directly in the hybrid_search module 
    # because it uses 'from src.config import ...' which creates local copies.
    import src.retrieval.hybrid_search as hs
    orig_d, orig_b, orig_e = hs.DENSE_WEIGHT, hs.BM25_WEIGHT, hs.ENTITY_WEIGHT
    hs.DENSE_WEIGHT, hs.BM25_WEIGHT, hs.ENTITY_WEIGHT = dense_w, bm25_w, ent_w
    
    total_ranks = []
    found_count = 0
    
    for q, gold in TEST_CASES:
        query_vec = embed_question(q, model)
        results = hybrid_retrieval(index, chunks, query_vec, q, top_k=25, book_dir=DATA_DIR/BOOK_ID)
        
        found = False
        rank = -1
        for i, res in enumerate(results):
            if gold.lower() in res["text"].lower():
                found = True
                rank = i + 1
                break
        
        if found:
            print(f"  ✅ [Rank {rank:2}] Q: {q}")
            total_ranks.append(rank)
            found_count += 1
        else:
            print(f"  ❌ [NOT FOUND] Q: {q}")
            
    print(f"\nSUMMARY: {found_count}/{len(TEST_CASES)} Found | Avg Rank (of found): {np.mean(total_ranks) if total_ranks else 'N/A':.2f}")
    
    # Restore weights
    hs.DENSE_WEIGHT, hs.BM25_WEIGHT, hs.ENTITY_WEIGHT = orig_d, orig_b, orig_e
