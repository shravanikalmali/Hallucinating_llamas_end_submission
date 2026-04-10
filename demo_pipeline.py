
import json
import sys
from pathlib import Path
from src.retrieval.pipeline import retrieve_and_answer
from src.retrieval_config import RETRIEVAL_CONFIG, config_summary

def run_demo():
    # 1. Configuration
    book_id = "c8e25c068b7c8a00ba00096e73ce7ea893c69aba"
    question = "Why did Phil refuse to testify against the pirates?"
    
    print("="*80)
    print("🔥 END-TO-END NARRATIVE QA PIPELINE DEMO 🔥")
    print("="*80)
    print(f"📖 Book ID  : {book_id}")
    print(f"❓ Question : {question}")
    print(f"⚙️  Config   : {config_summary()}")
    print("-" * 80)

    # 2. Run Pipeline
    # We use verbose=True to get some internal logs, but our result dict has the stages
    result = retrieve_and_answer(book_id, question, verbose=True)

    print("\n" + "="*80)
    print("📊 STAGE-BY-STAGE BREAKDOWN")
    print("="*80)

    # Stage 1: Question Anchoring
    print("\nSTAGED 1: Analysis & Anchoring")
    print(f"  └─ Question Type: {result.get('question_type')} ({result.get('question_subtype')})")
    print(f"  └─ Graph Anchors: {result['retrieval_log'].get('query_anchors', [])}")

    # Stage 2: Initial Hybrid Retrieval
    print("\nSTAGE 2: Initial Hybrid Retrieval (FAISS + BM25)")
    print(f"  └─ Total chunks found: {result.get('num_initial')}")
    print(f"  └─ Top 10 Chunk IDs  : {result['stages']['initial_retrieval_ids']}")

    # Stage 3: MMR Re-ranking
    print("\nSTAGE 3: Diversity Re-ranking (MMR)")
    print(f"  └─ Refined Chunk IDs : {result['stages']['mmr_ranked_ids']}")

    # Stage 4: Narrative Graph Expansion
    print("\nSTAGE 4: Graph-Neighbor & Causal Expansion")
    print(f"  └─ Final context size: {result.get('num_final')} chunks")
    print(f"  └─ Top 10 Final IDs  : {result['stages']['final_expanded_ids']}")

    # Stage 5: Final LLM Answer Generation
    print("\nSTAGE 5: LLM Generative Answering")
    print(f"  └─ Reasoning Mode    : {result.get('reasoning')}")
    print(f"  └─ Source Context    : Chunk {result.get('source_chunk_id')} (Segment {result.get('source_segment')})")
    
    print("\n" + "✨" * 40)
    print(f"FINAL ANSWER: {result['answer']}")
    print("✨" * 40 + "\n")

    # Save to JSON for records
    with open("pipeline_execution_log.json", "w") as f:
        json.dump(result, f, indent=4)
    print("✅ Full execution log saved to pipeline_execution_log.json")

if __name__ == "__main__":
    run_demo()
