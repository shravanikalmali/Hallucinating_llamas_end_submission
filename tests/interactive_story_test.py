import os
import sys
import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.getcwd())

from src.retrieval.hybrid_search import get_model, detect_question_type_info, embed_question
from src.retrieval.answer import _get_query_entities, extract_answer, _split_into_sentences
from src.retrieval.llm_answer import llm_extract_answer

STORY_PATH = "tests/test_children_story.txt"

def load_and_chunk_story(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        text = f.read()
    
    # Split into sections or paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append({
            "chunk_id": i,
            "text": p,
            "metadata": {"title": f"Paragraph {i}"}
        })
    return chunks

def interactive_test():
    print("=" * 80)
    print("INTERACTIVE STORY QA - The Hidden Treasure")
    print("=" * 80)
    
    model = get_model()
    chunks = load_and_chunk_story(STORY_PATH)
    
    # Pre-embed chunks
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    
    while True:
        query = input("\n[QUESTION] (or 'exit' to quit): ").strip()
        if not query or query.lower() == 'exit':
            break
            
        print("-" * 40)
        # STAGE 1: INTENT & ENTITIES
        qtype, qsubtype, t_cue, t_anchor = detect_question_type_info(query)
        q_entities = _get_query_entities(query)
        
        print(f"[STAGE 1] INTENT DETECTION")
        print(f"    Type    : {qtype}")
        print(f"    Subtype : {qsubtype}")
        print(f"    Entities: {list(q_entities)}")
        
        # STAGE 2: SEMANTIC RETRIEVAL
        q_emb = embed_question(query, model)
        if len(q_emb.shape) == 1: q_emb = q_emb.reshape(1, -1)
        
        similarities = np.dot(q_emb, chunk_embs.T)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        
        top_chunks = []
        print(f"\n[STAGE 2] SEMANTIC RETRIEVAL (Top 5 Chunks)")
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = chunks[idx].copy()
            chunk["score"] = score
            top_chunks.append(chunk)
            snippet = (chunk["text"][:120] + "...") if len(chunk["text"]) > 120 else chunk["text"]
            print(f"    - ID {idx} (Score: {score:.4f}): \"{snippet}\"")
            
        # STAGE 3: SYMBOLIC EXTRACTION (using verbose mode)
        print(f"\n[STAGE 3] SYMBOLIC EXTRACTION")
        # We need to pass verbose=True to extract_answer to see candidates
        ans_res = extract_answer(
            query, top_chunks, model=model, qtype=qtype, qsubtype=qsubtype, verbose=True
        )
        
        # STAGE 4: RESULT & FALLBACK
        print(f"\n[STAGE 4] FINAL PIPELINE DECISION")
        if ans_res["validated"]:
            print(f"    RESULT: {ans_res['answer']}")
            print(f"    REASON: {ans_res['reasoning']} (Score: {ans_res.get('score', 0):.2f})")
        else:
            reason = ans_res["reasoning"]
            print(f"    Symbolic Failed: {reason}")
            if reason != "not_answerable":
                print("    [FALLBACK] Calling Constrained LLM...")
                llm_res = llm_extract_answer(query, top_chunks)
                print(f"    LLM RESULT: {llm_res['answer']}")
                print(f"    LLM REASON: {llm_res['reasoning']}")
            else:
                print("    [GATE] Skipping LLM: Evidence not answerable.")
                print("    RESULT: NOT FOUND")

if __name__ == "__main__":
    interactive_test()
