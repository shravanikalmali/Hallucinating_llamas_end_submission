"""
NarCo-style Coherence Edge Builder.

Implements the two-stage LLM prompting from the NarCo paper:
1. Question Generation (retrospective questions from chunk j to chunk i).
2. Back Verification (filtering questions not grounded in evidence).
"""

import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils import load_chunks, save_json, load_json
from src.retrieval.hybrid_llm import _deepseek_chat

# ── NarCo Prompts ─────────────────────────────────────────────────────────────

PROMPT_STAGE1_T1 = (
    "You are an expert on reading and analyzing a wide variety of books. "
    "Given the following two snippets snippet_a and snippet_b from a book, "
    "where snippet_a happens before snippet_b, find concrete parts in both snippets "
    "that reflect this temporal relation, such that certain parts in snippet_a contribute "
    "as the preceding background or cause for specific events or situations in snippet_b.\n\n"
    "[snippet_a]\n{snippet_a}\n\n"
    "[snippet_b]\n{snippet_b}\n\n"
    "Please try your best to provide a brief markdown list of each important point that "
    "contains those specific parts from both snippets and briefly explains how one serves "
    "as the background or cause for the other (no more than 3 points).\n"
    "Note: only list evident and important points. If none found, say 'None'."
)

PROMPT_STAGE1_T2 = (
    "Please convert each of your listed points into the form of a question, "
    "such that each question asks about the cause or background (rather than outcome) "
    "of specific events in snippet_b, which can be answered by snippet_a. "
    "The question should ask about things in snippet_b that cannot be answered by snippet_b alone.\n"
    "Return each question starting with 'Q:'."
)

PROMPT_STAGE2 = (
    "Given the following snippet and a related question, determine if the snippet "
    "contains the answer. \n\n"
    "[snippet]\n{snippet}\n\n"
    "[question]\n{question}\n\n"
    "Reason briefly, then if answerable print [ANSWERABLE] followed by the specific "
    "supporting sentences. If not, print [UNANSWERABLE]."
)

# ── NarCo Edge Builder ────────────────────────────────────────────────────────

class NarCoBuilder:
    def __init__(self, window_size=4, max_workers=4):
        self.window_size = window_size
        self.max_workers = max_workers

    def generate_questions(self, snippet_a: str, snippet_b: str) -> list[str]:
        """Stage 1: Two-turn question generation."""
        # Turn 1
        messages = [{"role": "user", "content": PROMPT_STAGE1_T1.format(snippet_a=snippet_a, snippet_b=snippet_b)}]
        response1 = _deepseek_chat(messages)
        if not response1 or "None" in response1:
            return []

        # Turn 2
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": PROMPT_STAGE1_T2})
        response2 = _deepseek_chat(messages)
        if not response2:
            return []

        questions = [line[2:].strip() for line in response2.split("\n") if line.startswith("Q:")]
        return questions

    def verify_question(self, question: str, snippet_a: str, snippet_b: str) -> bool:
        """Stage 2: Back-verification."""
        # Check if snippet_b alone can answer it (should be False)
        msg_b = [{"role": "user", "content": PROMPT_STAGE2.format(snippet=snippet_b, question=question)}]
        resp_b = _deepseek_chat(msg_b)
        if resp_b and "[ANSWERABLE]" in resp_b:
            return False # Invalid: self-answerable

        # Check if snippet_a can answer it (should be True)
        msg_a = [{"role": "user", "content": PROMPT_STAGE2.format(snippet=snippet_a, question=question)}]
        resp_a = _deepseek_chat(msg_a)
        return resp_a and "[ANSWERABLE]" in resp_a

    def process_pair(self, i: int, j: int, chunks: list[dict]) -> dict:
        """Process a single pair of chunks (i < j)."""
        snippet_a = chunks[i]["text"]
        snippet_b = chunks[j]["text"]
        
        questions = self.generate_questions(snippet_a, snippet_b)
        valid_questions = []
        for q in questions:
            if self.verify_question(q, snippet_a, snippet_b):
                valid_questions.append(q)
        
        if valid_questions:
            return {
                "source_idx": i,
                "target_idx": j,
                "questions": valid_questions,
                "relation": "coherence_dependency"
            }
        return None

    def build_for_book(self, book_dir: Path):
        """Build NarCo edges for all chunks in a book within a sliding window."""
        chunks_file = book_dir / "chunks.jsonl"
        output_file = book_dir / "narco_edges.json"
        
        if not chunks_file.exists():
            return
            
        chunks = load_chunks(chunks_file)
        n = len(chunks)
        pairs_to_process = []
        
        for j in range(1, n):
            for i in range(max(0, j - self.window_size), j):
                pairs_to_process.append((i, j))
        
        print(f"  Building NarCo edges for {n} chunks ({len(pairs_to_process)} pairs)...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_pair, i, j, chunks) for i, j in pairs_to_process]
            for future in tqdm(futures, total=len(pairs_to_process)):
                res = future.result()
                if res:
                    results.append(res)
        
        save_json(output_file, results)
        print(f"    Saved {len(results)} NarCo edges to {output_file.name}")

def run(book_id: str = None, processed_dir: str = None, window_size: int = 4, max_workers: int = 4):
    from src.config import find_data_directory
    processed = find_data_directory(processed_dir)
    
    builder = NarCoBuilder(window_size=window_size, max_workers=max_workers)
    
    if book_id:
        book_dirs = [processed / book_id]
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
        
    for book_dir in book_dirs:
        print(f"\n── {book_dir.name}")
        builder.build_for_book(book_dir)
