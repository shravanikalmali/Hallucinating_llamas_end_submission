
import json
import os
import csv
import sys
import re
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.retrieval.pipeline import retrieve_and_answer
from src.config import find_data_directory

def run_example_cases(book_id, questions_csv):
    base_results_dir = Path("results") / book_id
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve processed directory
    processed_dir = find_data_directory() 
    book_dir = processed_dir / book_id
    
    # Load all chunks for the book to provide real text in mmr_chunks.txt
    chunks_file = book_dir / "chunks.jsonl"
    all_chunks = {}
    if chunks_file.exists():
        with open(chunks_file, "r") as f:
            for line in f:
                c = json.loads(line)
                all_chunks[c["chunk_id"]] = c

    questions = []
    with open(questions_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == book_id:
                questions.append({
                    "question": row[2],
                    "gold": row[3]
                })

    for i, q_item in enumerate(questions, 1):
        q_dir = base_results_dir / f"Question_{i}"
        q_dir.mkdir(parents=True, exist_ok=True)
        
        question = q_item["question"]
        gold = q_item["gold"]
        
        print(f"Processing Question {i}: {question}")
        
        # Capture verbose output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = retrieve_and_answer(
                book_id=book_id,
                question=question,
                verbose=True,
                gold_answer=gold
            )
        verbose_out = f.getvalue()
        
        # Save Question Info
        with open(q_dir / "summary.txt", "w") as sf:
            sf.write(f"Question: {question}\n")
            sf.write(f"Gold Answer: {gold}\n")
            sf.write(f"Predicted Answer: {result.get('answer', 'NOT FOUND')}\n")
            sf.write(f"Confidence: {result.get('answer_score', 0):.4f}\n")
            sf.write(f"Reasoning Path: {result.get('reasoning', 'unknown')}\n")

        # Save Full Verbose Pipeline Trace
        with open(q_dir / "pipeline_trace.txt", "w") as tf:
            tf.write(verbose_out)

        # Parse verbose_out for chunk IDs
        # Look for pattern: "- ID 493 (Score: 0.6134)"
        chunk_ids = re.findall(r"- ID (\d+) \(Score: [\d.]+\)", verbose_out)
        
        # MMR Chunks details with text
        with open(q_dir / "mmr_chunks.txt", "w") as mf:
            mf.write(f"Top MMR Chunks for Query: {question}\n")
            mf.write("="*60 + "\n")
            for cid_str in chunk_ids:
                cid = int(cid_str)
                chunk = all_chunks.get(cid, {})
                text = chunk.get("text", "Text not found in chunks.jsonl")
                mf.write(f"CHUNK ID: {cid}\n")
                mf.write(f"SEGMENT : {chunk.get('segment_id', 'N/A')}\n")
                mf.write(f"CONTENT :\n{text}\n")
                mf.write("-" * 40 + "\n")

        # Create a basic Mermaid graph from found entities and the source chunk
        ents = re.findall(r"ENTITIES: \[(.*?)\]", verbose_out)
        entity_list = []
        if ents:
            # simple parse from string like "'Phil', 'Devon'"
            entity_list = [e.strip().strip("'").strip('"') for e in ents[0].split(",")]
        
        ans = result.get('answer', 'NOT FOUND')
        
        graph_data = "graph TD\n"
        graph_data += f'    Q["{question}"]\n'
        for ent in entity_list:
            graph_data += f'    Q --> {ent.replace(" ", "_")}["{ent}"]\n'
        if ans and ans != "NOT FOUND":
            graph_data += f'    {ans.replace(" ", "_")}["{ans}"]\n'
            for ent in entity_list:
                graph_data += f'    {ent.replace(" ", "_")} -- "relation" --> {ans.replace(" ", "_")}\n'
        
        with open(q_dir / "query_graph.mermaid", "w") as gf:
            gf.write(graph_data)

    print(f"Data generation complete. Files saved in {base_results_dir}")

if __name__ == "__main__":
    book_id = "c8e25c068b7c8a00ba00096e73ce7ea893c69aba"
    questions_csv = "/tmp/target_questions.csv"
    run_example_cases(book_id, questions_csv)
