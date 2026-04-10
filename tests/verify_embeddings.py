"""
Verify that embeddings.npy matches chunks.jsonl for each book.

Checks: count match, dimension (768-d), unit norms, sequential similarity,
chunk ID ordering, word count sanity, optional re-embed verification.
"""

import json
import argparse
import numpy as np
from pathlib import Path

from src.config import MODEL_NAME, VECTOR_DIM

try:
    from sentence_transformers import SentenceTransformer
    REEMBED_AVAILABLE = True
except ImportError:
    REEMBED_AVAILABLE = False


EXPECTED_DIM = VECTOR_DIM  # 768
SIMILARITY_THRESHOLD = 0.999


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_book(book_folder: Path, reembed: bool = True) -> dict:
    chunks_file = book_folder / "chunks.jsonl"
    emb_file = book_folder / "embeddings.npy"

    if not chunks_file.exists() or not emb_file.exists():
        return {"error": f"Missing files in {book_folder}"}

    chunks = load_chunks(chunks_file)
    embeddings = np.load(emb_file)
    results = {}

    # Check 1: Count
    results["count"] = len(chunks) == embeddings.shape[0]
    # Check 2: Dimension
    results["dimension"] = embeddings.shape[1] == EXPECTED_DIM
    # Check 3: Norms
    norms = np.linalg.norm(embeddings, axis=1)
    results["norms"] = abs(norms.mean() - 1.0) < 0.001
    # Check 4: Chunk IDs
    ids = [c["chunk_id"] for c in chunks]
    results["chunk_ids"] = ids == list(range(len(chunks)))
    # Check 5: Word counts
    results["word_counts"] = all(c["word_count"] >= 1 for c in chunks)
    # Check 6: Sequential similarity
    if len(embeddings) >= 3:
        sims = [cosine_sim(embeddings[i], embeddings[i + 1])
                for i in range(min(20, len(embeddings) - 1))]
        results["adj_sim"] = np.mean(sims) > 0.3
    else:
        results["adj_sim"] = True

    if reembed and REEMBED_AVAILABLE:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        indices = np.linspace(0, len(chunks) - 1, min(20, len(chunks)), dtype=int)
        texts = ["search_document: " + chunks[i]["text"] for i in indices]
        fresh = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        mismatches = sum(1 for j, idx in enumerate(indices)
                        if cosine_sim(fresh[j], embeddings[idx]) < SIMILARITY_THRESHOLD)
        results["reembed"] = mismatches == 0

    book_id = book_folder.name
    all_passed = all(v for v in results.values() if isinstance(v, bool))
    icon = "✅" if all_passed else "❌"
    print(f"  {icon} {book_id}: {results}")
    return results


def run(processed_dir: str = None):
    """Entry point."""
    from src.config import PROCESSED_DIR
    pd = Path(processed_dir) if processed_dir else PROCESSED_DIR
    book_dirs = sorted([d for d in pd.iterdir() if d.is_dir()])
    print(f"Verifying {len(book_dirs)} books (expecting {EXPECTED_DIM}-d embeddings)…\n")
    for bd in book_dirs:
        verify_book(bd, reembed=False)
