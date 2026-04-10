"""
FAISS Index Builder — Phase 0.5

Builds per-book IndexFlatIP (exact cosine search on unit-normalised 768-d vectors).
"""

import json
import pickle
import argparse
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from src.config import VECTOR_DIM, PROCESSED_DIR
from src.utils import load_chunks


def load_embeddings(emb_file: Path) -> np.ndarray:
    """Load embeddings matrix and ensure float32."""
    return np.load(emb_file).astype(np.float32)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a flat inner-product index (cosine search for unit vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_book_index(book_id: str, index: faiss.IndexFlatIP,
                    chunks: list[dict], book_dir: Path) -> None:
    """Save FAISS index and chunk metadata to disk."""
    faiss_path = book_dir / f"{book_id}.faiss"
    pkl_path = book_dir / f"{book_id}_chunks.pkl"

    faiss.write_index(index, str(faiss_path))
    with open(pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"    Saved index    → {faiss_path}")
    print(f"    Saved metadata → {pkl_path}")


def process_book(book_dir: Path) -> bool:
    """Build FAISS index for a single book. Returns True on success."""
    book_id = book_dir.name
    emb_file = book_dir / "embeddings.npy"
    chunks_file = book_dir / "chunks.jsonl"

    if not emb_file.exists():
        print(f"  ⚠️  Skipping {book_id} — embeddings.npy not found")
        return False
    if not chunks_file.exists():
        print(f"  ⚠️  Skipping {book_id} — chunks.jsonl not found")
        return False

    print(f"\n── {book_id} {'─' * (55 - len(book_id))}")

    embeddings = load_embeddings(emb_file)
    chunks = load_chunks(chunks_file)

    print(f"    Embeddings : {embeddings.shape}")
    print(f"    Chunks     : {len(chunks)}")

    if embeddings.shape[0] != len(chunks):
        print(f"  ❌ Mismatch — {embeddings.shape[0]} embeddings vs {len(chunks)} chunks.")
        return False

    index = build_index(embeddings)
    print(f"    Index total: {index.ntotal} vectors")

    save_book_index(book_id, index, chunks, book_dir)
    return True


def run(book_id: str = None, processed_dir: str = None):
    """Main FAISS building entry point."""
    processed = Path(processed_dir) if processed_dir else PROCESSED_DIR

    print("=" * 60)
    print("  Build FAISS Index")
    print("=" * 60)
    print(f"  Vector dim    : {VECTOR_DIM}")
    print(f"  Index type    : IndexFlatIP (exact cosine search)")
    print(f"  Source dir    : {processed}")

    if book_id:
        book_dirs = [processed / book_id]
        if not book_dirs[0].exists():
            print(f"\n❌ Book folder not found: {book_dirs[0]}")
            return
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
        print(f"\n  Found {len(book_dirs)} book directories")

    success, failed = 0, 0
    for book_dir in tqdm(book_dirs, desc="Building indexes"):
        if process_book(book_dir):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}\n  Done! ✅ Indexed: {success} books")
    if failed:
        print(f"  ❌ Skipped: {failed} books")
    print("=" * 60)
