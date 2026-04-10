"""
Verify FAISS indexes match their chunks for all books.
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from src.config import VECTOR_DIM, PROCESSED_DIR


def verify_book(book_dir: Path) -> dict:
    book_id = book_dir.name
    faiss_path = book_dir / f"{book_id}.faiss"
    pkl_path = book_dir / f"{book_id}_chunks.pkl"
    emb_path = book_dir / "embeddings.npy"

    results = {}
    results["files_exist"] = faiss_path.exists() and pkl_path.exists()
    if not results["files_exist"]:
        print(f"  ❌ {book_id}: Missing files")
        return results

    index = faiss.read_index(str(faiss_path))
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    results["count_match"] = index.ntotal == len(chunks)
    results["dimension"] = index.d == VECTOR_DIM
    results["metadata_ok"] = (
        [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
        and len(set(c["book_id"] for c in chunks)) == 1
    )

    # Search sanity
    query = faiss.rev_swig_ptr(index.get_xb(), VECTOR_DIM).reshape(-1, VECTOR_DIM)[:1].copy()
    scores, indices = index.search(query, k=1)
    results["search_sanity"] = int(indices[0][0]) == 0 and float(scores[0][0]) > 0.999

    if emb_path.exists():
        emb = np.load(emb_path).astype(np.float32)
        s2, i2 = index.search(emb[0:1], k=1)
        results["cross_check"] = int(i2[0][0]) == 0 and float(s2[0][0]) > 0.999

    all_ok = all(v for v in results.values() if v is not None)
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} {book_id}")
    return results


def run(book_id: str = None, processed_dir: str = None):
    """Entry point."""
    pd = Path(processed_dir) if processed_dir else PROCESSED_DIR
    if book_id:
        book_dirs = [pd / book_id]
    else:
        book_dirs = sorted([d for d in pd.iterdir() if d.is_dir()])
    print(f"Verifying FAISS for {len(book_dirs)} books…\n")
    for bd in tqdm(book_dirs, desc="Verifying"):
        verify_book(bd)
