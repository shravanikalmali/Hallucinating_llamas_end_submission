"""
Embedding Splitter

Splits the global embeddings.npy into per-book files under processed_data/<book_id>/.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.config import EMBEDDINGS_DIR, PROCESSED_DIR


def split_embeddings(embeddings_dir: Path = EMBEDDINGS_DIR,
                     processed_dir: Path = PROCESSED_DIR) -> None:
    """Split the monolithic embeddings.npy into per-book files."""
    emb_file = embeddings_dir / "embeddings.npy"
    meta_file = embeddings_dir / "metadata.json"

    print("Loading embeddings …")
    embeddings = np.load(emb_file)
    print(f"  shape: {embeddings.shape}")

    print("Loading metadata …")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata: list[dict] = json.load(f)
    print(f"  entries: {len(metadata)}")

    assert len(metadata) == embeddings.shape[0], (
        f"Mismatch: {len(metadata)} metadata rows vs {embeddings.shape[0]} embedding rows"
    )

    # Group row indices by book_id
    book_indices: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(metadata):
        book_indices[entry["book_id"]].append(idx)

    print(f"Unique books: {len(book_indices)}")

    skipped = 0
    for book_id, indices in tqdm(book_indices.items(), desc="Splitting"):
        book_dir = processed_dir / book_id
        if not book_dir.exists():
            skipped += 1
            continue

        book_emb = embeddings[indices]
        np.save(book_dir / "embeddings.npy", book_emb)

        book_meta = [metadata[i] for i in indices]
        with open(book_dir / "metadata_emb.json", "w", encoding="utf-8") as f:
            json.dump(book_meta, f, indent=2)

    written = len(book_indices) - skipped
    print(f"Done – wrote embeddings into {written} book directories")
    if skipped:
        print(f"  (skipped {skipped} books with no matching folder)")


def run(processed_dir: str = None):
    """Entry point for embedding splitting."""
    processed = Path(processed_dir) if processed_dir else PROCESSED_DIR
    split_embeddings(processed_dir=processed)
