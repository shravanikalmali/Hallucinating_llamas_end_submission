"""
Embedding Pipeline — Phase 0.5

Encodes all chunks with nomic-ai/nomic-embed-text-v1.5 (768-d) and saves
a global embeddings.npy + metadata.json.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME, EMBED_BATCH_SIZE, PROCESSED_DIR, EMBEDDINGS_DIR


def load_all_chunks(processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    """Load all chunks from every book, enriched with book-level metadata."""
    metadata_file = processed_dir / "metadata.jsonl"
    book_meta: dict[str, dict] = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            book_meta[entry["story_id"]] = entry

    all_chunks: list[dict] = []
    book_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()], key=lambda p: p.name)

    for book_dir in tqdm(book_dirs, desc="Loading chunks"):
        book_id = book_dir.name
        chunks_file = book_dir / "chunks.jsonl"
        if not chunks_file.exists():
            continue
        meta = book_meta.get(book_id, {})
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunk["title"] = meta.get("title", "")
                chunk["author"] = meta.get("author", "")
                all_chunks.append(chunk)

    return all_chunks


def encode_chunks(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode texts into a (N, 768) float32 numpy matrix."""
    prefixed_texts = ["search_document: " + t for t in texts]
    embeddings = model.encode(
        prefixed_texts, batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def save_outputs(embeddings: np.ndarray, metadata: list[dict],
                 output_dir: Path = EMBEDDINGS_DIR) -> None:
    """Save embedding matrix and aligned metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.json"

    np.save(npy_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"Saved embeddings  → {npy_path}  shape={embeddings.shape}")
    print(f"Saved metadata    → {meta_path}  entries={len(metadata)}")


def run(processed_dir: str = None, device: str = None):
    """Main embedding entry point."""
    processed = Path(processed_dir) if processed_dir else PROCESSED_DIR

    print("=" * 60)
    print("Embedding Pipeline: Chunks → Vector Space (768-d)")
    print("=" * 60)

    all_chunks = load_all_chunks(processed)
    print(f"  Total chunks loaded: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks found. Run chunking first.")
        return

    texts: list[str] = []
    metadata: list[dict] = []
    for chunk in all_chunks:
        texts.append(chunk["text"])
        metadata.append({
            "book_id": chunk["book_id"], "chunk_id": chunk["chunk_id"],
            "segment_id": chunk["segment_id"], "timeline": chunk["timeline"],
            "start_sentence": chunk["start_sentence"], "end_sentence": chunk["end_sentence"],
            "start_word": chunk["start_word"], "end_word": chunk["end_word"],
            "word_count": chunk["word_count"],
            "title": chunk.get("title", ""), "author": chunk.get("author", ""),
        })

    print(f"\nLoading model '{MODEL_NAME}' on {device or 'default device'} …")
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"  ⚠️  Failed to load on {device}: {e}. Falling back to CPU.")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device="cpu")
        
    print(f"  Model dimension: {model.get_sentence_embedding_dimension()}")

    print(f"\nEncoding {len(texts)} chunks (batch_size={EMBED_BATCH_SIZE}) …")
    # Using a list comprehension to potentially save memory if items are large
    embeddings = encode_chunks(texts, model)

    assert embeddings.shape[0] == len(metadata)

    save_outputs(embeddings, metadata)
    print(f"\nDone! {embeddings.shape[0]} chunks embedded, {embeddings.nbytes / (1024**2):.1f} MB")
