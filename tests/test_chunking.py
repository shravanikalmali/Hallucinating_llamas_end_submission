"""
Test the chunking pipeline on a single book.
"""

import json
from pathlib import Path
from src.preprocessing.chunker import (
    index_sentences,
    build_chunks,
    apply_chunk_overlap,
    build_segments,
    calculate_timeline,
)
from src.config import PROCESSED_DIR


def main():
    """Test chunking on one book."""
    metadata_file = PROCESSED_DIR / "metadata.jsonl"
    with open(metadata_file, "r", encoding="utf-8") as f:
        first_book = json.loads(f.readline())

    book_id = first_book["story_id"]
    print(f"Testing on: {first_book.get('title', 'Unknown')}")
    print(f"Book ID: {book_id}")
    print("=" * 60)

    text_file = PROCESSED_DIR / book_id / "cleaned_text.txt"
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = index_sentences(text)
    print(f"✓ {len(sentences)} sentences")

    chunks = build_chunks(sentences)
    print(f"✓ {len(chunks)} base chunks")

    overlapped = apply_chunk_overlap(chunks, sentences)
    print(f"✓ {len(overlapped)} chunks (with overlap)")

    segments = build_segments(overlapped)
    print(f"✓ {len(segments)} segments")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Avg chunk: {sum(c['word_count'] for c in overlapped)/len(overlapped):.0f} words")
    print(f"Avg segment: {sum(s['word_count'] for s in segments)/len(segments):.0f} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
