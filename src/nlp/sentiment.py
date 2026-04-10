"""
Segment-Level Sentiment Analysis — Stage 1.6

Computes TextBlob polarity/subjectivity for each segment.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.config import find_data_directory
from src.utils import save_json

try:
    from textblob import TextBlob
except ImportError:
    print("❌ textblob not installed! pip install textblob && python -m textblob.download_corpora")
    sys.exit(1)


def compute_sentiment_for_book(book_dir: Path) -> dict | None:
    """Compute sentiment polarity for each segment in a book."""
    book_dir = Path(book_dir)
    chunks_file = book_dir / "chunks.jsonl"
    if not chunks_file.exists():
        print(f"  ⚠️  Skipping {book_dir.name} — chunks.jsonl not found")
        return None

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # Group by segment
    seg_chunks = defaultdict(list)
    for chunk in chunks:
        seg_chunks[chunk["segment_id"]].append(chunk)

    segment_texts = {}
    for seg_id, chunk_list in seg_chunks.items():
        chunk_list.sort(key=lambda x: x["chunk_id"])
        segment_texts[seg_id] = " ".join(ch["text"] for ch in chunk_list)

    scores = {}
    details = {}
    for seg_id, text in segment_texts.items():
        try:
            blob = TextBlob(text)
            scores[seg_id] = blob.sentiment.polarity
            details[seg_id] = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
                "word_count": len(text.split()),
            }
        except Exception as e:
            scores[seg_id] = 0.0
            details[seg_id] = {"polarity": 0.0, "subjectivity": 0.0,
                               "word_count": len(text.split()), "error": str(e)}

    save_json(book_dir / "segment_sentiment.json", scores)
    save_json(book_dir / "segment_sentiment_details.json", details)

    polarities = list(scores.values())
    avg = sum(polarities) / len(polarities) if polarities else 0
    print(f"  ✅ {len(scores)} segments, avg polarity: {avg:.3f}")
    return scores


def run(book_id: str = None, processed_dir: str = None, overwrite: bool = False):
    """Main sentiment entry point."""
    processed = find_data_directory(processed_dir)

    print("\n" + "=" * 60)
    print("  Segment-Level Sentiment — Stage 1.6")
    print("=" * 60)

    book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
    if book_id:
        book_dirs = [d for d in book_dirs if d.name == book_id]

    print(f"  Found {len(book_dirs)} book directories\n")

    processed_count, skipped = 0, 0
    for book_dir in tqdm(book_dirs, desc="Computing sentiment"):
        sentiment_file = book_dir / "segment_sentiment.json"
        if sentiment_file.exists() and not overwrite:
            skipped += 1
            continue
        try:
            compute_sentiment_for_book(book_dir)
            processed_count += 1
        except Exception as e:
            print(f"  ❌ Error: {book_dir.name}: {e}")

    print(f"\n{'='*60}\n  Sentiment Complete! ✅ {processed_count} books")
    if skipped:
        print(f"  ⏭️  Skipped: {skipped}")
    print("=" * 60)
