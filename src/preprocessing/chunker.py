"""
Chunking Pipeline — Phase 0

Pipeline: Raw Text → Sentences → Chunks (~250 words) → Segments (~2500 words)
with 1-sentence chunk overlap and 1-chunk segment overlap.
"""

import json
import nltk
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.config import (
    CHUNK_THRESHOLD_WORDS, SEGMENT_THRESHOLD_WORDS,
    CHUNK_OVERLAP_SENTENCES, SEGMENT_OVERLAP_CHUNKS,
    PROCESSED_DIR,
)

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def preprocess_text_for_sentences(text: str) -> str:
    """Join line-wrapped sentences within paragraphs while preserving paragraph breaks."""
    paragraphs = text.split("\n\n")
    processed = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        joined = " ".join(line.strip() for line in para.split("\n") if line.strip())
        joined = " ".join(joined.split())
        if joined:
            processed.append(joined)
    return "\n\n".join(processed)


def index_sentences(text: str) -> List[Dict]:
    """Stage 1: Split text into indexed sentence records using NLTK."""
    text = preprocess_text_for_sentences(text)
    sentences_raw = nltk.sent_tokenize(text)

    sentences = []
    word_position = 0
    for sentence_id, sent_text in enumerate(sentences_raw):
        sent_text = sent_text.strip()
        if not sent_text:
            continue
        words = sent_text.split()
        word_count = len(words)
        if word_count > 0:
            sentences.append({
                "sentence_id": sentence_id,
                "text": sent_text,
                "start_word": word_position,
                "end_word": word_position + word_count - 1,
                "word_count": word_count,
            })
            word_position += word_count
    return sentences


def build_chunks(sentences: List[Dict]) -> List[Dict]:
    """Stage 2: Group sentences into ~250-word chunks."""
    if not sentences:
        return []

    chunks = []
    chunk_id = 0
    i = 0

    while i < len(sentences):
        current_words = 0
        chunk_sentences = []
        while i < len(sentences):
            sent = sentences[i]
            current_words += sent["word_count"]
            chunk_sentences.append(sent)
            i += 1
            if current_words >= CHUNK_THRESHOLD_WORDS:
                break

        chunks.append({
            "chunk_id": chunk_id,
            "start_sentence": chunk_sentences[0]["sentence_id"],
            "end_sentence": chunk_sentences[-1]["sentence_id"],
            "start_word": chunk_sentences[0]["start_word"],
            "end_word": chunk_sentences[-1]["end_word"],
            "word_count": current_words,
            "text": " ".join(s["text"] for s in chunk_sentences),
        })
        chunk_id += 1

    return chunks


def apply_chunk_overlap(chunks: List[Dict], sentences: List[Dict]) -> List[Dict]:
    """Stage 3: Create 1-sentence overlap between adjacent chunks."""
    if len(chunks) <= 1:
        return chunks

    overlapped = []
    sentence_dict = {s["sentence_id"]: s for s in sentences}

    for i, chunk in enumerate(chunks):
        overlapped.append(chunk)
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            overlap_start = chunk["end_sentence"]
            overlap_end = next_chunk["end_sentence"]

            overlap_sentences = []
            total_words = 0
            for sent_id in range(overlap_start, overlap_end + 1):
                if sent_id in sentence_dict:
                    sent = sentence_dict[sent_id]
                    overlap_sentences.append(sent)
                    total_words += sent["word_count"]

            if overlap_sentences:
                overlapped.append({
                    "chunk_id": len(overlapped),
                    "start_sentence": overlap_sentences[0]["sentence_id"],
                    "end_sentence": overlap_sentences[-1]["sentence_id"],
                    "start_word": overlap_sentences[0]["start_word"],
                    "end_word": overlap_sentences[-1]["end_word"],
                    "word_count": total_words,
                    "text": " ".join(s["text"] for s in overlap_sentences),
                })

    # Re-index chunk_ids
    for idx, chunk in enumerate(overlapped):
        chunk["chunk_id"] = idx

    return overlapped


def build_segments(chunks: List[Dict]) -> List[Dict]:
    """Stage 4: Group chunks into ~2500-word segments with 1-chunk overlap."""
    if not chunks:
        return []

    segments = []
    segment_id = 0
    i = 0

    while i < len(chunks):
        if segment_id > 0 and i > 0:
            i -= SEGMENT_OVERLAP_CHUNKS

        current_words = 0
        segment_chunks = []
        while i < len(chunks):
            chunk = chunks[i]
            current_words += chunk["word_count"]
            segment_chunks.append(chunk)
            i += 1
            if current_words >= SEGMENT_THRESHOLD_WORDS:
                break

        segments.append({
            "segment_id": segment_id,
            "start_chunk": segment_chunks[0]["chunk_id"],
            "end_chunk": segment_chunks[-1]["chunk_id"],
            "start_sentence": segment_chunks[0]["start_sentence"],
            "end_sentence": segment_chunks[-1]["end_sentence"],
            "start_word": segment_chunks[0]["start_word"],
            "end_word": segment_chunks[-1]["end_word"],
            "word_count": current_words,
            "chunk_count": len(segment_chunks),
        })
        segment_id += 1

    return segments


def calculate_timeline(segment_id: int, total_segments: int) -> float:
    """Calculate timeline position (0.0 to 1.0) for a segment."""
    if total_segments <= 1:
        return 0.0
    return segment_id / (total_segments - 1)


def process_book(book_id: str, text: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process a single book through the complete chunking pipeline."""
    sentences = index_sentences(text)
    if not sentences:
        return [], [], []

    chunks = build_chunks(sentences)
    chunks = apply_chunk_overlap(chunks, sentences)
    segments = build_segments(chunks)
    return sentences, chunks, segments


def save_chunks_with_metadata(book_id: str, chunks: List[Dict], segments: List[Dict],
                               book_dir: Path):
    """Save chunks with segment information to JSONL."""
    chunk_to_segment = {}
    for segment in segments:
        for chunk_id in range(segment["start_chunk"], segment["end_chunk"] + 1):
            chunk_to_segment[chunk_id] = segment["segment_id"]

    total_segments = len(segments)
    chunks_file = book_dir / "chunks.jsonl"

    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            segment_id = chunk_to_segment.get(chunk_id, 0)
            timeline = calculate_timeline(segment_id, total_segments)
            entry = {
                "book_id": book_id, "chunk_id": chunk_id,
                "segment_id": segment_id, "timeline": round(timeline, 4),
                "start_sentence": chunk["start_sentence"],
                "end_sentence": chunk["end_sentence"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "word_count": chunk["word_count"],
                "text": chunk["text"],
            }
            f.write(json.dumps(entry) + "\n")


def run(processed_dir: str = None):
    """Main chunking entry point."""
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    metadata_file = processed_dir / "metadata.jsonl"

    print("Using NLTK for sentence tokenization")
    print("Reading metadata...")
    books = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            books.append(json.loads(line))

    print(f"Found {len(books)} books to process")

    total_chunks, total_segments = 0, 0
    for book in tqdm(books, desc="Processing books"):
        book_id = book["story_id"]
        book_dir = processed_dir / book_id
        text_file = book_dir / "cleaned_text.txt"
        if not text_file.exists():
            print(f"\nWarning: Text file not found for {book_id}")
            continue
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            sentences, chunks, segments = process_book(book_id, text)
            if not chunks:
                continue
            save_chunks_with_metadata(book_id, chunks, segments, book_dir)
            total_chunks += len(chunks)
            total_segments += len(segments)
        except Exception as e:
            print(f"\nError processing {book_id}: {e}")

    print(f"\n{'='*60}\nChunking Complete!")
    print(f"Total chunks: {total_chunks}, Total segments: {total_segments}")
    print(f"{'='*60}")
