"""
Named Entity Recognition — Stage 1.2

Extracts named entities from each chunk using spaCy, with strict filtering
to remove structural noise and common misclassifications.
"""

import json
import spacy
import sys
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.config import (
    SPACY_MODELS, NER_BATCH_SIZE, VALID_ENTITY_LABELS,
    INVALID_PATTERNS, CHAPTER_HEADING_PATTERNS, STOPLIST_ENTITIES,
    ENTITY_CORRECTIONS, PROCESSED_DIR, find_data_directory,
)
from src.utils import load_chunks


# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text_for_ner(text: str) -> str:
    """Remove chunking artifacts and prepare text for NER."""
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"—+", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    cleaned_words = [w for w in words if len(w) < 40]
    return " ".join(cleaned_words).strip()


def remove_chapter_headings(text: str) -> str:
    """DISABLED: Preserve all text including headings.

    Rationale: Headings may contain answers to questions like 'What is the chapter title?'
    Better to keep all signal and filter later if needed.
    """
    return text  # Return text unchanged


# ── Entity Filtering ─────────────────────────────────────────────────────────

def is_valid_entity(text: str, label: str) -> bool:
    """Keep it SIMPLE: just validate label and minimum length."""
    # Accept valid labels only
    if label not in {"PERSON", "GPE", "LOC", "ORG"}:
        return False

    # Minimum length check
    if len(text.strip()) < 2:
        return False

    return True


def apply_entity_corrections(text: str, label: str) -> tuple:
    """Apply manual corrections for known entities."""
    key = text.lower()
    if key in ENTITY_CORRECTIONS:
        return ENTITY_CORRECTIONS[key]
    return text, label


# ── NER Extraction ────────────────────────────────────────────────────────────

def extract_entities_from_chunks(book_dir: Path, nlp) -> List[Dict]:
    """Extract named entities from all chunks in a book."""
    chunks_file = book_dir / "chunks.jsonl"
    if not chunks_file.exists():
        print(f"  ⚠️  chunks.jsonl not found in {book_dir}")
        return []

    chunks = load_chunks(chunks_file)
    chunks.sort(key=lambda x: x["chunk_id"])

    cleaned_texts = [clean_text_for_ner(remove_chapter_headings(c["text"])) for c in chunks]

    chunk_entities = []
    docs = nlp.pipe(cleaned_texts, batch_size=NER_BATCH_SIZE)
    filtered_count, total_count = 0, 0

    for chunk, doc in zip(chunks, docs):
        entities = []
        seen = set()
        for ent in doc.ents:
            total_count += 1
            if not is_valid_entity(ent.text, ent.label_):
                filtered_count += 1
                continue
            corrected_text, corrected_label = apply_entity_corrections(ent.text, ent.label_)
            if corrected_text not in seen:
                entities.append({"text": corrected_text, "label": corrected_label})
                seen.add(corrected_text)

        chunk_entities.append({
            "chunk_id": chunk["chunk_id"],
            "segment_id": chunk.get("segment_id", 0),
            "entities": entities,
            "entity_count": len(entities),
        })

    if total_count > 0:
        print(f"     Filtered: {filtered_count}/{total_count} entities "
              f"({100*filtered_count/total_count:.1f}%)")
        print(f"     Kept: {total_count - filtered_count} clean entities")

    return chunk_entities


def save_entities(book_dir: Path, chunk_entities: List[Dict]) -> None:
    """Save extracted entities to entities.json."""
    entities_file = book_dir / "entities.json"
    with open(entities_file, "w", encoding="utf-8") as f:
        json.dump(chunk_entities, f, indent=2, ensure_ascii=False)

    total = sum(item["entity_count"] for item in chunk_entities)
    unique = len(set(e["text"] for item in chunk_entities for e in item["entities"]))
    print(f"    Saved: {entities_file}")
    print(f"    Total: {total} mentions, {unique} unique")


def process_book(book_dir: Path, nlp) -> bool:
    """Process NER for a single book."""
    book_id = book_dir.name
    if (book_dir / "entities.json").exists():
        print(f"  ℹ️  Skipping {book_id} — entities.json exists")
        return True

    print(f"\n── {book_id[:40]}{'…' if len(book_id) > 40 else ''}")
    chunk_entities = extract_entities_from_chunks(book_dir, nlp)
    if not chunk_entities:
        return False
    save_entities(book_dir, chunk_entities)
    return True


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_spacy_model(model_name: str = None):
    """Load best available spaCy model."""
    if model_name:
        nlp = spacy.load(model_name)
        print(f"  ✅ Loaded '{model_name}'")
        return nlp, model_name

    for name in SPACY_MODELS:
        try:
            nlp = spacy.load(name)
            print(f"  ✅ Loaded '{name}'")
            return nlp, name
        except OSError:
            continue

    print("  ❌ No spaCy models found!")
    sys.exit(1)


def run(book_id: str = None, processed_dir: str = None, model: str = None,
        overwrite: bool = False):
    """Main NER entry point."""
    processed = find_data_directory(processed_dir)

    print("=" * 60)
    print("  Named Entity Recognition (NER) — Stage 1.2")
    print("=" * 60)

    nlp, model_name = load_spacy_model(model)

    # Disable unnecessary components
    for component in ["parser", "lemmatizer"]:
        if component in nlp.pipe_names:
            nlp.disable_pipes(component)

    if book_id:
        book_dirs = [processed / book_id]
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
        print(f"\n  Found {len(book_dirs)} book directories")

    success, failed = 0, 0
    for book_dir in tqdm(book_dirs, desc="Processing books"):
        if overwrite and (book_dir / "entities.json").exists():
            (book_dir / "entities.json").unlink()
        if process_book(book_dir, nlp):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}\n  NER Complete! ✅ {success} books")
    if failed:
        print(f"  ❌ Failed: {failed}")
    print("=" * 60)
