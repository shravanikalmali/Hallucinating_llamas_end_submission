"""
Entity Filtering — Stage 1.4

Filters entities by global frequency, keeping only narratively significant ones.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Set
from collections import Counter

from src.config import DEFAULT_ENTITY_FREQ_THRESHOLD, find_data_directory
from src.utils import load_json, save_json


def normalize_entity(entity_text: str) -> str:
    """Normalize entity text: just strip whitespace (DO NOT remove articles)."""
    return entity_text.strip()


def canonicalize_entities(raw_entities: List[Dict], coref_map: Dict[str, str]) -> List[Dict]:
    """Map raw entity mentions to canonical IDs using coreference (non-destructive).

    Stores both surface form and canonical form to preserve original text appearance.
    """
    result = []
    for item in raw_entities:
        canonical_set = set()
        for ent in item["entities"]:
            text = ent["text"] if isinstance(ent, dict) else ent
            # Direct lookup only - NO nested normalize calls
            canonical = coref_map.get(text, text)
            if canonical:
                canonical_set.add(canonical)
        result.append({
            "chunk_id": item["chunk_id"],
            "segment_id": item.get("segment_id", 0),
            "entities": list(canonical_set),
        })
    return result


def filter_by_frequency(canonical_entities: List[Dict], threshold: int
                         ) -> tuple[List[Dict], Set[str]]:
    """NO FILTERING - Keep all entities (threshold is ignored).

    In narrative QA, answers are often rare named entities.
    Filtering by frequency removes exactly the entities we need.
    """
    freq = Counter(e for item in canonical_entities for e in item["entities"])

    # Keep ALL entities regardless of frequency
    filtered = []
    for item in canonical_entities:
        filtered.append({
            "chunk_id": item["chunk_id"], "segment_id": item["segment_id"],
            "entities": item["entities"], "entity_count": len(item["entities"]),
        })

    frequent = set(freq.keys())  # Return all for stats
    return filtered, frequent, freq


def process_book(book_dir: Path, threshold: int) -> bool:
    """Filter entities for a single book."""
    book_id = book_dir.name
    if (book_dir / "filtered_chunk_entities.json").exists():
        print(f"  ℹ️  Skipping {book_id} — already filtered")
        return True

    print(f"\n── {book_id[:40]}{'…' if len(book_id) > 40 else ''}")

    coref_file = book_dir / "coref_map.json"
    coref_map = load_json(coref_file) if coref_file.exists() else {}

    entities_file = book_dir / "entities.json"
    if not entities_file.exists():
        return False
    raw_entities = load_json(entities_file)

    canonical = canonicalize_entities(raw_entities, coref_map)
    filtered, frequent, freq = filter_by_frequency(canonical, threshold)

    total_before = sum(len(item["entities"]) for item in canonical)
    total_after = sum(item["entity_count"] for item in filtered)
    print(f"    Kept {len(frequent)} frequent entities, mentions: {total_before} → {total_after}")

    save_json(book_dir / "filtered_chunk_entities.json", filtered)
    save_json(book_dir / "frequent_entities.json", sorted(list(frequent)))
    save_json(book_dir / "entity_stats.json", {
        "total_entities": len(freq),
        "frequent_entities": len(frequent),
        "top_entities": [{"entity": e, "count": c} for e, c in freq.most_common(50)],
    })
    return True


def run(book_id: str = None, processed_dir: str = None,
        threshold: int = DEFAULT_ENTITY_FREQ_THRESHOLD, overwrite: bool = False):
    """Main entity filtering entry point."""
    processed = find_data_directory(processed_dir)

    print("=" * 60)
    print(f"  Entity Filtering — Stage 1.4 (threshold={threshold})")
    print("=" * 60)

    if book_id:
        book_dirs = [processed / book_id]
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
        print(f"\n  Found {len(book_dirs)} book directories")

    success, failed = 0, 0
    for book_dir in tqdm(book_dirs, desc="Filtering entities"):
        if overwrite:
            for fn in ["filtered_chunk_entities.json", "frequent_entities.json", "entity_stats.json"]:
                fp = book_dir / fn
                if fp.exists():
                    fp.unlink()
        if process_book(book_dir, threshold):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}\n  Filtering Complete! ✅ {success} books\n{'='*60}")
