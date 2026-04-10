"""
Candidate Generation — Entity Indexer

Creates a direct mapping from entities to chunk IDs for fast lookup.
This improves recall for "Who/What/Where" questions.
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.config import find_data_directory
from src.utils import load_json, save_json

def build_entity_index(book_dir: Path) -> dict:
    """Build a mapping from entity (text) to list of chunk_ids."""
    fce_file = book_dir / "filtered_chunk_entities.json"
    if not fce_file.exists():
        return {}

    chunk_entities = load_json(fce_file)
    entity_index = defaultdict(list)

    for item in chunk_entities:
        cid = item["chunk_id"]
        for ent in item["entities"]:
            # ent is expected to be the canonical name (string) 
            # as it comes from filtered_chunk_entities.json
            entity_index[ent.lower()].append(cid)

    # Convert defaultdict to regular dict for JSON serialization
    return dict(entity_index)

def process_book(book_dir: Path, overwrite: bool = False) -> bool:
    """Process a single book's entity index."""
    book_id = book_dir.name
    index_file = book_dir / "entity_index.json"

    if index_file.exists() and not overwrite:
        return True

    index = build_entity_index(book_dir)
    if not index:
        return False

    save_json(index_file, index)
    return True

def run(book_id: str = None, processed_dir: str = None, overwrite: bool = False):
    """Main entity index building entry point."""
    processed = find_data_directory(processed_dir)

    print("=" * 60)
    print("  Entity Indexing — Fast Candidate Lookup")
    print("=" * 60)

    if book_id:
        book_dirs = [processed / book_id]
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
    
    success = 0
    for book_dir in tqdm(book_dirs, desc="Building entity indexes"):
        if process_book(book_dir, overwrite):
            success += 1
            
    print(f"\nEntity Indexing Complete! ✅ {success} books")
