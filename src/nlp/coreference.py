"""
Coreference Resolution — Stage 1.3

Merged module combining:
- Neural coreference via fastcoref (optional)
- Rule-based name-matching coreference
- Pronoun resolution (he/she/him/her → nearest PERSON)
"""

import json
import spacy
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

from src.config import PERSON_PRONOUNS, SPACY_MODELS, find_data_directory
from src.utils import load_chunks, load_json, save_json


# ── Helpers ───────────────────────────────────────────────────────────────────

def choose_canonical_mention(cluster: List[str]) -> str:
    """Choose the best representative mention from a coreference cluster."""
    if not cluster:
        return ""
    freq = Counter(cluster)

    def score(mention):
        words = mention.split()
        is_proper = all(w[0].isupper() for w in words if w)
        return (is_proper, len(mention), freq[mention])

    return max(cluster, key=score)


# ── Name-Matching Coreference ─────────────────────────────────────────────────

def run_name_matching_coref(entities_per_chunk: List[Dict]) -> Dict[str, str]:
    """Simple rule-based coreference using name substring/overlap matching."""
    all_entities = set()
    for chunk_data in entities_per_chunk:
        for ent in chunk_data.get("entities", []):
            text = ent["text"] if isinstance(ent, dict) else ent
            if text and len(text.strip()) > 1:
                all_entities.add(text.strip())

    clusters = []
    processed = set()

    for entity in sorted(all_entities, key=len, reverse=True):
        if entity in processed:
            continue
        cluster = [entity]
        processed.add(entity)
        entity_lower = entity.lower()
        entity_words = set(entity_lower.split())

        for other in all_entities:
            if other in processed:
                continue
            other_lower = other.lower()
            other_words = set(other_lower.split())

            is_match = entity_lower in other_lower or other_lower in entity_lower
            if not is_match and len(entity_words) > 1 and len(other_words) > 0:
                overlap = entity_words & other_words
                if overlap and len(overlap) >= min(len(entity_words), len(other_words)) * 0.5:
                    is_match = True

            if is_match:
                cluster.append(other)
                processed.add(other)

        if len(cluster) > 1:
            clusters.append(cluster)

    mapping = {}
    for cluster in clusters:
        canonical = choose_canonical_mention(cluster)
        for mention in cluster:
            mapping[mention] = canonical

    print(f"    Found {len(clusters)} name clusters")
    print(f"    Mapped {len(mapping)} mentions → {len(set(mapping.values()))} canonical names")
    return mapping


# ── Pronoun Resolution ────────────────────────────────────────────────────────

def resolve_pronouns_in_chunk(chunk_text: str, chunk_entities: List[Dict],
                               previous_persons: List[str], nlp,
                               gender_hints: Dict[str, str] = None,
                               ) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """Resolve pronouns in a single chunk to nearest preceding PERSON.

    Uses gender hints when available: male pronouns prefer male entities,
    female pronouns prefer female entities.
    """
    from src.config import MALE_PRONOUNS, FEMALE_PRONOUNS

    if gender_hints is None:
        gender_hints = {}

    doc = nlp(chunk_text)
    pronoun_map = {}

    person_entities = [ent["text"] for ent in chunk_entities if ent["label"] == "PERSON"]
    # ONLY use pronouns from SAME CHUNK - do not cross-chunk resolution
    recency_stack = person_entities

    # Update gender hints from co-occurring pronouns and entities
    for token in doc:
        if token.pos_ == "PRON" and token.lemma_.lower() in PERSON_PRONOUNS:
            if token.lemma_.lower() in MALE_PRONOUNS and person_entities:
                for pe in person_entities:
                    if pe not in gender_hints:
                        gender_hints[pe] = "male"
            elif token.lemma_.lower() in FEMALE_PRONOUNS and person_entities:
                for pe in person_entities:
                    if pe not in gender_hints:
                        gender_hints[pe] = "female"

    for token in doc:
        if token.pos_ == "PRON" and token.lemma_.lower() in PERSON_PRONOUNS:
            if not recency_stack:
                continue

            pronoun_lower = token.lemma_.lower()

            # Gender-filtered resolution
            if pronoun_lower in MALE_PRONOUNS:
                candidates = [p for p in recency_stack
                              if gender_hints.get(p) in ("male", None)]
            elif pronoun_lower in FEMALE_PRONOUNS:
                candidates = [p for p in recency_stack
                              if gender_hints.get(p) in ("female", None)]
            else:
                candidates = recency_stack

            if candidates:
                pronoun_map[token.text] = candidates[0]
            elif recency_stack:
                pronoun_map[token.text] = recency_stack[0]

    updated_stack = person_entities + previous_persons
    return pronoun_map, updated_stack[:5], gender_hints


def merge_name_variants(chunk_entities: List[Dict]) -> Dict[str, str]:
    """Merge name variants (e.g. 'Tito Melema' and 'Tito' → 'Tito Melema')."""
    persons = [
        ent["text"] for ent_list in chunk_entities
        for ent in ent_list.get("entities", [])
        if ent["label"] == "PERSON"
    ]
    person_counts = Counter(persons)
    name_map = {}
    sorted_persons = sorted(set(persons), key=len, reverse=True)

    for longer in sorted_persons:
        for shorter in sorted_persons:
            if longer == shorter:
                continue
            if shorter.lower() in longer.lower():
                if person_counts[longer] >= person_counts[shorter]:
                    name_map[shorter] = longer
                else:
                    name_map[longer] = shorter
    return name_map


def run_pronoun_coref(book_dir: Path, nlp) -> Dict[str, str]:
    """Full coreference with pronoun resolution."""
    entities_file = book_dir / "entities.json"
    chunks_file = book_dir / "chunks.jsonl"

    if not entities_file.exists() or not chunks_file.exists():
        return {}

    chunk_entities = load_json(entities_file)
    chunks = load_chunks(chunks_file)
    chunks.sort(key=lambda x: x["chunk_id"])

    # Step 1: Name variants
    name_map = merge_name_variants(chunk_entities)
    print(f"  ✓ Found {len(name_map)} name variant mappings")

    # Step 2: Pronouns (with gender-aware resolution)
    pronoun_map = {}
    previous_persons = []
    gender_hints = {}
    for chunk, chunk_ent in tqdm(zip(chunks, chunk_entities), total=len(chunks),
                                  desc="  Resolving pronouns", leave=False):
        chunk_pmap, previous_persons, gender_hints = resolve_pronouns_in_chunk(
            chunk["text"], chunk_ent.get("entities", []), previous_persons, nlp,
            gender_hints=gender_hints)
        pronoun_map.update(chunk_pmap)

    print(f"  ✓ Resolved {len(pronoun_map)} pronouns")

    # Step 3: Merge all
    coref_map = {}
    coref_map.update(name_map)
    for pronoun, person in pronoun_map.items():
        coref_map[pronoun] = name_map.get(person, person)

    # Identity mappings for unmapped entities
    all_entities = set()
    for chunk_ent in chunk_entities:
        for ent in chunk_ent.get("entities", []):
            all_entities.add(ent["text"])
    for entity in all_entities:
        if entity not in coref_map:
            coref_map[entity] = name_map.get(entity, entity)

    return coref_map


# ── Neural Coreference ────────────────────────────────────────────────────────

def run_neural_coref(book_dir: Path, device: str = "cpu") -> Dict[str, str]:
    """Run fastcoref neural coreference resolution."""
    # Load full text
    cleaned_file = book_dir / "cleaned_text.txt"
    if not cleaned_file.exists():
        chunks = load_chunks(book_dir / "chunks.jsonl")
        chunks.sort(key=lambda x: x["chunk_id"])
        full_text = " ".join(c["text"] for c in chunks[::2])
    else:
        with open(cleaned_file, "r", encoding="utf-8") as f:
            full_text = f.read()

    if not full_text:
        return {}

    MAX_WORDS = 100000
    words = full_text.split()
    if len(words) > MAX_WORDS:
        full_text = " ".join(words[:MAX_WORDS])

    try:
        from fastcoref import FCoref
        model = FCoref(device=device)
        pred = model.predict(texts=[full_text])
        clusters = pred[0].get_clusters()

        mapping = {}
        for cluster in clusters:
            if not cluster:
                continue
            canonical = choose_canonical_mention(cluster)
            for mention in cluster:
                mapping[mention] = canonical

        print(f"    Neural coref: {len(clusters)} clusters, {len(mapping)} mappings")
        return mapping
    except Exception as e:
        print(f"    ⚠️  Neural coref failed: {e}")
        return {}


# ── Book Processing ───────────────────────────────────────────────────────────

def process_book(book_dir: Path, nlp, engine: str = "pronoun",
                 device: str = "cpu") -> bool:
    """Process coreference for a single book."""
    book_id = book_dir.name
    map_file = book_dir / "coref_map.json"

    if map_file.exists():
        print(f"  ℹ️  Skipping {book_id} — coref_map.json exists")
        return True

    print(f"\n── {book_id[:40]}{'…' if len(book_id) > 40 else ''}")

    if engine == "pronoun":
        coref_map = run_pronoun_coref(book_dir, nlp)
    elif engine == "neural":
        coref_map = run_neural_coref(book_dir, device)
        if not coref_map:
            # Fallback to name matching
            entities = load_json(book_dir / "entities.json") if (book_dir / "entities.json").exists() else []
            coref_map = run_name_matching_coref(entities)
    else:
        entities = load_json(book_dir / "entities.json") if (book_dir / "entities.json").exists() else []
        coref_map = run_name_matching_coref(entities)

    save_json(map_file, coref_map or {})
    print(f"    Saved: {map_file}")
    return True


def run(book_id: str = None, processed_dir: str = None,
        engine: str = "pronoun", device: str = "cpu", overwrite: bool = False):
    """Main coreference entry point."""
    processed = find_data_directory(processed_dir)

    print("=" * 60)
    print(f"  Coreference Resolution — Stage 1.3 ({engine})")
    print("=" * 60)

    nlp = None
    if engine == "pronoun":
        for name in SPACY_MODELS:
            try:
                nlp = spacy.load(name)
                nlp.disable_pipes([c for c in ["lemmatizer", "ner"] if c in nlp.pipe_names])
                print(f"  ✓ Loaded {name}")
                break
            except OSError:
                continue
        if nlp is None:
            print("  ❌ No spaCy model found")
            sys.exit(1)

    if book_id:
        book_dirs = [processed / book_id]
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
        print(f"\n  Found {len(book_dirs)} book directories")

    success = 0
    for book_dir in tqdm(book_dirs, desc="Processing books"):
        if overwrite and (book_dir / "coref_map.json").exists():
            (book_dir / "coref_map.json").unlink()
        if process_book(book_dir, nlp, engine, device):
            success += 1

    print(f"\n{'='*60}\n  Coreference Complete! ✅ {success} books\n{'='*60}")
