"""
Text Cleaning — Phase 0

Extracts text from Project Gutenberg files, cleans decorative separators,
normalises whitespace, and extracts metadata (title, author, etc.).
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import PROCESSED_DIR


def extract_metadata(header_text: str) -> Dict[str, str]:
    """Extract metadata from the Gutenberg header."""
    metadata = {"title": "", "author": "", "release_date": "", "language": ""}

    for key, pattern in [
        ("title",        r"Title:\s*(.+?)(?:\n|$)"),
        ("author",       r"Author:\s*(.+?)(?:\n|$)"),
        ("release_date", r"Release date:\s*(.+?)(?:\[|$)"),
        ("language",     r"Language:\s*(.+?)(?:\n|$)"),
    ]:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            metadata[key] = match.group(1).strip()

    return metadata


def clean_text(text: str) -> str:
    """Clean and normalise raw story text."""
    lines = text.split("\n")
    lines = [line.strip() for line in lines]

    cleaned_lines = []
    for line in lines:
        if not line:
            cleaned_lines.append(line)
            continue
        # Skip decorative separator lines
        if len(set(line)) == 1 and not line[0].isalnum():
            continue
        non_alnum = sum(1 for c in line if not c.isalnum() and c != " ")
        if non_alnum > 0.8 * len(line):
            unique_chars = set(line.replace(" ", ""))
            if len(unique_chars) <= 3:
                continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_story_content(file_path: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract main story content and metadata from a Gutenberg file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    # Find START marker
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    start_idx = full_text.find(start_marker)
    if start_idx == -1:
        start_marker = "***START OF THE PROJECT GUTENBERG EBOOK"
        start_idx = full_text.find(start_marker)
    if start_idx == -1:
        print(f"Warning: START marker not found in {file_path}")
        return None, None

    header_text = full_text[:start_idx]
    metadata = extract_metadata(header_text)

    start_line_end = full_text.find("\n", start_idx)
    if start_line_end == -1:
        start_line_end = start_idx + len(start_marker)
    else:
        start_line_end += 1

    # Find END marker
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    end_idx = full_text.find(end_marker, start_line_end)
    if end_idx == -1:
        end_marker = "***END OF THE PROJECT GUTENBERG EBOOK"
        end_idx = full_text.find(end_marker, start_line_end)

    story_text = full_text[start_line_end:end_idx] if end_idx != -1 else full_text[start_line_end:]
    return clean_text(story_text), metadata


def count_words(text: str) -> int:
    """Count words in the text."""
    return len(text.split())


def process_story(story_id: str, file_path: str, output_dir: str, story_url: str = "") -> bool:
    """Process a single story file: extract, clean, and save."""
    cleaned_text, metadata = extract_story_content(file_path)
    if cleaned_text is None:
        return False

    word_count = count_words(cleaned_text)
    story_dir = os.path.join(output_dir, story_id)
    os.makedirs(story_dir, exist_ok=True)

    with open(os.path.join(story_dir, "cleaned_text.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    full_metadata = {
        "story_id": story_id,
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "release_date": metadata.get("release_date", ""),
        "language": metadata.get("language", ""),
        "total_words": word_count,
        "source_url": story_url,
    }
    with open(os.path.join(story_dir, "story_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2)

    return True


def run(data_dir: str = None, documents_csv: str = None):
    """Main preprocessing entry point."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = Path(data_dir) if data_dir else base_dir / "data"
    documents_csv = Path(documents_csv) if documents_csv else base_dir / "narrativeqa-master" / "narrativeqa-master" / "documents.csv"
    output_dir = PROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("Reading documents.csv...")
    documents = []
    with open(documents_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["kind"] == "gutenberg":
                documents.append(row)

    print(f"Found {len(documents)} Gutenberg documents to process")

    success_count, failed_count = 0, 0
    all_metadata = []

    for i, doc in enumerate(documents, 1):
        story_id = doc["document_id"]
        story_file = data_dir / f"{story_id}.jsonl"

        if not story_file.exists():
            print(f"[{i}/{len(documents)}] File not found: {story_file}")
            failed_count += 1
            continue

        print(f"[{i}/{len(documents)}] Processing {doc.get('wiki_title', story_id)}...")
        success = process_story(story_id, str(story_file), str(output_dir), doc.get("story_url", ""))

        if success:
            success_count += 1
            metadata_file = output_dir / story_id / "story_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    all_metadata.append(json.load(f))
        else:
            failed_count += 1

    # Create consolidated metadata
    metadata_jsonl = output_dir / "metadata.jsonl"
    with open(metadata_jsonl, "w", encoding="utf-8") as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata) + "\n")

    print(f"\n{'='*60}")
    print(f"Preprocessing Complete! Success: {success_count}, Failed: {failed_count}")
    print(f"{'='*60}")
