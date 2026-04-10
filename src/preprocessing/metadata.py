"""
Metadata Consolidation

Merges per-book story_metadata.json files into a single metadata.jsonl.
"""

import json
from pathlib import Path

from src.config import PROCESSED_DIR


def consolidate_metadata(processed_dir: Path = None) -> int:
    """Consolidate all individual story_metadata.json files into one metadata.jsonl."""
    processed_dir = processed_dir or PROCESSED_DIR
    output_file = processed_dir / "metadata.jsonl"

    all_metadata = []
    for story_dir in sorted(processed_dir.iterdir()):
        if story_dir.is_dir():
            metadata_file = story_dir / "story_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        all_metadata.append(json.load(f))
                except Exception as e:
                    print(f"Error reading {metadata_file}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata) + "\n")

    print(f"Consolidated {len(all_metadata)} metadata entries into {output_file}")
    return len(all_metadata)


def run(processed_dir: str = None):
    """Entry point for metadata consolidation."""
    path = Path(processed_dir) if processed_dir else None
    count = consolidate_metadata(path)
    print(f"Successfully created metadata.jsonl with {count} entries")
