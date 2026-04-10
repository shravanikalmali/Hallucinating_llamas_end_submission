"""
Shared utility functions used across multiple pipeline stages.
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Any

from src.config import STOP_WORDS


# ── I/O Helpers ───────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    """Load chunk records from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_json(path: Path) -> Any:
    """Load a JSON file and return its contents, or None if missing."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Save a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Math Helpers ──────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (fast path for unit vectors)."""
    return float(np.dot(a, b))


def cosine_similarity_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with norm handling (for non-normalised vectors)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Text Helpers ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase tokenization with stop-word removal."""
    return [w for w in re.findall(r"\w+", text.lower())
            if w not in STOP_WORDS and len(w) > 1]


def keyword_overlap_ratio(question: str, sentence: str) -> float:
    """Fraction of question content-words found in the sentence."""
    q_words = set(tokenize(question))
    if not q_words:
        return 0.0
    s_words = set(tokenize(sentence))
    return len(q_words & s_words) / len(q_words)
