"""
Centralised configuration for the INLP Hallucinating‑Llamas pipeline.

All paths, model names, thresholds, and lexicons that were previously
scattered across individual scripts live here.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repo root

# Preferred processed-data location in this workspace.
# This is checked first so downstream stages (faiss/index/query) use it by default.
WORKSPACE_PROCESSED_DIR = Path(
    "/home/shravanikalmali/Desktop/inlp_final/drive-download-20260408T131838Z-3-001"
)

DATA_DIR_CANDIDATES = [
    WORKSPACE_PROCESSED_DIR,
    PROJECT_ROOT / "preprocessed_data",
    PROJECT_ROOT / "processed_data",
]
PROCESSED_DIR = next((p for p in DATA_DIR_CANDIDATES if p.exists()), PROJECT_ROOT / "processed_data")
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"


def find_data_directory(override: str | None = None) -> Path:
    """Return the first existing data directory, or *override* if given."""
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified data directory not found: {p}")
    for candidate in DATA_DIR_CANDIDATES:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        f"No data directory found. Searched: {[str(c) for c in DATA_DIR_CANDIDATES]}"
    )


# ── Embedding Model ──────────────────────────────────────────────────────────
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_DIM = 768
EMBED_BATCH_SIZE = 4

# Cross‑encoder for answer re‑ranking
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking / Segmentation ──────────────────────────────────────────────────
CHUNK_THRESHOLD_WORDS = 250
SEGMENT_THRESHOLD_WORDS = 2500
CHUNK_OVERLAP_SENTENCES = 1
SEGMENT_OVERLAP_CHUNKS = 1

# ── NER ───────────────────────────────────────────────────────────────────────
SPACY_MODELS = ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]
NER_BATCH_SIZE = 32
VALID_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"}

INVALID_PATTERNS = [
    r"CHAPTER", r"BOOK", r"PART", r"CONTENTS", r"Illustration",
    r"SECTION", r"VOLUME", r"PROLOGUE", r"EPILOGUE", r"PREFACE",
    r"INDEX", r"APPENDIX", r"NOTES", r"FOOTNOTE",
]
CHAPTER_HEADING_PATTERNS = [
    r"^\s*(CHAPTER|BOOK|PART|SECTION)\s+[IVXLCDM\d]+\.?\s*$",
    r"^\s*(CHAPTER|BOOK|PART|SECTION)\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\s*$",
    r"^\s*[IVXLCDM]+\.?\s*$",
    r"^\s*\d+\.?\s*$",
]
STOPLIST_ENTITIES = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "last", "next", "previous",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "prophet", "masque", "spirit", "stranger", "lady", "gentleman",
    "hose", "sop", "bush", "pierce", "fed", "chuck", "bray", "rob", "cud",
    "rouge", "asse", "budget", "spur", "pike", "nick", "peg", "meg", "gill",
    "ceri", "easter",
}
ENTITY_CORRECTIONS = {
    "romola": ("Romola", "PERSON"), "tito": ("Tito", "PERSON"),
    "tito melema": ("Tito Melema", "PERSON"), "baldassarre": ("Baldassarre", "PERSON"),
    "bardo": ("Bardo", "PERSON"), "bardo de bardi": ("Bardo de Bardi", "PERSON"),
    "tessa": ("Tessa", "PERSON"), "nello": ("Nello", "PERSON"),
    "savonarola": ("Savonarola", "PERSON"), "fra girolamo": ("Fra Girolamo", "PERSON"),
    "bernardo del nero": ("Bernardo del Nero", "PERSON"), "piero": ("Piero", "PERSON"),
    "dolfo": ("Dolfo", "PERSON"), "bratti": ("Bratti", "PERSON"),
    "brigida": ("Brigida", "PERSON"), "camilla": ("Camilla", "PERSON"),
    "cennini": ("Cennini", "PERSON"), "cronaca": ("Cronaca", "PERSON"),
    "dino": ("Dino", "PERSON"),
    "florence": ("Florence", "GPE"), "firenze": ("Florence", "GPE"),
    "rome": ("Rome", "GPE"), "milan": ("Milan", "GPE"),
    "venice": ("Venice", "GPE"), "naples": ("Naples", "GPE"),
    "arno": ("Arno", "LOC"),
    "duomo": ("Duomo", "FAC"), "san marco": ("San Marco", "FAC"),
    "signoria": ("Signoria", "ORG"), "medici": ("Medici", "ORG"),
}

# ── Entity Filtering ─────────────────────────────────────────────────────────
DEFAULT_ENTITY_FREQ_THRESHOLD = 1  # No aggressive filtering - keep all entities

# ── Coreference ───────────────────────────────────────────────────────────────
PERSON_PRONOUNS = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}

# ── Retrieval Defaults ────────────────────────────────────────────────────────
DEFAULT_TOP_K = 40        # Initial candidate pool
DEFAULT_MMR_LAMBDA = 0.7  # Balance relevance (0.7) with diversity (0.3) - was 0.9
DEFAULT_FINAL_COUNT = 15  # More context for sentence extraction - was 8
DEFAULT_EXPANSION_MAX = 2  # Minimal expansion
SUMMARY_BIAS = 1.2
HYBRID_ALPHA = 0.4        # Adjusted: 0.5BM25 + 0.5dense (see hybrid_search.py)
ENTITY_WEIGHT = 0.2       # Entity boost for query-matching chunks
BM25_WEIGHT = 0.5         # Balanced - was 0.3 (too low for keyword matching)
DENSE_WEIGHT = 0.5        # Balanced - was 0.7 (too high for dense-only)

EXPANSION_SIM_THRESHOLD = 0.35
KEYWORD_OVERLAP_WEIGHT = 0.4
CROSS_ENCODER_TOP_N = 30   # Increased from 20

# Sentence-level ranking & fusion
SENTENCE_SCORE_WEIGHT = 0.7
CHUNK_SCORE_WEIGHT = 0.3
FINAL_SIM_THRESHOLD = 0.25
MAX_CHUNKS_PER_SEGMENT = 2

QTYPE_ENTITY_LABELS = {
    "where": {"GPE", "LOC", "FAC"},
    "who":   {"PERSON", "ORG"},
    "when":  {"DATE", "TIME"},
}
FACTOID_QTYPES = {"where", "who", "when", "what"}

# ── LLM Answer Extraction ─────────────────────────────────────────────────────
USE_LLM_AS_DEFAULT = False  # Set to True to use LLM for all answers
LLM_MODEL_NAME = "google/flan-t5-small"  # Using a small, efficient model
LLM_MAX_CONTEXT_LENGTH = 3000  # Increased for OpenRouter models

# OpenRouter Settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "am", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need",
    "what", "who", "where", "when", "why", "how", "which", "whom", "whose",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we", "us",
    "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "if", "then", "than", "too", "very", "just", "about",
})
