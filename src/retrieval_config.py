"""
Centralised retrieval configuration — toggle system.

Every retrieval component can be independently enabled/disabled,
enabling ablation experiments from a single config dict or JSON file.
"""

import json
from pathlib import Path
from copy import deepcopy


# ── Default configuration  ────────────────────────────────────────────────────

RETRIEVAL_CONFIG: dict[str, bool] = {
    "use_dense":            True,
    "use_bm25":             True,
    "use_mmr":              True,            # Disable MMR for clean diversity
    "use_neighbors":        False,
    "use_graph_expansion":  False,            # Disable expansion to avoid noise
    "use_graph_reasoning":  True,             # Enable multi-hop inference
    "use_hybrid_llm":       False,
    "use_cross_encoder":    True,             # Keep for sentence ranking
    "use_llm_answer":       True,             # Use strict span-extraction prompt
}


def get_config() -> dict[str, bool]:
    """Return a copy of the current retrieval config."""
    return deepcopy(RETRIEVAL_CONFIG)


def set_config(overrides: dict[str, bool]) -> None:
    """Merge *overrides* into the global retrieval config."""
    for key, value in overrides.items():
        if key not in RETRIEVAL_CONFIG:
            raise KeyError(f"Unknown config key: {key}")
        RETRIEVAL_CONFIG[key] = value


def load_config(path: str | Path) -> dict[str, bool]:
    """Load a JSON config file, apply it, and return the resulting config."""
    with open(path, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    set_config(overrides)
    return get_config()


def reset_config() -> None:
    """Reset all toggles to clean QA baseline: simple retrieval + inference."""
    RETRIEVAL_CONFIG.update({
        "use_dense": True,
        "use_bm25": True,           # BM25-heavy retrieval
        "use_mmr": True,            # Keep MMR but with high lambda (0.9)
        "use_neighbors": False,     # No extra noise
        "use_graph_expansion": False,  # Disable - adds noise
        "use_graph_reasoning": True,   # Enable inference
        "use_hybrid_llm": False,
        "use_cross_encoder": True,  # Sentence re-ranking
        "use_llm_answer": True,    # Use sentence fusion only
    })


def config_summary() -> str:
    """Human-readable one-liner of enabled/disabled components."""
    parts = []
    for key, val in RETRIEVAL_CONFIG.items():
        label = key.replace("use_", "")
        parts.append(f"{label}={'ON' if val else 'OFF'}")
    return " | ".join(parts)
