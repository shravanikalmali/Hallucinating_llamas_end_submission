"""
Entity-Based Chunk Expansion

Uses the static knowledge graph to find additional chunks that share
entities with the current selection, with a query-relevance gate.
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path

from src.config import DEFAULT_EXPANSION_MAX, EXPANSION_SIM_THRESHOLD
from src.utils import cosine_similarity


def entity_expansion(
    selected_chunks: list[dict], all_chunks: list[dict], book_dir: Path,
    query_vec: np.ndarray | None = None, all_embeddings: np.ndarray | None = None,
    max_add: int = DEFAULT_EXPANSION_MAX,
) -> list[dict]:
    """Find additional chunks sharing entities with the selection.

    Query-relevance gate: only chunks with cosine similarity > threshold are accepted.
    Prefers chunks from segments not yet represented.
    """
    graph_file = book_dir / "graph.graphml"
    filtered_ents_file = book_dir / "filtered_chunk_entities.json"

    if not graph_file.exists() or not filtered_ents_file.exists():
        return selected_chunks

    G = nx.read_graphml(str(graph_file))
    with open(filtered_ents_file, "r", encoding="utf-8") as f:
        chunk_entities = json.load(f)

    chunk_ent_map: dict[int, list[str]] = {
        item["chunk_id"]: item["entities"] for item in chunk_entities
    }

    selected_ids = {c["chunk_id"] for c in selected_chunks}
    selected_segments = {c["segment_id"] for c in selected_chunks}

    # Collect seed entities from selected chunks
    seed_entities: set[str] = set()
    for c in selected_chunks:
        seed_entities.update(chunk_ent_map.get(c["chunk_id"], []))

    if not seed_entities:
        return selected_chunks

    # Expand via graph neighbours
    expanded_entities = set(seed_entities)
    for ent in seed_entities:
        if G.has_node(ent):
            for neighbour in G.neighbors(ent):
                expanded_entities.add(neighbour)

    # Score candidates
    scored: list[tuple[float, int, dict]] = []
    for chunk in all_chunks:
        cid, sid = chunk["chunk_id"], chunk["segment_id"]
        if cid in selected_ids:
            continue
        ents = set(chunk_ent_map.get(cid, []))
        overlap = ents & expanded_entities
        if not overlap:
            continue

        sim = 0.0
        if query_vec is not None and all_embeddings is not None:
            sim = cosine_similarity(query_vec, all_embeddings[cid])
            if sim < EXPANSION_SIM_THRESHOLD:
                continue

        segment_bonus = 2.0 if sid not in selected_segments else 0.0
        score = len(overlap) + segment_bonus + sim
        scored.append((score, cid, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    added = 0
    expansion_threshold = 2.5  # Only add if score > threshold (entity overlap + similarity)
    for score, cid, chunk in scored:
        if added >= max_add:
            break
        # Filter: only add expanded chunks with strong relevance signal
        if score < expansion_threshold:
            continue
        expanded = dict(chunk)
        expanded["expansion"] = True
        expanded["expansion_score"] = score
        expanded["faiss_idx"] = cid
        selected_chunks.append(expanded)
        selected_ids.add(cid)
        added += 1

    return selected_chunks
