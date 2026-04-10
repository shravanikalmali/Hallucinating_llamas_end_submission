"""
Re-Ranking — Global MMR + Chunk Neighbour Inclusion

Maximal Marginal Relevance re-ranking with adaptive λ,
summary-bias, final-segment forcing, and neighbour expansion.
"""

import numpy as np

from src.config import (
    DEFAULT_MMR_LAMBDA, DEFAULT_FINAL_COUNT,
    SUMMARY_BIAS, FACTOID_QTYPES,
)
from src.utils import cosine_similarity


def global_mmr(
    query_vec: np.ndarray, candidates: list[dict],
    all_embeddings: np.ndarray, all_chunks: list[dict],
    lam: float = DEFAULT_MMR_LAMBDA, final_count: int = DEFAULT_FINAL_COUNT,
    qtype: str = "general", qsubtype: str = "default",
    graph_context: dict | None = None,
    event_context: dict | None = None  # New
) -> list[dict]:
    """Graph-Aware Maximal Marginal Relevance.

    Classic MMR (Relevance vs Diversity) enhanced with narrative signals:
    - Event Coverage: Reward chunks that introduce new narrative events.
    - Anchor Connectivity: Reward chunks that have paths to query anchor entities.
    - Path Strength: Reward chunks with validated NarCo edges to already selected chunks.
    """
    if not candidates:
        return []

    is_factoid = qtype in FACTOID_QTYPES
    # For QA: keep lambda high (0.7+) to maintain relevant chunks
    # Only factoid questions get slight boost to relevance
    effective_lam = min(lam + 0.05, 1.0) if is_factoid else lam
    if qsubtype == "narrator":
        effective_lam = lam  # Use standard lambda for narrator questions
    bias = 1.0 if is_factoid else SUMMARY_BIAS

    max_segment = max(c["segment_id"] for c in all_chunks) if all_chunks else 0

    # ── Graph Context Setup ──────────────────────────────────────────
    G = graph_context.get("graph") if graph_context else None
    narco_map = graph_context.get("narco_map", {}) if graph_context else {}
    query_anchors = graph_context.get("query_anchors", set()) if graph_context else set()

    # ── Reranking Logic ──────────────────────────────────────────────
    first_chunk_per_seg: dict[int, int] = {}
    for c in candidates:
        sid, cid = c["segment_id"], c["chunk_id"]
        if sid not in first_chunk_per_seg or cid < first_chunk_per_seg[sid]:
            first_chunk_per_seg[sid] = cid

    selected: list[dict] = []
    selected_vecs: list[np.ndarray] = []
    selected_event_types: set[str] = set()
    selected_chunk_ids: set[int] = set()
    remaining = list(candidates)

    for _ in range(min(final_count, len(remaining))):
        best_chunk, best_mmr = None, -float("inf")
        for chunk in remaining:
            cid = chunk["chunk_id"]
            rel = chunk["score"]

            # 1. Traditional Biases
            if bias > 1.0 and cid == first_chunk_per_seg.get(chunk["segment_id"]):
                rel *= bias

            # 2. Graph-Aware Relevance Boosts (weighted additively, not multiplicatively)
            graph_boost = 0.0
            if G and cid in narco_map:
                # Anchor Connectivity: Does it connect to query targets?
                chunk_ents = set(narco_map[cid].get("entities", []))
                if chunk_ents & query_anchors:
                    graph_boost += 0.3

                # NarCo Connectivity: Does it have edges to chunks already selected?
                for scid in selected_chunk_ids:
                    if (scid, cid) in narco_map.get("edges", {}) or (cid, scid) in narco_map.get("edges", {}):
                        graph_boost += 0.15

            # 3. Diversity Calculation (segment-aware)
            # Only penalize similarity within SAME segment (remove scene duplicates)
            # Allow diverse chunks from DIFFERENT segments (preserve narrative progression)
            div = 0.0
            if selected_vecs:
                c_vec = all_embeddings[chunk["faiss_idx"]]
                # Only compute diversity penalty for chunks in the same segment
                segment_selected = [sv for i, sv in enumerate(selected_vecs)
                                   if list(selected_chunk_ids)[i] in
                                   {s_c["chunk_id"] for s_c in selected
                                    if s_c.get("segment_id") == chunk["segment_id"]}]
                if segment_selected:
                    div = max(cosine_similarity(c_vec, sv) for sv in segment_selected)
                # Chunks from different segments get div=0 (no penalty)

            # 4. Event Coverage Boost using EventManager data
            event_boost = 0.0
            if event_context:
                chunk_events = event_context["chunk_to_events"].get(cid, [])
                new_events = [ev for ev in chunk_events if ev["type"] not in selected_event_types]
                if new_events:
                    # Reward chunks that explain new types of narrative movements
                    event_boost = 0.4 * len(new_events)

            # 5. Proper MMR: combine all relevance signals additively, then apply lambda weighting
            rel_total = rel + 0.3 * event_boost + 0.2 * graph_boost
            mmr = effective_lam * rel_total - (1 - effective_lam) * div
            if mmr > best_mmr:
                best_mmr = mmr
                best_chunk = chunk

        if best_chunk is not None:
            bc = dict(best_chunk)
            bc["mmr_score"] = best_mmr
            selected.append(bc)
            selected_vecs.append(all_embeddings[bc["faiss_idx"]])
            selected_chunk_ids.add(bc["chunk_id"])

            if event_context:
                chunk_events = event_context["chunk_to_events"].get(bc["chunk_id"], [])
                for ev in chunk_events:
                    selected_event_types.add(ev["type"])

            remaining = [c for c in remaining if c["faiss_idx"] != bc["faiss_idx"]]

    # Force final-segment inclusion
    has_final = any(c["segment_id"] == max_segment for c in selected)
    if not has_final:
        final_cands = sorted(
            [c for c in candidates if c["segment_id"] == max_segment],
            key=lambda c: c["score"], reverse=True,
        )
        if final_cands:
            fc = dict(final_cands[0])
            fc["mmr_score"] = fc["score"]
            fc["forced_final"] = True
            if len(selected) >= final_count:
                selected[-1] = fc
            else:
                selected.append(fc)

    # Narrator questions benefit from early narrative coverage.
    if qsubtype == "narrator" and all_chunks:
        min_segment = min(c["segment_id"] for c in all_chunks)
        has_early = any(c.get("segment_id") == min_segment for c in selected)
        if not has_early:
            early_cands = sorted(
                [c for c in candidates if c.get("segment_id") == min_segment],
                key=lambda c: c.get("score", 0.0),
                reverse=True,
            )
            if early_cands:
                ec = dict(early_cands[0])
                ec["mmr_score"] = ec.get("score", 0.0)
                ec["forced_early"] = True
                if len(selected) >= final_count:
                    selected[-1] = ec
                else:
                    selected.append(ec)

    selected.sort(key=lambda c: c["mmr_score"], reverse=True)
    return selected


def include_neighbors(selected_chunks: list[dict], all_chunks: list[dict]) -> list[dict]:
    """Add chunk_id ± 1 neighbours of every selected chunk for context."""
    selected_ids = {c["chunk_id"] for c in selected_chunks}
    chunk_map = {c["chunk_id"]: c for c in all_chunks}

    neighbors: list[dict] = []
    for c in selected_chunks:
        cid = c["chunk_id"]
        for nid in (cid - 1, cid + 1):
            if nid not in selected_ids and nid in chunk_map:
                nc = dict(chunk_map[nid])
                nc["neighbor_of"] = cid
                nc["faiss_idx"] = nid
                neighbors.append(nc)
                selected_ids.add(nid)

    return selected_chunks + neighbors
