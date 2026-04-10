"""
Graph-Based Reasoning

Builds a query-specific dynamic graph from the selected chunks and
traverses it by question type (who/where/when/what/why) to directly
extract an answer entity or narrow the evidence chunks.
"""

import re
import json
import os
import hashlib
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import Optional

from src.utils import load_json, cosine_similarity_safe


# ── Relation Type Detection ──────────────────────────────────────────────────

CAUSAL_KEYWORDS = {
    "cause", "caused", "because", "lead", "led", "corrupt", "corrupted",
    "influence", "influenced", "turn", "turned", "make", "made", "force", "forced",
    "reason", "due to", "since", "as a result", "trigger", "triggered", "motivate"
}

CONFLICT_KEYWORDS = {
    "kill", "killed", "murder", "fight", "fought", "battle", "war", "enemy",
    "enemies", "attack", "attacked", "harm", "harmed", "hurt", "wound", "conflict",
    "enemy", "hate", "hated", "betray", "betrayed", "rival", "villains", "evil"
}

SPATIAL_KEYWORDS = {
    "live", "lives", "lived", "living", "located", "locate", "location",
    "travel", "traveled", "travels", "visit", "visited", "visits", "go", "gone",
    "place", "region", "land", "home", "stay", "stayed", "stay", "at", "in"
}

TEMPORAL_KEYWORDS = {
    "after", "before", "when", "during", "while", "then", "later", "earlier",
    "first", "finally", "next", "eventually", "suddenly", "immediately", "once"
}


def _infer_relation_type(edge: dict, chunk_text: str | None = None) -> dict:
    """Infer relation type from edge data and associated chunk text hints.

    Returns dict with:
        - primary: "causal" | "conflict" | "spatial" | "temporal" | "cooccurrence"
        - confidence: 0.0-1.0
        - weight_bonus: extra score multiplier
    """
    segments = edge.get("segments", [])
    weight = float(edge.get("weight", 1.0))
    text = (chunk_text or "").lower()

    keyword_sets = [
        ("causal", CAUSAL_KEYWORDS, 0.75, 2.0),
        ("conflict", CONFLICT_KEYWORDS, 0.7, 1.8),
        ("spatial", SPATIAL_KEYWORDS, 0.65, 1.4),
        ("temporal", TEMPORAL_KEYWORDS, 0.6, 1.25),
    ]

    best_primary = "cooccurrence"
    best_confidence = 0.3
    best_bonus = 1.0

    for primary, keywords, base_confidence, bonus in keyword_sets:
        hit_count = sum(1 for kw in keywords if kw in text)
        if hit_count:
            confidence = min(0.95, base_confidence + 0.05 * min(hit_count, 3))
            if confidence > best_confidence:
                best_primary = primary
                best_confidence = confidence
                best_bonus = bonus

    # Default: pure co-occurrence
    result = {"primary": best_primary, "confidence": best_confidence, "weight_bonus": best_bonus}

    # Check for high co-occurrence patterns (multiple segments with same pair)
    if best_primary == "cooccurrence" and len(segments) >= 3:
        result["confidence"] = 0.6

    # Look for relation hints in segment proximity (simple heuristic)
    # If entities appear close in graph (many shared segments), likely spatial or ongoing
    if len(segments) >= 2:
        result["weight_bonus"] = 1.5

    return result


def _score_relation_for_question(
    neighbor: str, edge: dict, target: str, qtype: str,
    entity_labels: dict[str, set[str]], question_text: str
) -> float:
    """Score an edge for a given question type.

    Returns composite score incorporating:
        - RELATION TYPE from static graph (not question keywords!)
        - NER label matching
        - co-occurrence weight
        - recency (latest segment)
    """
    weight = float(edge.get("weight", 1.0))
    segments = edge.get("segments", [])
    max_seg = max(segments) if segments else 0

    # CRITICAL: Check actual edge relation type, not question keywords
    edge_relation = edge.get("relation", "cooccurrence").lower()

    base_score = weight
    label_bonus = 0.0
    relation_bonus = 0.0
    recency_bonus = max_seg * 0.1

    neighbor_labels = entity_labels.get(neighbor, set())

    # WHO questions: boost PERSON/ORG + conflict/causal edge types
    if qtype == "who":
        if neighbor_labels & {"PERSON", "ORG"}:
            label_bonus = 2.0
        # Boost edges that represent actual conflicts/causality
        if edge_relation in {"conflict", "fight", "kill", "murder", "hate", "betray"}:
            relation_bonus = 3.0
        elif edge_relation in {"family", "friend", "alliance"}:
            relation_bonus = 1.0

    # WHERE questions: boost GPE/LOC/FAC + spatial edge relations
    elif qtype == "where":
        if neighbor_labels & {"GPE", "LOC", "FAC"}:
            label_bonus = 2.5
        if edge_relation in {"located_in", "lives_in", "travels_to", "visits", "origin"}:
            relation_bonus = 2.5
        elif edge_relation in {"cooccurrence"}:
            relation_bonus = 0.5

    # WHY questions: HEAVILY boost causal/conflict relations from static graph
    elif qtype == "why":
        if edge_relation in {"cause", "caused", "corrupt", "corrupted", "influence", "influenced"}:
            relation_bonus = 4.0  # Strongest boost for true causality
        elif edge_relation in {"conflict", "fight", "hate", "betray"}:
            relation_bonus = 3.0
        elif edge_relation in {"family", "friend", "alliance"}:
            relation_bonus = 1.0

    return base_score + label_bonus + relation_bonus + recency_bonus


def _has_temporal_signals(text: str) -> bool:
    """Check if text contains temporal references."""
    text_lower = text.lower()
    # Check for temporal keywords
    if any(kw in text_lower for kw in TEMPORAL_KEYWORDS):
        return True
    # Check for years (YYYY format)
    if re.search(r"\b\d{4}\b", text):
        return True
    # Check for date patterns
    if re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", text, re.I):
        return True
    return False


def _get_entity_labels_map(
    entities_data: list[dict] | None, graph_nodes: set[str],
) -> dict[str, set[str]]:
    """Map graph node names to their NER labels via case-insensitive matching."""
    if not entities_data:
        return {}
    label_map: dict[str, set[str]] = defaultdict(set)
    raw_entries = [(ent["text"].lower(), ent["label"])
                   for entry in entities_data for ent in entry.get("entities", [])]
    nodes_lower = {n: n.lower() for n in graph_nodes}
    for node, nl in nodes_lower.items():
        for raw_text, label in raw_entries:
            if raw_text == nl or raw_text in nl or nl in raw_text:
                label_map[node].add(label)
    return dict(label_map)


def _infer_target_type_from_labels(labels: set[str]) -> str:
    """Map NER labels to a coarse target type."""
    if labels & {"PERSON"}:
        return "PERSON"
    if labels & {"GPE", "LOC", "FAC"}:
        return "LOCATION"
    if labels & {"DATE", "TIME"}:
        return "TIME"
    if labels & {"ORG"}:
        return "ORG"
    return "ENTITY"


def _detect_relation_hint(question: str, qtype: str) -> str | None:
    """Extract question-level relation hint used for pattern filtering."""
    q = question.lower()

    relation_keywords = {
        "marriage": {"marry", "married", "wife", "husband", "wed", "wedding", "spouse"},
        "family": {"father", "mother", "brother", "sister", "son", "daughter", "uncle", "aunt", "grandfather", "grandmother"},
        "conflict": {"fight", "fought", "war", "enemy", "kill", "killed", "murder", "betray", "hate"},
        "causal": {"why", "because", "reason", "caused", "cause", "led to", "due to", "since"},
        "location": {"where", "located", "live", "lived", "from", "in"},
        "time": {"when", "year", "before", "after", "during"},
        "identity": {"who", "name", "real name", "proper name"},
    }

    for relation, kws in relation_keywords.items():
        if any(kw in q for kw in kws):
            return relation

    if qtype == "why":
        return "causal"
    if qtype == "where":
        return "location"
    if qtype == "when":
        return "time"
    if qtype == "who":
        return "identity"
    return None


def _build_question_constraints(
    question: str,
    qtype: str,
    target_entities: list[str],
    entity_labels: dict[str, set[str]],
) -> dict:
    """Convert question into graph constraints for EPR-style filtering."""
    target_entity = target_entities[0] if target_entities else None
    target_type = None
    if target_entity is not None:
        target_type = _infer_target_type_from_labels(entity_labels.get(target_entity, set()))

    constraints = {
        "target_entity": target_entity,
        "target_entities": target_entities,
        "target_type": target_type,
        "relation": _detect_relation_hint(question, qtype),
        "qtype": qtype,
    }
    return constraints


def _pattern_matches_constraints(pattern: dict, constraints: dict) -> bool:
    """Filter path patterns against question constraints."""
    if not pattern.get("path"):
        return False

    relation_hint = constraints.get("relation")
    relations = pattern.get("relations", [])

    if relation_hint == "causal":
        if not any(r in {"causal", "conflict", "cooccurrence"} for r in relations):
            return False
    elif relation_hint == "location":
        if not any(r in {"spatial", "located_in", "lives_in", "travels_to", "cooccurrence"} for r in relations):
            return False
    elif relation_hint == "time":
        if not any(r in {"temporal", "cooccurrence"} for r in relations):
            return False
    elif relation_hint in {"marriage", "family", "conflict", "identity"}:
        # For these, allow relation or fallback to co-occurrence if confidence is high.
        relation_ok = any(relation_hint in str(r) for r in relations)
        if not relation_ok:
            avg_conf = float(pattern.get("avg_confidence", 0.0))
            if avg_conf < 0.55:
                return False

    return True


def _extract_graph_patterns(
    G: nx.Graph,
    target_entities: list[str],
    max_hops: int = 3,
) -> list[dict]:
    """Extract multi-hop path patterns from query graph."""
    if not target_entities:
        return []

    primary = target_entities[0]
    if primary not in G:
        return []

    candidate_ends: set[str] = set(G.nodes())
    for t in target_entities[1:]:
        if t in G:
            candidate_ends.add(t)
    for nb in G.neighbors(primary):
        candidate_ends.add(nb)

    patterns: list[dict] = []
    for end in candidate_ends:
        if end == primary:
            continue
        try:
            paths = nx.all_simple_paths(G, source=primary, target=end, cutoff=max_hops)
            for path in paths:
                if len(path) < 2:
                    continue

                relations: list[str] = []
                edge_confidences: list[float] = []
                edge_weights: list[float] = []
                edge_recency: list[int] = []

                for i in range(len(path) - 1):
                    e = G[path[i]][path[i + 1]]
                    relations.append(str(e.get("relation", "cooccurrence")))
                    edge_confidences.append(float(e.get("relation_confidence", 0.3)))
                    edge_weights.append(float(e.get("weight", 1.0)) * float(e.get("weight_bonus", 1.0)))
                    segs = e.get("segments", [])
                    edge_recency.append(max(segs) if segs else 0)

                patterns.append({
                    "path": path,
                    "relations": relations,
                    "avg_confidence": float(np.mean(edge_confidences)) if edge_confidences else 0.0,
                    "edge_weight_sum": float(np.sum(edge_weights)) if edge_weights else 0.0,
                    "recency": max(edge_recency) if edge_recency else 0,
                    "path_length": len(path) - 1,
                })
        except nx.NetworkXNoPath:
            continue

    return patterns


def _score_graph_pattern(pattern: dict, G: nx.Graph) -> float:
    """Score path patterns using confidence + frequency + recency - length penalty."""
    path = pattern.get("path", [])
    if len(path) < 2:
        return -1.0

    node_freq = 0.0
    for node in path:
        node_freq += float(G.nodes[node].get("frequency", 1.0))

    edge_conf = float(pattern.get("avg_confidence", 0.0))
    edge_strength = float(pattern.get("edge_weight_sum", 0.0))
    recency = float(pattern.get("recency", 0.0))
    path_len = float(pattern.get("path_length", len(path) - 1))

    score = (
        1.8 * edge_conf +
        0.35 * edge_strength +
        0.25 * node_freq +
        0.08 * recency -
        0.6 * path_len
    )
    return score


def _is_central_character_question(question: str) -> bool:
    """Detect questions asking for the main / central character."""
    q = question.lower()
    phrases = (
        "main character",
        "main protagonist",
        "protagonist",
        "central character",
        "who is this story about",
        "who is the story about",
        "who does this story follow",
    )
    return any(phrase in q for phrase in phrases)


def _score_central_character_candidates(
    static_graph: nx.Graph,
    entity_labels: dict[str, set[str]],
) -> list[tuple[str, float, dict]]:
    """Score book-wide candidates for central-character questions.

    Uses static-graph degree, segment spread, and PERSON label preference.
    """
    if static_graph is None or static_graph.number_of_nodes() == 0:
        return []

    degree_values = [float(static_graph.degree(n)) for n in static_graph.nodes()]
    max_degree = max(degree_values) if degree_values else 1.0
    max_segments = 1.0
    for _, data in static_graph.nodes(data=True):
        segs = str(data.get("segments", "")).split(",") if data.get("segments") else []
        max_segments = max(max_segments, float(len([s for s in segs if s.strip()])))

    scored: list[tuple[str, float, dict]] = []
    for node, data in static_graph.nodes(data=True):
        labels = entity_labels.get(node, set())
        segs = [s for s in str(data.get("segments", "")).split(",") if s.strip()]
        degree = float(static_graph.degree(node))
        degree_norm = degree / max_degree if max_degree > 0 else 0.0
        segment_norm = len(segs) / max_segments if max_segments > 0 else 0.0

        label_bonus = 0.0
        if "PERSON" in labels:
            label_bonus = 2.5
        elif "ORG" in labels:
            label_bonus = 1.0
        elif labels & {"GPE", "LOC", "FAC"}:
            label_bonus = -0.5
        else:
            label_bonus = -1.0

        score = (2.2 * degree_norm) + (1.6 * segment_norm) + label_bonus
        scored.append((node, score, {"degree": degree, "segments": len(segs), "labels": sorted(labels)}))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _detect_narrator_entity(
    selected_chunks: list[dict],
    chunk_ent_map: dict[int, list[str]],
    entity_labels: dict[str, set[str]],
) -> tuple[str | None, int | None]:
    """Detect likely narrator entity using first-person and early-timeline evidence."""
    pronoun_pattern = re.compile(r"\b(i|my|me|mine|myself)\b", re.I)
    scores: dict[str, float] = defaultdict(float)
    best_chunk_for_entity: dict[str, int] = {}

    for chunk in selected_chunks:
        cid = chunk["chunk_id"]
        entities = chunk_ent_map.get(cid, [])
        if not entities:
            continue

        text = chunk.get("text", "")
        first_person = bool(pronoun_pattern.search(f" {text} "))
        timeline = float(chunk.get("timeline", 0.5))
        early_bonus = 1.0 - timeline

        for ent in entities:
            score = early_bonus
            if first_person:
                score += 2.0
            if "PERSON" in entity_labels.get(ent, set()):
                score += 0.6
            scores[ent] += score
            if ent not in best_chunk_for_entity:
                best_chunk_for_entity[ent] = cid

    if not scores:
        return None, None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_entity = ranked[0][0]
    return best_entity, best_chunk_for_entity.get(best_entity)


def extract_target_entities(
    question: str, graph_nodes: list[str], model=None,
) -> list[str]:
    """Find graph-node names relevant to the question.

    Uses embedding similarity when a model is available (more robust),
    falling back to substring matching otherwise.
    """
    # ── Embedding-based matching (preferred) ──────────────────────────────
    if model is not None and graph_nodes:
        try:
            q_vec = model.encode(
                ["search_query: " + question],
                convert_to_numpy=True, normalize_embeddings=True,
            )[0]
            node_vecs = model.encode(
                ["search_document: " + n for n in graph_nodes],
                convert_to_numpy=True, normalize_embeddings=True,
            )
            scores = node_vecs @ q_vec
            scored = [(node, float(scores[i]))
                      for i, node in enumerate(graph_nodes)]
            scored.sort(key=lambda x: x[1], reverse=True)
            # Take nodes with similarity > 0.3 (at least top-1)
            targets = [n for n, s in scored if s > 0.3]
            if not targets and scored:
                targets = [scored[0][0]]
            return targets
        except Exception:
            pass  # Fall through to regex fallback

    # ── Regex / substring fallback ────────────────────────────────────────
    q_lower = question.lower()
    targets = []
    for node in graph_nodes:
        pattern = r"\b" + re.escape(node.lower()) + r"\b"
        if re.search(pattern, q_lower):
            targets.append(node)
    # Token subset matching
    if not targets:
        q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
        for node in graph_nodes:
            n_tokens = set(re.findall(r"\b\w+\b", node.lower()))
            if n_tokens and n_tokens.issubset(q_tokens):
                targets.append(node)
    targets.sort(key=len, reverse=True)
    return targets


def build_dynamic_graph(
    selected_chunks: list[dict], book_dir: Path,
    entities_data: list[dict] | None = None,
) -> tuple[nx.Graph, dict[str, set[str]]]:
    """Build a query-specific dynamic graph from the selected chunks.

    Uses chunk-local co-occurrence plus lightweight relation typing from
    keyword cues, and stores temporal state on nodes for query-time reasoning.
    """
    filtered_ents_file = book_dir / "filtered_chunk_entities.json"
    static_graph_file = book_dir / "graph.graphml"

    if not filtered_ents_file.exists():
        return nx.Graph(), {}

    chunk_entities = load_json(filtered_ents_file)
    chunk_ent_map = {item["chunk_id"]: item["entities"] for item in chunk_entities}
    chunk_text_map = {chunk["chunk_id"]: chunk.get("text", "") for chunk in selected_chunks}
    segment_chunks: dict[int, list[dict]] = defaultdict(list)
    for chunk in selected_chunks:
        segment_chunks[chunk["segment_id"]].append(chunk)

    seg_entities: dict[int, set[str]] = defaultdict(set)
    all_entities: set[str] = set()
    entity_segments: dict[str, set[int]] = defaultdict(set)
    entity_frequency: dict[str, int] = defaultdict(int)

    for chunk in selected_chunks:
        cid, sid = chunk["chunk_id"], chunk["segment_id"]
        for e in chunk_ent_map.get(cid, []):
            seg_entities[sid].add(e)
            all_entities.add(e)
            entity_segments[e].add(sid)
            entity_frequency[e] += 1

    if not all_entities:
        return nx.Graph(), {}

    entity_labels = _get_entity_labels_map(entities_data, all_entities)
    G = nx.Graph()

    for entity in all_entities:
        segs = sorted(entity_segments.get(entity, set()))
        G.add_node(
            entity,
            segments=segs,
            frequency=int(entity_frequency.get(entity, 0)),
            first_seen=segs[0] if segs else None,
            last_seen=segs[-1] if segs else None,
            recency=segs[-1] if segs else 0,
            labels=list(entity_labels.get(entity, set())),
        )

    # Co-occurrence edges with relation type enrichment
    for sid, ents in seg_entities.items():
        ents_list = sorted(ents)
        # Keep a temporal snapshot for this segment on each participating node.
        for ent in ents_list:
            node_segments = G.nodes[ent].setdefault("segment_history", [])
            if sid not in node_segments:
                node_segments.append(sid)

        related_chunks = segment_chunks.get(sid, [])
        if not related_chunks:
            continue

        for i in range(len(ents_list)):
            for j in range(i + 1, len(ents_list)):
                e1, e2 = ents_list[i], ents_list[j]
                relation_counts: dict[str, int] = defaultdict(int)
                best_relation = {"primary": "cooccurrence", "confidence": 0.3, "weight_bonus": 1.0}
                chunk_ids: list[int] = []

                for chunk in related_chunks:
                    relation_info = _infer_relation_type({"segments": [sid], "weight": 1.0}, chunk.get("text", ""))
                    relation_counts[relation_info["primary"]] += 1
                    chunk_ids.append(chunk["chunk_id"])
                    if relation_info["confidence"] >= best_relation["confidence"]:
                        best_relation = relation_info

                if G.has_edge(e1, e2):
                    edge = G[e1][e2]
                    edge["weight"] += len(related_chunks)
                    edge["segments"].append(sid)
                    edge.setdefault("chunk_ids", []).extend(chunk_ids)
                    existing_counts = edge.get("relation_counts", {})
                    if not isinstance(existing_counts, defaultdict):
                        existing_counts = defaultdict(int, existing_counts)
                    edge["relation_counts"] = existing_counts
                    for rel_name, rel_count in relation_counts.items():
                        edge["relation_counts"][rel_name] += rel_count

                    if best_relation["confidence"] >= edge.get("relation_confidence", 0.0):
                        edge["relation"] = best_relation["primary"]
                        edge["relation_confidence"] = best_relation["confidence"]
                    edge["weight_bonus"] = max(edge.get("weight_bonus", 1.0), best_relation["weight_bonus"])
                else:
                    G.add_edge(
                        e1, e2,
                        weight=len(related_chunks),
                        segments=[sid],
                        chunk_ids=chunk_ids,
                        relation=best_relation["primary"],
                        relation_confidence=best_relation["confidence"],
                        weight_bonus=best_relation["weight_bonus"],
                        relation_counts=dict(relation_counts),
                    )

    # Add lightweight temporal continuity on nodes for segment-to-segment reasoning.
    for entity, segs in entity_segments.items():
        ordered_segs = sorted(segs)
        G.nodes[entity]["segment_history"] = ordered_segs
        if len(ordered_segs) >= 2:
            G.nodes[entity]["temporal_span"] = ordered_segs[-1] - ordered_segs[0]

    # CRITICAL: Enrich with static graph relations (not just node data)
    if static_graph_file.exists():
        try:
            static_G = nx.read_graphml(str(static_graph_file))
            for node in G.nodes():
                if static_G.has_node(node):
                    sd = static_G.nodes[node]
                    if "segments" in sd:
                        G.nodes[node]["all_segments"] = [
                            int(s) for s in sd["segments"].split(",") if s.strip()
                        ]

            # Transfer relation types from static graph
            for e1, e2, data in static_G.edges(data=True):
                if G.has_edge(e1, e2):
                    relation_type = data.get("relation", "cooccurrence")
                    G[e1][e2]["relation"] = relation_type
                    # Increase weight for strong static relations
                    if relation_type in ["conflict", "family", "alliance"]:
                        G[e1][e2]["weight"] = max(G[e1][e2]["weight"], 2)
        except Exception:
            pass  # If static graph load fails, continue with co-occurrence only

    return G, entity_labels


def _maybe_save_query_graph(G: nx.Graph, book_dir: Path, question: str) -> Path | None:
    """Optionally persist the dynamic query graph to disk for debugging.

    Controlled by environment variable ``SAVE_QUERY_GRAPH``.
    Accepted true-ish values: ``1``, ``true``, ``yes``, ``on``.
    """
    enabled = os.getenv("SAVE_QUERY_GRAPH", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    out_dir = book_dir / "query_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    question_id = hashlib.md5(question.encode("utf-8")).hexdigest()[:12]
    out_path = out_dir / f"query_graph_{question_id}.graphml"

    def _graphml_safe_value(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple, set, dict)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    try:
        safe_G = nx.Graph()
        for node, attrs in G.nodes(data=True):
            safe_G.add_node(node, **{k: _graphml_safe_value(v) for k, v in attrs.items()})
        for u, v, attrs in G.edges(data=True):
            safe_G.add_edge(u, v, **{k: _graphml_safe_value(vv) for k, vv in attrs.items()})

        nx.write_graphml(safe_G, str(out_path))
        return out_path
    except Exception:
        return None


def collect_graph_hints(
    question: str,
    qtype: str,
    selected_chunks: list[dict],
    book_dir: Path,
    entities_data: list[dict] | None = None,
    model=None,
    qsubtype: str = "default",
    max_patterns: int = 5,
) -> dict:
    """Collect graph constraints and top-scoring path patterns.

    Used by Phase 4 Hybrid reasoning when graph extraction is inconclusive.
    """
    G, entity_labels = build_dynamic_graph(selected_chunks, book_dir, entities_data)
    if G.number_of_nodes() < 2:
        return {
            "graph_failed": True,
            "reason": "graph_too_small",
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "constraints": None,
            "target_entities": [],
            "top_patterns": [],
        }

    targets = extract_target_entities(question, list(G.nodes()), model=model)
    if not targets:
        return {
            "graph_failed": True,
            "reason": "no_target_entity",
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "constraints": None,
            "target_entities": [],
            "top_patterns": [],
        }

    constraints = _build_question_constraints(question, qtype, targets, entity_labels)
    patterns = _extract_graph_patterns(G, targets, max_hops=3)
    filtered = [p for p in patterns if _pattern_matches_constraints(p, constraints)]
    for p in filtered:
        p["score"] = _score_graph_pattern(p, G)
    filtered.sort(key=lambda x: x.get("score", -1.0), reverse=True)

    top_patterns = filtered[:max_patterns]
    best_score = float(top_patterns[0].get("score", -1.0)) if top_patterns else -1.0

    return {
        "graph_failed": not bool(top_patterns and best_score >= 1.15),
        "reason": "low_pattern_score" if top_patterns and best_score < 1.15 else ("no_valid_patterns" if not top_patterns else "ok"),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "constraints": constraints,
        "target_entities": targets,
        "top_patterns": top_patterns,
        "best_pattern_score": best_score,
    }


def graph_reasoning(
    question: str, qtype: str, selected_chunks: list[dict],
    all_chunks: list[dict], book_dir: Path,
    entities_data: list[dict] | None = None,
    model=None, use_cross_encoder: bool = True, verbose: bool = False,
    qsubtype: str = "default",
) -> dict | None:
    """Attempt graph-based reasoning before falling back to sentence extraction."""
    static_graph_file = book_dir / "graph.graphml"

    q_lower = question.lower()
    if qtype == "what" and any(kw in q_lower for kw in ("translate", "translated", "means", "symbolize", "symbolises", "symbolizes")):
        if verbose:
            print("    Definition/translation question detected, skipping graph narrowing")
        return None

    # Book-wide central-character path: use the offline graph, not just local chunks.
    if qtype == "who" and (qsubtype == "central_character" or _is_central_character_question(question)) and static_graph_file.exists():
        try:
            static_G = nx.read_graphml(str(static_graph_file))
            entity_labels = _get_entity_labels_map(entities_data, set(static_G.nodes()))
            scored = _score_central_character_candidates(static_G, entity_labels)
            if scored:
                answer_entity, answer_score, meta = scored[0]

                best_evidence = None
                for chunk in selected_chunks:
                    if answer_entity.lower() in chunk.get("text", "").lower():
                        best_evidence = chunk
                        break
                if best_evidence is None and selected_chunks:
                    best_evidence = max(selected_chunks, key=lambda c: c.get("score", 0.0))

                if verbose:
                    top_preview = [(n, f"{s:.2f}") for n, s, _ in scored[:5]]
                    print(f"    Central character candidates: {top_preview}")
                    print(f"    Selected central character: {answer_entity}")

                return {
                    "answer": answer_entity,
                    "answer_entity": answer_entity,
                    "answer_score": min(0.98, 0.55 + 0.15 * answer_score),
                    "source_chunk_id": best_evidence["chunk_id"] if best_evidence else -1,
                    "source_segment": best_evidence["segment_id"] if best_evidence else -1,
                    "context": best_evidence.get("text", "") if best_evidence else "",
                    "reasoning": "graph_centrality",
                    "graph_target": answer_entity,
                    "graph_nodes": static_G.number_of_nodes(),
                    "graph_edges": static_G.number_of_edges(),
                    "graph_confidence": min(0.95, 0.6 + 0.1 * min(meta.get("segments", 0), 5)),
                }
        except Exception:
            # Fall back to the local dynamic-graph path below.
            pass

    G, entity_labels = build_dynamic_graph(selected_chunks, book_dir, entities_data)
    saved_query_graph = _maybe_save_query_graph(G, book_dir, question)
    if verbose and saved_query_graph is not None:
        print(f"    Saved query graph → {saved_query_graph}")
    if G.number_of_nodes() < 2:
        if verbose:
            print("    Graph too small for reasoning, falling back")
        return None

    targets = extract_target_entities(question, list(G.nodes()), model=model)
    if not targets:
        if verbose:
            print("    No target entity found in question, falling back")
        return None

    constraints = _build_question_constraints(question, qtype, targets, entity_labels)

    # For identity/name questions, prioritize entities explicitly mentioned in the question.
    if constraints.get("relation") == "identity":
        q_lower_exact = question.lower()
        explicit_targets = []
        for node in G.nodes():
            pat = r"\b" + re.escape(node.lower()) + r"\b"
            if re.search(pat, q_lower_exact):
                explicit_targets.append(node)
        if explicit_targets:
            dedup = []
            for t in explicit_targets + targets:
                if t not in dedup:
                    dedup.append(t)
            targets = dedup
            constraints = _build_question_constraints(question, qtype, targets, entity_labels)

    if verbose:
        print(f"    Constraints: {constraints}")

    # Step 3.x: multi-hop pattern extraction + EPR-style filtering + scoring
    patterns = _extract_graph_patterns(G, targets, max_hops=3)
    if patterns:
        filtered_patterns = [p for p in patterns if _pattern_matches_constraints(p, constraints)]
    else:
        filtered_patterns = []

    for p in filtered_patterns:
        p["score"] = _score_graph_pattern(p, G)

    filtered_patterns.sort(key=lambda x: x.get("score", -1.0), reverse=True)

    best_pattern = filtered_patterns[0] if filtered_patterns else None
    if best_pattern is not None:
        best_pattern_score = float(best_pattern.get("score", -1.0))
        if verbose:
            print(
                f"    Best pattern: path={best_pattern.get('path')} "
                f"relations={best_pattern.get('relations')} score={best_pattern_score:.3f}"
            )

        # Confidence check before accepting graph answer
        if best_pattern_score >= 1.15:
            pattern_path = best_pattern.get("path", [])
            if len(pattern_path) >= 2:
                candidate_entity = pattern_path[-1]

                # For factoid entity questions, directly return graph answer entity.
                # Identity is handled by dedicated alias/name logic below to avoid
                # co-occurrence path hijacking (e.g., selecting a frequent neighbor).
                if qtype in {"who", "where"} or constraints.get("relation") in {"family", "marriage"}:
                    best_evidence = None
                    fce = book_dir / "filtered_chunk_entities.json"
                    chunk_ent_map: dict[int, list[str]] = {}
                    if fce.exists():
                        for item in load_json(fce):
                            chunk_ent_map[item["chunk_id"]] = item["entities"]
                    for chunk in selected_chunks:
                        ents = set(chunk_ent_map.get(chunk["chunk_id"], []))
                        if candidate_entity in ents:
                            best_evidence = chunk
                            break
                    if best_evidence is None and selected_chunks:
                        best_evidence = selected_chunks[0]

                    confidence = min(0.96, 0.55 + 0.07 * best_pattern_score)
                    return {
                        "answer": candidate_entity,
                        "answer_entity": candidate_entity,
                        "answer_score": confidence,
                        "source_chunk_id": best_evidence["chunk_id"] if best_evidence else -1,
                        "source_segment": best_evidence["segment_id"] if best_evidence else -1,
                        "context": best_evidence.get("text", "") if best_evidence else "",
                        "reasoning": "graph_pattern",
                        "graph_target": targets[0],
                        "graph_nodes": G.number_of_nodes(),
                        "graph_edges": G.number_of_edges(),
                        "graph_confidence": confidence,
                        "graph_pattern": best_pattern,
                        "constraints": constraints,
                    }
        else:
            if verbose:
                print(f"    graph_failed=True (low pattern score: {best_pattern_score:.3f})")
    elif verbose:
        print("    graph_failed=True (no valid patterns)")

    target = targets[0]
    if verbose:
        print(f"    Target entity: {target}")
        print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if target not in G:
        return None

    neighbors = list(G.neighbors(target))
    if not neighbors:
        if verbose:
            print(f"    Target '{target}' has no neighbours, falling back")
        return None

    # Identity/name questions (e.g., "real name", "proper name")
    # Prefer PERSON neighbors with explicit naming evidence.
    if constraints.get("relation") == "identity":
        # Alias form: "What is X's real/proper name?"
        alias_match = re.search(r"what\s+is\s+(.+?)'s\s+(?:real|proper)\s+name", q_lower)
        if alias_match:
            alias = alias_match.group(1).strip()
            alias_candidates: dict[str, float] = defaultdict(float)

            person_like_nodes = [
                n for n in G.nodes()
                if ("PERSON" in set(G.nodes[n].get("labels", [])) or "ORG" in set(G.nodes[n].get("labels", [])))
            ]

            for chunk in selected_chunks:
                text = chunk.get("text", "")
                text_lower = text.lower()
                if alias not in text_lower:
                    continue

                # Split for local matching and score nearby explicit names.
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                for sent in sentences:
                    sent_lower = sent.lower()
                    if alias not in sent_lower:
                        continue

                    # Strong apposition heuristic:
                    # choose PERSON/ORG node nearest before the alias mention.
                    alias_pos = sent_lower.find(alias)
                    if alias_pos >= 0:
                        nearest_node = None
                        nearest_pos = -1
                        for node in person_like_nodes:
                            nl = node.lower()
                            pos = sent_lower.rfind(nl, 0, alias_pos)
                            if pos > nearest_pos and nl != alias:
                                nearest_pos = pos
                                nearest_node = node
                        if nearest_node is not None:
                            alias_candidates[nearest_node] += 3.5

                    for node in person_like_nodes:
                        nl = node.lower()
                        if nl in sent_lower and nl != alias:
                            score = 1.0
                            if "real name" in sent_lower or "proper name" in sent_lower:
                                score += 2.5
                            if "\"" in sent or "'" in sent:
                                score += 0.4
                            alias_candidates[node] += score

            if alias_candidates:
                ranked_alias = sorted(alias_candidates.items(), key=lambda x: x[1], reverse=True)
                answer_entity = ranked_alias[0][0]
                best_evidence = None
                for chunk in selected_chunks:
                    tl = (chunk.get("text", "") or "").lower()
                    if alias in tl and answer_entity.lower() in tl:
                        best_evidence = chunk
                        break
                if best_evidence is None:
                    best_evidence = selected_chunks[0]

                confidence = min(0.98, 0.62 + 0.09 * ranked_alias[0][1])
                if verbose:
                    print(f"    Alias '{alias}' candidates: {[(c, f'{s:.2f}') for c, s in ranked_alias[:5]]}")

                return {
                    "answer": answer_entity,
                    "answer_entity": answer_entity,
                    "answer_score": confidence,
                    "source_chunk_id": best_evidence["chunk_id"],
                    "source_segment": best_evidence["segment_id"],
                    "context": best_evidence.get("text", ""),
                    "reasoning": "graph_identity_alias",
                    "graph_target": targets[0],
                    "graph_nodes": G.number_of_nodes(),
                    "graph_edges": G.number_of_edges(),
                    "graph_confidence": confidence,
                    "constraints": constraints,
                }

        naming_markers = ("real name", "proper name", "called", "known as", "is ", "was ")
        weak_name_markers = (" by name",)
        identity_candidates: list[tuple[str, float]] = []

        for nb in neighbors:
            labels = set(G.nodes[nb].get("labels", []))
            if "PERSON" not in labels and "ORG" not in labels:
                continue

            edge = G[target][nb]
            base = float(edge.get("weight", 1.0)) * float(edge.get("weight_bonus", 1.0))
            conf = float(edge.get("relation_confidence", 0.3))
            marker_bonus = 0.0

            for chunk in selected_chunks:
                text = (chunk.get("text", "") or "").lower()
                if target.lower() in text and nb.lower() in text:
                    if any(m in text for m in naming_markers):
                        marker_bonus += 1.5
                    if "real name" in q_lower or "proper name" in q_lower:
                        if "real name" in text or "proper name" in text:
                            marker_bonus += 2.0
                    if any(wm in text for wm in weak_name_markers):
                        marker_bonus -= 0.8

            identity_candidates.append((nb, base + conf + marker_bonus))

        if identity_candidates:
            identity_candidates.sort(key=lambda x: x[1], reverse=True)
            answer_entity = identity_candidates[0][0]
            best_evidence = None
            for chunk in selected_chunks:
                text = (chunk.get("text", "") or "").lower()
                if target.lower() in text and answer_entity.lower() in text:
                    best_evidence = chunk
                    break
            if best_evidence is None:
                best_evidence = selected_chunks[0]

            confidence = min(0.97, 0.6 + 0.08 * identity_candidates[0][1])
            if verbose:
                print(f"    Identity candidates: {[(c, f'{s:.2f}') for c, s in identity_candidates[:5]]}")

            return {
                "answer": answer_entity,
                "answer_entity": answer_entity,
                "answer_score": confidence,
                "source_chunk_id": best_evidence["chunk_id"],
                "source_segment": best_evidence["segment_id"],
                "context": best_evidence.get("text", ""),
                "reasoning": "graph_identity",
                "graph_target": target,
                "graph_nodes": G.number_of_nodes(),
                "graph_edges": G.number_of_edges(),
                "graph_confidence": confidence,
                "constraints": constraints,
            }

    # Load chunk-entity mapping
    chunk_ent_map: dict[int, list[str]] = {}
    fce = book_dir / "filtered_chunk_entities.json"
    if fce.exists():
        for item in load_json(fce):
            chunk_ent_map[item["chunk_id"]] = item["entities"]

    if qtype == "who" and qsubtype == "narrator":
        narrator_entity, narrator_cid = _detect_narrator_entity(selected_chunks, chunk_ent_map, entity_labels)
        if narrator_entity is not None:
            evidence = next((c for c in selected_chunks if c["chunk_id"] == narrator_cid), selected_chunks[0])
            if verbose:
                print(f"    Narrator candidate: {narrator_entity}")
            return {
                "answer": narrator_entity,
                "answer_entity": narrator_entity,
                "answer_score": 0.82,
                "source_chunk_id": evidence["chunk_id"],
                "source_segment": evidence["segment_id"],
                "context": evidence.get("text", ""),
                "reasoning": "graph_narrator",
                "graph_target": narrator_entity,
                "graph_nodes": G.number_of_nodes(),
                "graph_edges": G.number_of_edges(),
                "graph_confidence": 0.82,
            }

    answer_entity: str | None = None
    evidence_chunk_ids: list[int] = []

    # ── "who" — PERSON/ORG neighbour with relation-aware scoring ──────────────
    if qtype == "who":
        scored = []
        for nb in neighbors:
            labels = set(G.nodes[nb].get("labels", []))
            if not (labels & {"PERSON", "ORG"}):
                continue
            edge = G[target][nb]
            relation_score = _score_relation_for_question(
                nb, edge, target, qtype, entity_labels, question
            )
            segs = edge.get("segments", [])
            scored.append((nb, relation_score))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            answer_entity = scored[0][0]
            if verbose:
                print(f"    WHO candidates: {[(nb, f'{sc:.2f}') for nb, sc in scored[:3]]}")

    # ── "where" — GPE/LOC/FAC neighbour with relation-aware scoring ────────────
    elif qtype == "where":
        scored = []
        for nb in neighbors:
            labels = set(G.nodes[nb].get("labels", []))
            if not (labels & {"GPE", "LOC", "FAC"}):
                continue
            edge = G[target][nb]
            relation_score = _score_relation_for_question(
                nb, edge, target, qtype, entity_labels, question
            )
            scored.append((nb, relation_score))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            answer_entity = scored[0][0]
            if verbose:
                print(f"    WHERE candidates: {[(nb, f'{sc:.2f}') for nb, sc in scored[:3]]}")

    # ── "when" — target's segments with temporal signal boost ────────────────
    elif qtype == "when":
        target_segs = set(G.nodes[target].get("segments", []))
        chunk_ent_map_local = chunk_ent_map  # Use outer scope's map

        # Collect all chunks from target segments for sentence-level processing
        temporal_chunks = []
        for chunk in selected_chunks:
            if chunk["segment_id"] in target_segs:
                temporal_chunks.append(chunk["chunk_id"])

        evidence_chunk_ids = temporal_chunks

        if verbose:
            print(f"    WHEN: collected {len(evidence_chunk_ids)} chunks for temporal sentence ranking")

    # ── "what" / "narrative" — top 2-3 latest segments + related entities ──────
    elif qtype in ("what", "narrative"):
        target_segs = sorted(G.nodes[target].get("segments", []))
        if target_segs:
            # Focus on top 2-3 latest segments for continuity
            focus_segs = set(target_segs[-3:]) if len(target_segs) >= 3 else set(target_segs[-2:]) if len(target_segs) >= 2 else set(target_segs)

            for chunk in selected_chunks:
                if chunk["segment_id"] in focus_segs:
                    evidence_chunk_ids.append(chunk["chunk_id"])

            # Also include related entities from those segments
            for other in targets[1:]:
                if other in G and G.has_edge(target, other):
                    edge = G[target][other]
                    for seg in edge.get("segments", []):
                        if seg in focus_segs:
                            for chunk in selected_chunks:
                                if chunk["segment_id"] == seg and chunk["chunk_id"] not in evidence_chunk_ids:
                                    evidence_chunk_ids.append(chunk["chunk_id"])

            if verbose:
                print(f"    WHAT: focus segments {focus_segs}, collected {len(evidence_chunk_ids)} chunks")

    # ── "why" — causal/conflict relation-aware scoring from static graph ────────
    elif qtype == "why":
        edge_info = []

        for nb in neighbors:
            edge = G[target][nb]
            relation_score = _score_relation_for_question(
                nb, edge, target, qtype, entity_labels, question
            )
            segs = edge.get("segments", [])
            edge_info.append((nb, relation_score, segs))

        edge_info.sort(key=lambda x: x[1], reverse=True)

        if edge_info:
            top_neighbor, top_score, segs = edge_info[0]
            for seg in segs:
                for chunk in selected_chunks:
                    if chunk["segment_id"] == seg:
                        evidence_chunk_ids.append(chunk["chunk_id"])

            if verbose:
                print(f"    WHY: top neighbor '{top_neighbor}' (score={top_score:.2f}), "
                      f"candidates: {[(nb, f'{sc:.2f}') for nb, sc, _ in edge_info[:3]]}")

    # ── Direct entity answer (who/where) ──────────────────────────────────
    if answer_entity:
        best_evidence = None
        for chunk in selected_chunks:
            ents = set(chunk_ent_map.get(chunk["chunk_id"], []))
            if answer_entity in ents and target in ents:
                best_evidence = chunk
                break
        if not best_evidence:
            for chunk in selected_chunks:
                if answer_entity in set(chunk_ent_map.get(chunk["chunk_id"], [])):
                    best_evidence = chunk
                    break
        if not best_evidence:
            best_evidence = selected_chunks[0]

        text = best_evidence.get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        best_sent, best_score = text[:500], -1
        for sent in sentences:
            score = 0
            if answer_entity.lower() in sent.lower():
                score += 2
            if target.lower() in sent.lower():
                score += 1
            if score > best_score:
                best_score = score
                best_sent = sent

        if verbose:
            print(f"    Graph answer: {answer_entity}")

        # Compute confidence based on neighbor count and edge strength
        neighbors_count = len(neighbors)
        edge_strength = edge.get("weight", 1) if answer_entity else 1
        confidence = min(0.95, 0.5 + 0.1 * min(neighbors_count, 5) + 0.15 * min(edge_strength / 3, 1))

        final_answer = answer_entity if (qtype == "who" and qsubtype in {"narrator", "central_character", "entity_identity"}) else best_sent

        return {
            "answer": final_answer, "answer_entity": answer_entity,
            "answer_score": confidence, "source_chunk_id": best_evidence["chunk_id"],
            "source_segment": best_evidence["segment_id"],
            "context": text, "reasoning": "graph",
            "graph_target": target,
            "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
            "graph_confidence": confidence,
        }

    # ── Evidence chunk narrowing (what/when/why) ──────────────────────────
    if evidence_chunk_ids:
        evidence_set = set(evidence_chunk_ids)
        evidence_chunks = [c for c in selected_chunks if c["chunk_id"] in evidence_set]
        if evidence_chunks:
            if verbose:
                print(f"    Graph narrowed to {len(evidence_chunks)} evidence chunks")
            from src.retrieval.answer import extract_answer
            # Pass qtype to extract_answer for context-aware processing (e.g., temporal sentence prioritization)
            answer_result = extract_answer(question, evidence_chunks, model,
                                           use_cross_encoder=use_cross_encoder,
                                           qtype=qtype, qsubtype=qsubtype)

            # Compute graph confidence based on graph structure
            graph_confidence = min(0.85, 0.4 + 0.1 * len(targets))

            # Combine cross-encoder score with graph confidence for final answer score
            cross_encoder_score = answer_result.get("answer_score", 0.5)
            combined_confidence = 0.6 * cross_encoder_score + 0.4 * graph_confidence

            answer_result["reasoning"] = "graph_guided"
            answer_result["graph_target"] = target
            answer_result["graph_nodes"] = G.number_of_nodes()
            answer_result["graph_edges"] = G.number_of_edges()
            answer_result["graph_confidence"] = graph_confidence
            answer_result["cross_encoder_score"] = cross_encoder_score  # Keep original for transparency
            answer_result["answer_score"] = combined_confidence  # Updated final score

            # If confidence too low, allow fallback by returning None
            if combined_confidence < 0.35:
                if verbose:
                    print(f"    Combined confidence too low ({combined_confidence:.2f}), triggering fallback")
                return None

            return answer_result

    if verbose:
        print("    Graph reasoning inconclusive, falling back")
    return None
