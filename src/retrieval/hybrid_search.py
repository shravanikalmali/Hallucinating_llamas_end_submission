"""
Hybrid Search — Parallel Candidate Generation (Dense + BM25 + Entity)

Combines dense vector retrieval, sparse BM25 scoring, and direct entity lookup.
Implements parallel candidate generation to maximize recall.
"""

import re
import math
import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer

from src.config import (
    MODEL_NAME, QTYPE_ENTITY_LABELS,
    DEFAULT_TOP_K, PROCESSED_DIR,
    DENSE_WEIGHT, BM25_WEIGHT, ENTITY_WEIGHT,
)
from src.utils import tokenize, load_json


# ── Model Cache ───────────────────────────────────────────────────────────────

_model_cache: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _model_cache


# ── Loading ───────────────────────────────────────────────────────────────────

def load_book_index(book_id: str, processed_dir: Path = PROCESSED_DIR):
    """Load the FAISS index and chunk metadata for a specific book."""
    book_dir = processed_dir / book_id
    faiss_path = book_dir / f"{book_id}.faiss"
    pkl_path = book_dir / f"{book_id}_chunks.pkl"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Chunk metadata not found: {pkl_path}")

    index = faiss.read_index(str(faiss_path))
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)

    assert index.ntotal == len(chunks), (
        f"Index/metadata mismatch: {index.ntotal} vectors vs {len(chunks)} chunks"
    )
    return index, chunks


def load_entities(book_dir: Path) -> list[dict] | None:
    """Load raw entities.json if available."""
    ents_file = book_dir / "entities.json"
    if not ents_file.exists():
        return None
    with open(ents_file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_entity_index(book_dir: Path) -> dict:
    """Load entity_index.json if available."""
    index_file = book_dir / "entity_index.json"
    if not index_file.exists():
        return {}
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_narrative_context(book_dir: Path) -> dict:
    """Load the narrative graph and NarCo edges for retrieval context."""
    graph_path = book_dir / "graph.graphml"
    narco_path = book_dir / "narco_edges.json"
    
    context = {
        "graph": None,
        "narco_edges": [],
        "narco_map": {}
    }
    
    # 1. Load GraphML
    if graph_path.exists():
        import networkx as nx
        try:
            context["graph"] = nx.read_graphml(str(graph_path))
        except Exception:
            pass
            
    # 2. Load NarCo edges
    if narco_path.exists():
        context["narco_edges"] = load_json(narco_path)
        
        # Build quick-lookup map for chunks
        n_map = {}
        for edge in context["narco_edges"]:
            sid, tid = edge["source_idx"], edge["target_idx"]
            if tid not in n_map: n_map[tid] = {"edges": {}, "events": [], "entities": []}
            n_map[tid]["edges"][sid] = edge["questions"]
            
        # Add node details if graph is loaded
        if context["graph"]:
            for node, data in context["graph"].nodes(data=True):
                segs = data.get("segments", "").split(",")
                for s in segs:
                    if not s: continue
                    chunk_id = int(s) # Assuming segment mapping was chunk_id
                    if chunk_id not in n_map: n_map[chunk_id] = {"edges": {}, "events": [], "entities": []}
                    if data.get("type") == "event":
                        n_map[chunk_id]["events"].append(node)
                    else:
                        n_map[chunk_id]["entities"].append(node)
        context["narco_map"] = n_map
        
    return context


# ── Question Embedding ────────────────────────────────────────────────────────

def embed_question(question: str, model: SentenceTransformer = None) -> np.ndarray:
    """Encode a question with 'search_query:' prefix. Returns unit-normalised 1-D vector."""
    if model is None:
        model = get_model()
    prefixed = "search_query: " + question
    vec = model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0].astype(np.float32)


# ── Question Type Detection ──────────────────────────────────────────────────

def detect_question_type_info(question: str) -> tuple[str, str]:
    """Classify question into (qtype, qsubtype) for adaptive retrieval."""
    q = question.lower().strip().rstrip("?").strip()

    # ── Coarse type ─────────────────────────────────────────────────────
    if "in what year" in q or "what year" in q or "which year" in q:
        qtype = "when"
    elif "how many years" in q or "how long" in q or "how many year" in q:
        qtype = "duration"
    elif q.startswith("where") or any(x in f" {q} " for x in [" where ", "in what place", "what country", "what city", "what place"]):
        qtype = "where"
    elif q.startswith("who") or " who " in f" {q} ":
        qtype = "who"
    elif q.startswith("when") or " when " in f" {q} ":
        qtype = "when"
    elif q.startswith("why"):
        qtype = "why"
    elif q.startswith("how"):
        qtype = "how"
    elif any(q.startswith(p) for p in ("what happens", "what occurred", "what took place")):
        qtype = "narrative"
    elif q.startswith("what"):
        qtype = "what"
    else:
        qtype = "general"

    # ── Subtype refinement ──────────────────────────────────────────────
    qsubtype = "default"

    if qtype == "who":
        if any(x in q for x in ("teller", "narrator", "who tells", "point of view", "narrates")):
            qsubtype = "narrator"
        elif any(x in q for x in ("main character", "protagonist", "central character")):
            qsubtype = "central_character"
        else:
            qsubtype = "entity_identity"

    elif qtype == "why":
        if any(x in q for x in ("cause", "caused", "reason", "why did", "why was", "war office", "failed", "fail")):
            qsubtype = "causal_event"
        else:
            qsubtype = "explanatory"

    elif qtype == "when":
        if "year" in q:
            qsubtype = "year"
        else:
            qsubtype = "temporal_event"

    elif qtype == "duration":
        qsubtype = "duration"

    elif qtype == "what":
        if any(x in q for x in ("real name", "proper name", "true name", "identity of", "who is known as", "known as", "species", "birds", "animal", "creature")):
            qsubtype = "identity"
        elif any(x in q for x in ("translate", "translated", "means", "meaning", "symbol", "symbolize", "symbolises", "symbolizes")):
            qsubtype = "definition"
        else:
            qsubtype = "description"

    # Temporal cue detection
    temporal_cue = None
    anchor_event = None
    
    # 1. Relative temporal anchors: "after the trial", "before the death"
    # Skip common pronouns to find the actual event/entity
    match = re.search(r"(after|before|since|following)\s+(the\s+|his\s+|her\s+|their\s+|he\s+|she\s+|it\s+)?(\w+)", q)
    if match:
        temporal_cue = match.group(1)
        # If the word is 'is' or 'was', look further? 
        # For "after he is acquitted", group(3) might be "is".
        # Let's be more robust: find the first non-stopword noun-ish thing.
        potential_anchor = match.group(3)
        from src.config import STOP_WORDS
        if potential_anchor in STOP_WORDS or potential_anchor in ("is", "was", "be", "been"):
            # Try a broader search for events in the rest of the question
            for word in q[match.end():].split():
                if word.lower() not in STOP_WORDS:
                    anchor_event = word.lower()
                    break
        else:
            anchor_event = potential_anchor
    
    # 2. General temporal markers
    if not temporal_cue:
        if "after" in q or "later" in q or "following" in q:
            temporal_cue = "after"
        elif "before" in q or "earlier" in q or "prior" in q:
            temporal_cue = "before"
        elif "end" in q or "finally" in q or "at the last" in q:
            temporal_cue = "end"
        elif "beginning" in q or "start" in q or "initially" in q:
            temporal_cue = "start"

    return qtype, qsubtype, temporal_cue, anchor_event


def detect_question_type(question: str) -> str:
    """Backward-compatible wrapper returning only coarse question type."""
    qtype, _ = detect_question_type_info(question)
    return qtype


# ── BM25 Scoring ─────────────────────────────────────────────────────────────

def compute_bm25_scores(chunks: list[dict], question: str,
                        k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 scores for every chunk against the question."""
    q_tokens = tokenize(question)
    if not q_tokens:
        return [0.0] * len(chunks)

    doc_tokens_list = [tokenize(c.get("text", "")) for c in chunks]
    doc_lens = [len(t) for t in doc_tokens_list]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
    N = len(chunks)

    df: Counter = Counter()
    for tokens in doc_tokens_list:
        for t in set(tokens):
            df[t] += 1

    idf = {term: math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
           for term in set(q_tokens)}

    scores: list[float] = []
    for i, tokens in enumerate(doc_tokens_list):
        tf = Counter(tokens)
        score = 0.0
        dl = doc_lens[i]
        for term in q_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            num = f * (k1 + 1)
            den = f + k1 * (1 - b + b * dl / avgdl)
            score += idf.get(term, 0) * num / den
        scores.append(score)
    return scores


# ── Entity Matching ──────────────────────────────────────────────────────────

def extract_query_entities(question: str, entity_names: list[str]) -> list[str]:
    """Extract entities that appear in the query string."""
    query_lower = question.lower()
    found = []
    # Sort by length descending to catch longer names first
    for name in sorted(entity_names, key=len, reverse=True):
        if name.lower() in query_lower:
            found.append(name)
    return found


def is_noise_chunk(text: str) -> bool:
    """Identify if a chunk is likely metadata, headers, or formatting noise."""
    t = text.strip()
    if not t: return True
    # Book metadata patterns (e.g. ID 532)
    if t.startswith("_By_") or t.startswith("Produced by") or "Project Gutenberg" in t:
        return True
    if "Illustrations by" in t or "frontispiece" in t.lower():
        return True
    # Header patterns
    if re.match(r"^(CHAPTER|SECTION|BOOK) [IVXLCDM\d]+", t, re.IGNORECASE):
        return True
    # Very high ratio of non-alphanumeric chars (decoration/formatting)
    alnum_ratio = len(re.findall(r"\w", t)) / (len(t) + 1)
    if len(t) < 200 and alnum_ratio < 0.5:
        return True
    return False


# ── Narrative Intent Extraction ──────────────────────────────────────────────

NARRATIVE_INTENT_LEXICON = {
    "trial": ["trial", "court", "judge", "testify", "legal", "sentence", "judgment"],
    "death": ["death", "kill", "murder", "died", "execute", "execution", "slain"],
    "escape": ["escape", "flee", "rescue", "liberate", "rescued"],
    "betrayal": ["betray", "traitor", "treachery", "deceive"],
    "support": ["support", "defend", "protect", "help", "save", "stands up for"],
    "affiliation": ["serve", "join", "ally", "friend"],
    "conflict": ["war", "battle", "fight", "soldier", "army", "combat", "fight", "survive"],
}

def extract_narrative_intent(question: str) -> dict[str, set[str]]:
    """Extract structured narrative components: EVENT and RELATION."""
    q_lower = question.lower()
    decomposition = {
        "events": set(),
        "relations": set(),
    }
    
    # Map questions to core narrative events/relations without overhead synonyms
    EVENT_MAP = {
        "trial": ["trial", "court", "judge", "testify", "legal", "sentence", "judgment", "acquitted", "charges"],
        "death": ["death", "kill", "murder", "died", "execute", "execution", "slain"],
        "conflict": ["war", "battle", "fight", "soldier", "army", "combat"],
        "escape": ["escape", "flee"],
    }
    RELATION_MAP = {
        "support": ["support", "defend", "protect", "help", "save", "stands up for", "stood up"],
        "betrayal": ["betray", "traitor", "treachery", "deceive"],
        "affiliation": ["serve", "join", "ally"],
    }

    for event, keywords in EVENT_MAP.items():
        if any(k in q_lower for k in keywords):
            decomposition["events"].add(event)
            # Only include the specific keyword found to keep it discriminative
            for k in keywords:
                if k in q_lower: decomposition["events"].add(k)
                    
    for rel, keywords in RELATION_MAP.items():
        if any(k in q_lower for k in keywords):
            decomposition["relations"].add(rel)
            for k in keywords:
                if k in q_lower: decomposition["relations"].add(k)
                    
    return decomposition


# ── Hybrid Retrieval ──────────────────────────────────────────────────────────

def hybrid_retrieval(
    index, chunks: list[dict], query_vec: np.ndarray, question: str,
    top_k: int = DEFAULT_TOP_K, entities_data: list[dict] | None = None,
    qtype: str = "general", mode: str = "hybrid", qsubtype: str = "default",
    book_dir: Path | None = None, temporal_cue: str | None = None,
    event_ctx: dict | None = None, query_anchors: set[str] | None = None,
    anchor_event: str | None = None, query_decomposition: dict | None = None,
) -> list[dict]:
    """Retrieve chunks using parallel candidate generation: Dense + BM25 + Entity.
    
    Implements Step 2 (Parallel Candidate Generation) and Step 1 (Entity Index).
    """
    use_dense = mode in ("hybrid", "dense_only")
    use_bm25 = mode in ("hybrid", "bm25_only")
    use_entity = mode == "hybrid" and book_dir is not None

    # Load Entity Index
    entity_index = {}
    if use_entity:
        entity_index = load_entity_index(book_dir)

    # 1. Parallel Candidate Generation (BEFORE MMR)
    candidate_indices = set()
    
    # DENSE Candidates
    dense_scores_map = {}
    if use_dense:
        # Increase search pool for higher recall
        d_search_k = min(top_k * 2, index.ntotal)
        d_scores, d_indices = index.search(query_vec.reshape(1, -1), d_search_k)
        for s, idx in zip(d_scores[0], d_indices[0]):
            if idx != -1:
                candidate_indices.add(int(idx))
                dense_scores_map[int(idx)] = float(s)

    # BM25 Candidates
    bm25_raw = [0.0] * len(chunks)
    if use_bm25:
        bm25_raw = compute_bm25_scores(chunks, question)
        bm25_top_k = sorted(range(len(chunks)), key=lambda i: bm25_raw[i], reverse=True)[:top_k]
        for idx in bm25_top_k:
            candidate_indices.add(idx)

    # ENTITY Candidates (Direct lookup)
    entity_hits_map = Counter()
    if use_entity and entity_index:
        query_entities = extract_query_entities(question, list(entity_index.keys()))
        for ent in query_entities:
            for cid in entity_index.get(ent.lower(), []):
                # We need index, not chunk_id. Assuming cid is chunk_id which is index.
                # If chunk_id != index, we'd need a mapping.
                # In this system, chunk_id is usually same as index for a single book.
                candidate_indices.add(int(cid))
                entity_hits_map[int(cid)] += 1

    if not candidate_indices:
        return []

    # 2. Score and Re-rank
    
    # Normalization constants
    d_vals = list(dense_scores_map.values())
    d_min = min(d_vals) if d_vals else 0.0
    d_max = max(d_vals) if d_vals else 1.0
    d_range = d_max - d_min if d_max > d_min else 1.0

    b_max = max(bm25_raw) if any(v > 0 for v in bm25_raw) else 1.0
    e_max = max(entity_hits_map.values()) if entity_hits_map else 1.0

    # Calculate event anchor timeline if applicable
    event_anchor_time = None
    if anchor_event and event_ctx:
        found_times = []
        for ev in event_ctx.get("events", []):
            if anchor_event.lower() in ev.get("type", "").lower() or anchor_event.lower() in ev.get("action", "").lower():
                # Get timeline for this chunk
                c_idx = ev.get("chunk_id")
                if c_idx is not None and c_idx < len(chunks):
                    found_times.append(float(chunks[c_idx].get("timeline", 0.5)))
        if found_times:
            event_anchor_time = sum(found_times) / len(found_times)

    results = []
    for idx in candidate_indices:
        # ── Noise Filtering ──────────────────────────────────────────────
        if is_noise_chunk(chunks[idx].get("text", "")):
            continue

        # Normalized scores
        d_norm = (dense_scores_map.get(idx, d_min) - d_min) / d_range if d_range > 0 else 0.0
        b_norm = bm25_raw[idx] / b_max if b_max > 0 else 0.0
        e_norm = entity_hits_map[idx] / e_max if e_max > 0 else 0.0
        
        # Adaptive weighting
        eff_bm25 = BM25_WEIGHT
        eff_dense = DENSE_WEIGHT
        if qtype in ("who", "where", "what"):
            eff_bm25 = min(0.6, BM25_WEIGHT + 0.15)
            eff_dense = max(0.2, DENSE_WEIGHT - 0.15)
        
        score = (eff_dense * d_norm + 
                 eff_bm25 * b_norm + 
                 ENTITY_WEIGHT * e_norm)
        
        # ── Temporal Boosting & Filtering ──────────────────────────────
        timeline = float(chunks[idx].get("timeline", 0.5))
        
        # Case A: Specific event anchor (e.g. "after the trial")
        if event_anchor_time is not None:
            if temporal_cue in ("after", "following", "since"):
                if timeline > event_anchor_time: score += 0.2
                else: score -= 0.1 # Penalty for wrong side of event
            elif temporal_cue in ("before", "prior"):
                if timeline < event_anchor_time: score += 0.2
                else: score -= 0.1

        # Case B: General markers
        elif temporal_cue == "start":
            score += 0.15 * (1.0 - timeline)
        elif temporal_cue == "end":
            score += 0.15 * timeline

        # ── Event Triple Boosting ──────────────────────────────────────
        if event_ctx:
            c_events = event_ctx.get("chunk_to_events", {}).get(idx, [])
            answer_type = query_decomposition.get("answer_type") if query_decomposition else "GENERAL"
            
            for ev in c_events:
                # Direct match: character mentioned AND event type/action mentioned
                char_match = any(a.lower() in str(ev.get("who", "")).lower() for a in query_anchors) if query_anchors else False
                type_match = any(a.lower() in ev.get("type", "").lower() for a in query_anchors) if query_anchors else False
                action_match = any(a.lower() in ev.get("action", "").lower() for a in query_anchors) if query_anchors else False
                
                if char_match and (type_match or action_match):
                    score += 0.25 # Significant boost for grounding match
                    
                # Sub-Step: PERSON boosting for 'Who' questions in Event Chunks
                if answer_type == "PERSON" and (type_match or action_match):
                    # Check if there's any capitalized entity in this chunk that isn't the query character
                    text = chunks[idx].get("text", "")
                    if re.search(r"\b[A-Z][a-z]+\b", text):
                        score += 0.15 

        # Narrator timeline bonus (Legacy)
        if qsubtype == "narrator":
            score += 0.1 * (1.0 - timeline)

        chunk = dict(chunks[idx])
        chunk["score"] = score
        chunk["dense_score"] = d_norm
        chunk["bm25_score"] = b_norm
        chunk["entity_score"] = e_norm
        chunk["faiss_idx"] = idx
        results.append(chunk)

    results.sort(key=lambda c: c["score"], reverse=True)
    return results[:top_k]
