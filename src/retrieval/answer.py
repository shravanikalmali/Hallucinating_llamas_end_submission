"""
Answer Extraction — Scored Symbolic Extraction (Symbolic Core)
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util

from src.config import (
    KEYWORD_OVERLAP_WEIGHT, SENTENCE_SCORE_WEIGHT, CHUNK_SCORE_WEIGHT
)
from src.retrieval.hybrid_search import get_model, embed_question

# ── Validators ──────────────────────────────────────────────────────────────

def validate_person(answer: str) -> bool:
    if not answer: return False
    ans = str(answer).strip()
    a_low = ans.lower()
    PRONOUNS = {"he", "she", "it", "who", "what", "whatever", "they", "this", "that", "there", "i", "we", "you", "whom", "him", "her", "them"}
    GENERIC_WORDS = {"the", "this", "town", "city", "ship", "road", "square", "cook", "method", "in", "at", "by", "of", "chapter", "book", "illustration", "page", "dogs", "method", "but", "so", "and", "with", "from", "for", "to", "on", "island", "place", "man", "woman", "boy", "girl"}
    NATIONALITIES = {"american", "english", "british", "french", "spanish", "german", "dutch", "portuguese"}
    if a_low in GENERIC_WORDS or a_low in PRONOUNS or a_low in NATIONALITIES: return False
    if len(ans) < 3: return False
    if ans.isupper() and len(ans) > 3: return False
    return bool(re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", ans))

def validate_location(answer: str) -> bool:
    if not answer: return False
    ans = str(answer).strip()
    GENERIC_LOCS = {"town", "city", "ship", "road", "square", "place", "island", "sea", "ocean", "river"}
    if ans.lower() in GENERIC_LOCS: return False
    if ans.isupper() and len(ans) > 3: return False
    return bool(re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", ans)) and len(ans.split()) <= 4

def validate_phrase(answer: str) -> bool:
    if not answer: return False
    ans = str(answer).strip()
    words = ans.split()
    PRONOUNS = {"he", "she", "it", "they", "who", "what", "whatever"}
    if ans.lower() in PRONOUNS: return False
    return 1 <= len(words) <= 10 and "." not in ans and "," not in ans.split()[-1]

# ── Extraction Logic ─────────────────────────────────────────────────────────

def _get_query_entities(question: str) -> set[str]:
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    return set(entities)

def _split_into_sentences(text: str) -> list[str]:
    pattern = re.compile(r"(?<=[.!?])\s+")
    return [s.strip() for s in pattern.split(text) if len(s.strip()) > 5]

def _query_aware_filter(sentences: list[str], qtype: str, qsubtype: str) -> list[str]:
    if not sentences: return []
    filtered = []
    q_low = qtype.lower()
    qs_low = qsubtype.lower()
    for s in sentences:
        s_low = s.lower()
        has_cap = bool(re.search(r"\b[A-Z][a-z]+\b", s))
        if q_low == "who" or qs_low == "identity":
            has_verb = bool(re.search(r"\b(said|testify|defend|stood|known|called|met|kill|put|sent|saw|is|was|became|named|fought|served|asked)\b", s_low))
            if has_cap or has_verb: filtered.append(s)
        elif q_low == "where" or qs_low == "location":
            if any(m in s_low for m in ["in ", "at ", "to ", "from ", "near ", "into ", "entered ", "returned "]) and has_cap: filtered.append(s)
        elif q_low == "why":
            if any(m in s_low for m in ["because", "since", "reason", "due to", "order to", "forced"]): filtered.append(s)
        elif "charge" in qs_low or "trial" in qs_low:
            if any(m in s_low for m in ["charge", "guilty", "piracy", "murder", "accused", "court", "witness", "robbery", "felon"]): filtered.append(s)
        elif "end" in qs_low or "end" in q_low:
            if any(m in s_low for m in ["finally", "at last", "end", "sailed", "departed", "left", "died", "stayed", "buried"]): filtered.append(s)
    return filtered if filtered else sentences

def is_answerable(question: str, top_sents: list[dict], best_candidate: dict | None, model: SentenceTransformer, qsubtype: str) -> bool:
    """3-Layer Answerability Gate."""
    q_emb = embed_question(question, model)
    if len(q_emb.shape) == 1: q_emb = q_emb.reshape(1, -1)
    m_sim = 0.0
    for s in top_sents[:8]:
        s_emb = model.encode(s["sentence"], convert_to_numpy=True, show_progress_bar=False)
        if len(s_emb.shape) == 1: s_emb = s_emb.reshape(1, -1)
        sim = float(np.dot(q_emb, s_emb.T).item())
        if sim > m_sim: m_sim = sim
    if m_sim < 0.28: return False # Slightly relaxed
    
    rev_list = [float(s["evidence_relevance"]) for s in top_sents]
    top_ev = max(rev_list) if rev_list else 0.0
    if top_ev < 0.12: return False
    
    if not best_candidate: return False
    score = float(best_candidate.get("score", 0.0))
    limit = 1.2 # Generalized threshold
    if qsubtype in ["who", "trial_who", "serve_relation"]: limit = 1.6
    return score >= limit

# ── Scored Evidence Extractors ───────────────────────────────────────────────

def _score_candidate(cand: str, sentence: str, q_entities: set[str], triggers: list[str], qsubtype: str) -> float:
    score = 0.0
    c_low = cand.lower()
    s_low = sentence.lower()
    for t in triggers:
        t_low = t.lower()
        if t_low in s_low:
            c_idx = s_low.find(c_low)
            t_idx = s_low.find(t_low)
            if c_idx != -1 and t_idx != -1:
                if t_low in ["said", "replied", "testified", "asked"]: score += 1.5 if c_idx < t_idx else 0.2
                else: score += max(0.0, 2.5 - (abs(c_idx - t_idx) / 40.0))
    for ent in q_entities:
        e_low = ent.lower()
        if e_low == c_low or c_low in e_low or e_low in c_low:
            score -= 15.0
            continue
        if e_low in s_low:
            e_idx = s_low.find(e_low)
            c_idx = s_low.find(c_low)
            if e_idx != -1 and c_idx != -1:
                dist = abs(c_idx - e_idx)
                score += max(0.0, 2.5 - (dist / 25.0))
                mid = s_low[min(e_idx, c_idx):max(e_idx, c_idx)]
                if any(k in mid for k in [" is ", " was ", " as ", " called ", " named "]): score += 2.0
    if cand[0].isupper(): score += 0.8
    if qsubtype == "identity": score += 1.0
    if c_low in ["man", "woman", "the", "he", "she", "it", "who", "they"]: score -= 5.0
    return score

def extract_who_relation_scored(top_sents: list[dict], q_entities: set[str], triggers: list[str], qsubtype: str) -> dict | None:
    candidates = []
    for s in top_sents:
        found = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", s["sentence"])
        for cand in found:
            sc = _score_candidate(cand, s["sentence"], q_entities, triggers, qsubtype)
            candidates.append({"answer": cand, "score": sc, "context": s["sentence"], "chunk_id": s["chunk_id"]})
    if not candidates: return None
    best = max(candidates, key=lambda x: x["score"])
    return {"answer": best["answer"], "reasoning": "who_relation_scored", "context": best["context"], "score": best["score"], "chunk_id": best["chunk_id"]}

def extract_identity_scored(top_sents: list[dict], q_entities: set[str]) -> dict | None:
    candidates = []
    for s in top_sents:
        text = s["sentence"]
        for ent in q_entities:
            m1 = re.search(rf"\b{re.escape(ent)},\s+(?:a|the|his|her)\s+([^,.;!]{3,35})", text, re.I)
            if m1: 
                cand = m1.group(1).strip()
                is_name = bool(re.search(r"\b[A-Z][a-z]+\b", cand))
                sc = 3.5 if not is_name else 1.0
                candidates.append({"answer": cand, "score": sc, "context": text, "chunk_id": s["chunk_id"]})
            m2 = re.search(rf"\b{re.escape(ent)}\s+(?:is|was|became|known as|named|called)\s+([^,.;!]{3,35})", text, re.I)
            if m2:
                cand = m2.group(1).strip()
                is_name = bool(re.search(r"\b[A-Z][a-z]+\b", cand))
                sc = 3.0 if not is_name else 1.5
                candidates.append({"answer": cand, "score": sc, "context": text, "chunk_id": s["chunk_id"]})
    if not candidates: return None
    best = max(candidates, key=lambda x: x["score"])
    return {"answer": best["answer"], "reasoning": "identity_scored", "context": best["context"], "score": best["score"], "chunk_id": best["chunk_id"]}

def extract_affiliation_scored(top_sents: list[dict], q_entities: set[str]) -> dict | None:
    AFF_KEYWORDS = ["Royalist", "Roundhead", "Parliament", "Loyalist", "Rebel", "Federation", "Empire", "Alliance", "Order", "Side"]
    candidates = []
    for s in top_sents:
        for k in AFF_KEYWORDS:
            if k.lower() in s["sentence"].lower():
                sc = _score_candidate(k, s["sentence"], q_entities, ["side", "party", "war", "allegiance", "serve"], "affiliation")
                candidates.append({"answer": k, "score": sc, "context": s["sentence"], "chunk_id": s["chunk_id"]})
    if not candidates: return None
    best = max(candidates, key=lambda x: x["score"])
    return {"answer": best["answer"], "reasoning": "affiliation_scored", "context": best["context"], "score": best["score"], "chunk_id": best["chunk_id"]}

def extract_serve_relation_scored(top_sents: list[dict], q_entities: set[str]) -> dict | None:
    candidates = []
    triggers = ["serve under", "serve with", "joined", "fought for", "command of", "captain of", "master of", "called"]
    for s in top_sents:
        for t in triggers:
            m = re.search(rf"\b{re.escape(t)}\b\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", s["sentence"], re.I)
            if m:
                cand = m.group(1).strip()
                sc = _score_candidate(cand, s["sentence"], q_entities, [t], "serve_relation")
                candidates.append({"answer": cand, "score": sc + 2.0, "context": s["sentence"], "chunk_id": s["chunk_id"]})
    if not candidates: return None
    best = max(candidates, key=lambda x: x["score"])
    return {"answer": best["answer"], "reasoning": "serve_scored", "context": best["context"], "score": best["score"], "chunk_id": best["chunk_id"]}

def extract_location_scored(top_sents: list[dict], q_entities: set[str]) -> dict | None:
    candidates = []
    for s in top_sents:
        text = s["sentence"]
        m = re.search(r"\b(?:to|in|at|from|near|into|returned to|entered)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b", text, re.I)
        if m:
            cand = m.group(1).strip()
            sc = _score_candidate(cand, text, q_entities, ["go", "arrive", "stay", "sails", "journey", "end", "return"], "location")
            candidates.append({"answer": cand, "score": sc, "context": text, "chunk_id": s["chunk_id"]})
    if not candidates: return None
    best = max(candidates, key=lambda x: x["score"])
    return {"answer": best["answer"], "reasoning": "location_scored", "context": best["context"], "score": best["score"], "chunk_id": best["chunk_id"]}

def extract_charges_scored(top_sents: list[dict], q_entities: set[str]) -> dict | None:
    KEYWORDS = ["piracy", "murder", "robbery", "theft", "felony", "traitor", "mutiny"]
    candidates = []
    for s in top_sents:
        t_low = s["sentence"].lower()
        for k in KEYWORDS:
            if k in t_low:
                sc = _score_candidate(k, s["sentence"], q_entities, ["trial", "charge", "accused", "guilty"], "charges")
                candidates.append({"answer": k, "score": sc, "context": s["sentence"], "chunk_id": s["chunk_id"]})
    if not candidates: return None
    best = max(candidates, key=lambda x: x["score"])
    return {"answer": best["answer"], "reasoning": "charges_scored", "context": best["context"], "score": best["score"], "chunk_id": best["chunk_id"]}

def extract_answer(
    question: str, final_chunks: list[dict],
    model: SentenceTransformer = None, use_cross_encoder: bool = True,
    qtype: str = "general",
    qsubtype: str = "default",
    graph_context: dict | None = None,
    verbose: bool = False,
    book_id: str = ""
) -> dict:
    if not final_chunks: return {"answer": "NOT FOUND", "validated": False, "reasoning": "no_chunks"}
    q_entities = _get_query_entities(question)
    sentence_candidates = []
    for chunk in final_chunks:
        raw_sents = _split_into_sentences(chunk.get("text", ""))
        filtered = _query_aware_filter(raw_sents, qtype, qsubtype)
        for s in filtered:
            sentence_candidates.append({"sentence": s, "chunk_id": chunk["chunk_id"], "evidence_relevance": float(chunk.get("score", 0.0))})
    if not sentence_candidates: return {"answer": "NOT FOUND", "validated": False, "reasoning": "no_filtered_sents"}
    sentence_candidates.sort(key=lambda x: x["evidence_relevance"], reverse=True)
    top_sents = sentence_candidates[:15]
    qs = qsubtype.lower()
    ql = question.lower()
    res = None
    if qs == "identity" or (ql.startswith("who is") and not any(k in ql for k in ["under", "serve", "joined"])):
        res = extract_identity_scored(top_sents, q_entities)
    elif "trial" in ql or qs == "trial_who":
        res = extract_who_relation_scored(top_sents, q_entities, ["stood up", "defended", "replied", "testified", "witness", "trial"], qs)
    elif qs == "serve_relation" or "serve" in ql or "under" in ql or "joined" in ql:
        res = extract_serve_relation_scored(top_sents, q_entities)
    elif qs == "affiliation" or any(k in ql for k in ["side", "party", "war", "allegiance"]):
        res = extract_affiliation_scored(top_sents, q_entities)
    elif qtype == "where" or qs == "location":
        res = extract_location_scored(top_sents, q_entities)
    elif "charge" in ql or "accused" in ql:
        res = extract_charges_scored(top_sents, q_entities)
    elif qs == "end_state" or "at the end" in ql:
        res = extract_who_relation_scored(top_sents, q_entities, ["finally", "at last", "sailed", "departed", "died", "stayed", "buried"], qs)
    if not res:
        if qtype == "who": res = extract_who_relation_scored(top_sents, q_entities, ["said", "met", "killed", "put", "called", "named"], qs)
        elif qtype == "where": res = extract_location_scored(top_sents, q_entities)
    if res and res.get("answer"):
        cand = res["answer"]
        v_ok = False
        if qs == "identity": v_ok = validate_phrase(cand) or validate_person(cand)
        elif qtype == "who" or qs == "trial_who": v_ok = validate_person(cand)
        elif qtype == "where" or qs == "location": v_ok = validate_location(cand)
        else: v_ok = validate_phrase(cand)
        if v_ok:
            if not is_answerable(question, top_sents, res, model, qs):
                 return {"answer": "NOT FOUND", "validated": False, "reasoning": "not_answerable"}
            if verbose: print(f"    [Symbolic FINAL] {cand} ({res['reasoning']}) | Score: {res.get('score'):.2f}")
            return {"answer": cand, "validated": True, "reasoning": res["reasoning"], "source_chunk_id": res.get("chunk_id"), "context": res["context"], "score": res.get("score")}
    return {"answer": "NOT FOUND", "validated": False, "reasoning": "symbolic_miss"}
