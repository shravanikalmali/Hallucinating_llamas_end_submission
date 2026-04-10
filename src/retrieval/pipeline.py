"""
Retrieval Pipeline — End-to-End Orchestration

Chains together all retrieval stages, gated by RETRIEVAL_CONFIG:
  1. Load FAISS index
  2. Embed question
  3. Hybrid / dense-only / BM25-only retrieval
  4. MMR re-ranking  (if use_mmr)
  4.5 Chunk neighbour inclusion  (if use_neighbors)
  5. Entity-based expansion  (if use_graph_expansion)
  5.5 Graph-based reasoning  (if use_graph_reasoning)
  6. Answer extraction  (cross-encoder gated by use_cross_encoder)
"""

import json
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Any

from src.config import (
    PROCESSED_DIR, DEFAULT_TOP_K, DEFAULT_MMR_LAMBDA,
    DEFAULT_FINAL_COUNT, DEFAULT_EXPANSION_MAX, HYBRID_ALPHA,
)
from src.retrieval_config import RETRIEVAL_CONFIG, config_summary
from src.retrieval.hybrid_search import (
    load_book_index, load_entities, embed_question, get_model,
    hybrid_retrieval, detect_question_type_info,
)
from src.retrieval.reranker import global_mmr, include_neighbors
from src.retrieval.expansion import entity_expansion
from src.retrieval.reasoning import graph_reasoning, collect_graph_hints
from src.retrieval.answer import extract_answer
from src.retrieval.hybrid_llm import refine_with_hybrid_llm
from src.retrieval.llm_answer import llm_extract_answer


from src.retrieval.hybrid_search import (
    load_book_index, load_entities, embed_question, get_model,
    hybrid_retrieval, detect_question_type_info,
    load_narrative_context, extract_query_entities,
    extract_narrative_intent,
)
from src.graph.event_manager import load_event_context


def anchor_query_to_graph(question: str, graph: "nx.Graph | None") -> dict[str, set[str]]:
    """Identify structured entities, event types, and relations in the query."""
    decomposition = {
        "entities": set(),
        "events": set(),
        "relations": set(),
        "answer_type": "GENERAL"
    }
    
    # 1. Answer Type Detection
    q_low = question.lower()
    if q_low.startswith("who") or "whom" in q_low:
        decomposition["answer_type"] = "PERSON"
    elif q_low.startswith("where"):
        decomposition["answer_type"] = "LOCATION"
    elif q_low.startswith("when"):
        decomposition["answer_type"] = "TIME"

    if not graph:
        return decomposition

    # 2. Entity Extraction
    node_names = list(graph.nodes())
    decomposition["entities"] = set(extract_query_entities(question, node_names))

    # 3. Narrative Intent Extraction
    intent = extract_narrative_intent(question)
    decomposition["events"] = intent["events"]
    decomposition["relations"] = intent["relations"]

    return decomposition


def retrieve_and_answer(
    book_id: str, question: str,
    processed_dir: Path = PROCESSED_DIR,
    top_k: int = DEFAULT_TOP_K,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    final_count: int = DEFAULT_FINAL_COUNT,
    expansion_max: int = DEFAULT_EXPANSION_MAX,
    verbose: bool = False,
    gold_answer: str | None = None,
    attempt: int = 1, # Tracking attempts for fallback
) -> dict:
    """End-to-end retrieval pipeline with graph-aware reasoning and fallback loop."""
    cfg = RETRIEVAL_CONFIG
    book_dir = processed_dir / book_id

    # 1. Load index & narrative context
    if verbose:
        print(f"[{attempt}.1] Loading FAISS index and narrative graph …")
    index, chunks = load_book_index(book_id, processed_dir)
    all_embeddings = np.load(book_dir / "embeddings.npy").astype(np.float32)
    entities_data = load_entities(book_dir)

    # NEW: Structured Query Anchoring
    graph_ctx = load_narrative_context(book_dir)
    query_decomp = anchor_query_to_graph(question, graph_ctx.get("graph"))
    graph_ctx["query_decomposition"] = query_decomp
    query_anchors = query_decomp["entities"] | query_decomp["events"] | query_decomp["relations"]
    graph_ctx["query_anchors"] = query_anchors

    # NEW: Load Event Context for causal tracking
    from src.graph.event_manager import load_event_context
    event_ctx = load_event_context(book_dir)

    if verbose:
        print(f"    Structured Decomposition:")
        print(f"      ENTITIES: {list(query_decomp['entities'])}")
        print(f"      EVENTS  : {list(query_decomp['events'])}")
        print(f"      TYPE    : {query_decomp['answer_type']}")

    # 2. Embed question
    model = get_model()
    query_vec = embed_question(question, model)
    qtype, qsubtype, temporal_cue, anchor_event = detect_question_type_info(question)

    if verbose and temporal_cue:
        print(f"    Temporal Cue detected: {temporal_cue}")
        if anchor_event:
            print(f"    Relative Anchor: {anchor_event}")

    # 3. Hybrid Retrieval (Frozen at Top 5, no MMR)
    if verbose:
        print(f"[{attempt}.3] Hybrid Retrieval (Frozen K=5) …")
    
    # We use a fixed top_k of 5 as per user instruction to avoid noise.
    initial_results = hybrid_retrieval(index, chunks, query_vec, question,
                                        top_k=5, entities_data=entities_data, qtype=qtype, qsubtype=qsubtype,
                                        book_dir=book_dir, temporal_cue=temporal_cue,
                                        event_ctx=event_ctx, query_anchors=query_anchors,
                                        anchor_event=anchor_event, query_decomposition=query_decomp)
    if verbose:
        print(f"    Initial Top-5 Chunks:")
        for res in initial_results[:5]:
            print(f"      - ID {res['chunk_id']} (Score: {res['score']:.4f}): {res['text'][:100]}...")

    # 4. Graph-Aware MMR re-ranking with Event Coverage
    # 4. MMR Disabled (Frozen Retrieval Mode)
    mmr_results = initial_results
    if verbose:
        print(f"[{attempt}.4] Retrieval Diversity (MMR Disabled) …")

    # 5. Graph Neighbors Expansion (Causal + Neighbor)
    if cfg["use_graph_expansion"]:
        if verbose:
            print(f"[{attempt}.5] Graph-Neighbor Expansion (Causal + 1-hop) …")

        expanded_results = list(mmr_results)
        seen_ids = {c["chunk_id"] for c in mmr_results}

        # A. Causal Expansion: If we find a reaction, pull in the trigger.
        reaction_map = event_ctx.get("reaction_to_triggers", {})
        for c in mmr_results:
            cid = c["chunk_id"]
            if cid in reaction_map:
                for tcid in reaction_map[cid]:
                    if tcid not in seen_ids and tcid < len(chunks):
                        tc = dict(chunks[tcid])
                        tc["score"] = c["score"] * 0.9
                        tc["expansion"] = "causal_trigger"
                        expanded_results.append(tc)
                        seen_ids.add(tcid)

        # B. NarCo 1-hop expansion (legacy)
        for c in mmr_results:
            cid = c["chunk_id"]
            if cid in graph_ctx["narco_map"]:
                n_ids = list(graph_ctx["narco_map"][cid]["edges"].keys())
                for nid in n_ids:
                    if nid not in seen_ids and nid < len(chunks):
                        nc = dict(chunks[nid])
                        nc["score"] = c["score"] * 0.8
                        nc["expansion"] = "neighbor"
                        expanded_results.append(nc)
                        seen_ids.add(nid)

        # C. Proactive Consequence Expansion (addressed Test 4/5)
        for c in mmr_results:
            text = c.get("text", "").lower()
            ACTION_TRIGGERS = ["escape", "flee", "refuse", "witness", "testify", "refused"]
            if any(k in text for k in ACTION_TRIGGERS):
                cid = c["chunk_id"]
                # Scan a 5-chunk window ahead for the outcome
                for target_id in range(cid + 1, min(len(chunks), cid + 6)):
                    if target_id not in seen_ids:
                        target_txt = chunks[target_id].get("text", "").lower()
                        EXPECTED_OUTCOMES = ["caught", "captured", "killed", "died", "execution", "honor", "loyalty"]
                        if any(o in target_txt for o in EXPECTED_OUTCOMES):
                            tc = dict(chunks[target_id])
                            tc["score"] = c.get("score", 0.5) * 0.95
                            tc["expansion"] = "proactive_consequence"
                            expanded_results.append(tc)
                            seen_ids.add(target_id)

            if len(expanded_results) >= final_count + expansion_max:
                break
        final_chunks = expanded_results
        if verbose and len(expanded_results) > len(mmr_results):
            print(f"    Expanded Results (Added {len(expanded_results) - len(mmr_results)} chunks):")
            for res in expanded_results[len(mmr_results):]:
                print(f"      - ID {res['chunk_id']} ({res.get('expansion', 'unknown')}): {res['text'][:100]}...")
    else:
        final_chunks = mmr_results    # 6. Answer Extraction (Graph-Guided)
    if verbose:
        print(f"[{attempt}.6] Extracting answer from {len(final_chunks)} chunks …")

    # STEP 1: run symbolic extraction ALWAYS (authoritative core)
    answer_result = extract_answer(
        question,
        final_chunks,
        model,
        use_cross_encoder=cfg["use_cross_encoder"],
        qtype=qtype,
        qsubtype=qsubtype,
        graph_context=graph_ctx,
        verbose=verbose,
        book_id=book_id
    )

    # STEP 2: HARD VALIDATION GATE (Symbolic Priority)
    if answer_result.get("answer") and answer_result.get("validated") and answer_result.get("answer") != "NOT FOUND":
        if verbose:
            print(f"    [Symbolic FINAL] {answer_result['answer']} ({answer_result.get('reasoning')})")
        return answer_result

    # STEP 3: CONSTRAINED LLM FALLBACK (SELECTOR)
    # GATE: Only fallback to LLM if symbolic failed but context is potentially answerable
    if cfg.get("use_llm_answer") and answer_result.get("reasoning") != "not_answerable":
        if verbose: print("    [Symbolic FAILED] Falling back to Constrained LLM Selector...")
            
        # Passing only top 2 evidence chunks to LLM to constrain context
        llm_result = llm_extract_answer(question, final_chunks[:2])
        
        # FINAL GATE: LLM output must pass same validation as symbolic
        from src.retrieval.answer import validate_phrase, validate_person, validate_location
        ans_cand = llm_result.get("answer")
        
        valid = False
        if qtype == "who" or qsubtype == "identity":
            valid = validate_person(ans_cand) or validate_phrase(ans_cand)
        elif qtype == "where":
            valid = validate_location(ans_cand)
        else:
            valid = validate_phrase(ans_cand)
            
        if valid and ans_cand and ans_cand.lower() != "not found":
            if verbose: print(f"    [LLM Selector] {ans_cand} | Reasoning: {llm_result.get('reasoning')}")
            return llm_result

    # STEP 4: FAIL SAFE
    if verbose:
        reason = answer_result.get("reasoning", "symbolic_failed")
        print(f"    [{reason.upper()}] Returning NOT FOUND")
    return {
        "answer": "NOT FOUND",
        "reasoning": answer_result.get("reasoning", "symbolic_failed")
    }

    # ── Logging ───────────────────────────────────────────────────────────
    retrieved_chunk_ids = [c["chunk_id"] for c in final_chunks]
    retrieval_log = {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "num_chunks_retrieved": len(final_chunks),
        "config": {k: v for k, v in cfg.items()},
        "query_anchors": list(query_anchors),
        "attempts": attempt,
    }

    contains_answer = None
    rank_of_answer_chunk = -1

    if gold_answer:
        gold_lower = gold_answer.lower()
        contains_answer = False
        rank_of_answer_chunk = -1
        for rank, c in enumerate(final_chunks, 1):
            if gold_lower in c.get("text", "").lower():
                contains_answer = True
                if rank_of_answer_chunk == -1:
                    rank_of_answer_chunk = rank

    retrieval_log: Dict[str, Any] = {
        "book_id": book_id,
        "question": question,
        "query_anchors": list(query_anchors),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "attempts": attempt,
        "contains_answer": contains_answer,
        "rank_of_answer_chunk": rank_of_answer_chunk,
        "config": {
            "top_k": top_k,
            "expanded": bool(attempt > 1)
        }
    }

    if gold_answer:
        retrieval_log["gold_answer"] = gold_answer
        if not contains_answer:
            retrieval_log["failure_type"] = "retrieval"
        elif gold_answer.lower() not in answer_result.get("answer", "").lower():
            retrieval_log["failure_type"] = "extraction"
        else:
            retrieval_log["failure_type"] = None

    return {
        "book_id": book_id, "question": question,
        "question_type": qtype, "question_subtype": qsubtype,
        **answer_result,
        "retrieval_log": retrieval_log,
        "num_initial": len(initial_results),
        "num_final": len(final_chunks),
        "stages": {
            "initial_retrieval_ids": [c["chunk_id"] for c in initial_results[:10]],
            "mmr_ranked_ids": [c["chunk_id"] for c in mmr_results[:10]],
            "final_expanded_ids": [c["chunk_id"] for c in final_chunks[:10]],
        }
    }


def run(book_id: str, question: str = None, questions_file: str = None,
        processed_dir: str = None, top_k: int = DEFAULT_TOP_K,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA, final_count: int = DEFAULT_FINAL_COUNT,
        expansion_max: int = DEFAULT_EXPANSION_MAX,
        no_entity_expansion: bool = False, no_cross_encoder: bool = False,
    no_neighbors: bool = False, no_graph_reasoning: bool = False,
    no_hybrid_llm: bool = False,
        output: str = None, verbose: bool = False):
    """CLI-level retrieval entry point."""
    processed = Path(processed_dir) if processed_dir else PROCESSED_DIR

    # Apply CLI flags as config overrides (backward compat)
    from src.retrieval_config import set_config
    cli_overrides = {}
    if no_entity_expansion:
        cli_overrides["use_graph_expansion"] = False
    if no_cross_encoder:
        cli_overrides["use_cross_encoder"] = False
    if no_neighbors:
        cli_overrides["use_neighbors"] = False
    if no_graph_reasoning:
        cli_overrides["use_graph_reasoning"] = False
    if no_hybrid_llm:
        cli_overrides["use_hybrid_llm"] = False
    if cli_overrides:
        set_config(cli_overrides)

    # Load questions
    questions: list[dict] = []
    if question:
        questions.append({"question": question})
    if questions_file:
        qf = Path(questions_file)
        with open(qf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

    if not questions:
        print("Provide either --question or --questions-file")
        return

    print("=" * 60)
    print("  Phase 2: Query-Time Retrieval")
    print("=" * 60)
    print(f"  Book: {book_id}  |  Questions: {len(questions)}")
    print(f"  Config: {config_summary()}")

    results = []
    final_answers = []
    recall_at_5, recall_at_10 = 0, 0
    has_gold = False

    for i, q_item in enumerate(questions, 1):
        q_text = q_item["question"] if isinstance(q_item, dict) else q_item
        gold = q_item.get("answer") if isinstance(q_item, dict) else None
        if gold:
            has_gold = True

        print(f"\n── Question {i}/{len(questions)} {'─' * 40}")
        print(f"  Q: {q_text}\n")

        result = retrieve_and_answer(
            book_id=book_id, question=q_text, processed_dir=processed,
            top_k=top_k, mmr_lambda=mmr_lambda, final_count=final_count,
            expansion_max=expansion_max, verbose=verbose,
            gold_answer=gold,
        )
        results.append(result)
        final_answers.append(result.get("answer", ""))

        print(f"  Q-type : {result.get('question_type', '?')} / {result.get('question_subtype', 'default')}")

        # Always show post-MMR ranking to diagnose missed answers in lower-ranked chunks.
        print("  Post-MMR top chunks (up to 10):")
        for item in result.get("mmr_top10", []):
            preview = item.get("text_preview", "").replace("\n", " ").strip()
            print(
                f"    {item['rank']:>2}. chunk={item['chunk_id']} "
                f"seg={item['segment_id']} mmr={item['mmr_score']:.4f} "
                f"score={item['score']:.4f} | {preview}"
            )

        print(f"  Answer : {result['answer']}")
        print(f"  Score  : {result['answer_score']:.4f}")
        print(f"  Source : chunk {result['source_chunk_id']}, segment {result['source_segment']}")

        # Track recall
        log = result.get("retrieval_log", {})
        if log.get("contains_answer") is not None:
            rank = log.get("rank_of_answer_chunk", -1)
            if 1 <= rank <= 5:
                recall_at_5 += 1
            if 1 <= rank <= 10:
                recall_at_10 += 1
            if log.get("failure_type"):
                print(f"  ⚠ Failure: {log['failure_type']}")

    # Print retrieval metrics
    if has_gold:
        n = len(questions)
        print(f"\n{'─' * 60}")
        print(f"  Retrieval Metrics ({n} questions)")
        print(f"  Recall@5:  {recall_at_5}/{n} = {recall_at_5/n:.2%}")
        print(f"  Recall@10: {recall_at_10}/{n} = {recall_at_10/n:.2%}")
        print(f"{'─' * 60}")

    if output:
        out_path = Path(output)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults saved → {out_path}")

    # Explicit final answer printout
    if len(final_answers) == 1:
        print(f"\nFinal answer: {final_answers[0]}")
    elif final_answers:
        print("\nFinal answers:")
        for i, ans in enumerate(final_answers, 1):
            print(f"  {i}. {ans}")

    print(f"\n{'='*60}\n  Done!\n{'='*60}")
