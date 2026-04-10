"""
Hybrid LLM Reasoning (Optional)

Phase 4 bridge: combines graph hints + top retrieval evidence and asks an LLM
(DeepSeek-compatible Chat Completions API) to refine the final answer.

This module is fail-safe:
- If API key is missing, returns None.
- If API call fails or parsing fails, returns None.
"""

import json
import os
from typing import Any
from urllib import request, error


def _build_messages(question: str, graph_hints: dict[str, Any] | None, chunks: list[dict]) -> list[dict[str, str]]:
    """Construct compact system/user messages for answer refinement."""
    # Ensure chunks is a list and slice safely
    safe_chunks = chunks[:10] if isinstance(chunks, list) else []
    evidence_lines = []
    for i, c in enumerate(safe_chunks, 1):
        txt = (c.get("text", "") or "").replace("\n", " ").strip()
        etype = c.get("expansion", "retrieval")
        evidence_lines.append(
            f"[{i}] chunk={c.get('chunk_id')} type={etype} :: {txt[:800]}"
        )

    graph_json = json.dumps(graph_hints or {}, ensure_ascii=False)

    system_msg = (
        "You are a narrative reasoning engine. Follow this protocol:\n"
        "1. Identify primary events/actions in the chunks.\n"
        "2. Find the 'Trigger Event' that explains the 'Reaction'.\n"
        "3. Explicitly state characters' motivations or identities if missing verbatim.\n"
        "Use ONLY provided evidence. Output JSON with keys: 'answer', 'confidence', 'rationale'."
    )

    user_msg = (
        f"Question: {question}\n\n"
        f"Narrative Graph Context:\n{graph_json}\n\n"
        f"Evidence Chunks (ordered by event flow/relevance):\n" + "\n".join(evidence_lines) + "\n\n"
        "Reason Step-by-Step in the 'rationale' field before giving the final answer."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _deepseek_chat(messages: list[dict[str, str]], timeout_sec: int = 20) -> str | None:
    """Call DeepSeek-compatible chat API using stdlib only."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 160,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            return obj.get("choices", [{}])[0].get("message", {}).get("content")
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _parse_llm_json(text: str | None) -> dict[str, Any] | None:
    """Parse model output expected as JSON object."""
    if not text:
        return None

    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        answer = str(obj.get("answer", "")).strip()
        if not answer:
            return None
        confidence = float(obj.get("confidence", 0.55))
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(obj.get("rationale", "")).strip()
        return {"answer": answer, "confidence": confidence, "rationale": rationale}
    except Exception:
        return None


def refine_with_hybrid_llm(
    question: str,
    graph_hints: dict[str, Any] | None,
    chunks: list[dict],
    fallback_answer: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return refined answer dict, or None if unavailable/failed.

    Intended usage:
      - call when graph reasoning is inconclusive,
      - pass graph hints + top retrieval chunks,
      - if output is weak, caller can continue to fallback extractor.
    """
    messages = _build_messages(question, graph_hints, chunks)
    content = _deepseek_chat(messages)
    parsed = _parse_llm_json(content)
    if not parsed:
        return None

    best_chunk = chunks[0] if chunks else {}

    # Blend optional fallback score to avoid overconfidence spikes.
    base = 0.55
    if fallback_answer and isinstance(fallback_answer.get("answer_score"), (int, float)):
        base = 0.5 + 0.25 * float(fallback_answer["answer_score"])
    final_score = max(0.0, min(1.0, 0.65 * parsed["confidence"] + 0.35 * base))

    return {
        "answer": parsed["answer"],
        "answer_score": final_score,
        "source_chunk_id": best_chunk.get("chunk_id", -1),
        "source_segment": best_chunk.get("segment_id", -1),
        "context": best_chunk.get("text", ""),
        "reasoning": "hybrid_llm",
        "llm_rationale": parsed.get("rationale", ""),
        "graph_failed": True,
    }
