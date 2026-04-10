"""
LLM Answering Module

Uses a local generative model (Flan-T5) to format and finalize answers
based on the retrieved context and user query.
"""

import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any

from src.config import (
    LLM_MODEL_NAME, LLM_MAX_CONTEXT_LENGTH, 
    OPENROUTER_API_KEY, OPENROUTER_MODEL
)

_llm_model = None
_llm_tokenizer = None

def get_llm():
    """Load and cache the local LLM model and tokenizer."""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        print(f"Loading Local LLM model '{LLM_MODEL_NAME}' ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)
    return _llm_model, _llm_tokenizer

def generate_llm_answer(question: str, context_chunks: List[Dict[str, Any]], reasoning_hint: str = None) -> str:
    """
    Generate an answer using either OpenRouter (if key available) or local LLM.
    Includes an optional reasoning_hint (narrative facts/inferences).
    """
    # Combine top chunks into a single context string
    full_context = ""
    for chunk in context_chunks:
        full_context += chunk.get("text", "") + " "
    
    # Truncate context to avoid token limits
    truncated_context = full_context[:LLM_MAX_CONTEXT_LENGTH]
    
    hint_str = f"\nFact to consider: {reasoning_hint}\n" if reasoning_hint else ""
    
    # ── Path A: OpenRouter ──────────────────────────────────────────────────
    if OPENROUTER_API_KEY and not OPENROUTER_API_KEY.startswith("YOUR_"):
        try:
            print(f"Calling OpenRouter ({OPENROUTER_MODEL}) ...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a factual narrative assistant. Answer ONLY using the provided context. If the answer is not explicitly present in the text, say 'NOT FOUND'. Return a short phrase only. Do not explain."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{truncated_context}\n{hint_str}\nQuestion: {question}\n\nAnswer:"
                        }
                    ]
                }),
                timeout=20
            )
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                print(f"OpenRouter Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"OpenRouter Request failed: {e}")
            
    # ── Path B: Local T5 (Fallback) ──────────────────────────────────────────
    print("Falling back to local T5 model ...")
    model, tokenizer = get_llm()
    device = next(model.parameters()).device
    
    input_text = f"Context: {truncated_context}\n{hint_str}\nQuestion: {question}\nAnswer using context only or say 'NOT FOUND':"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def postprocess_llm_answer(answer: str) -> str | None:
    """Kill long answers or full sentences to ensure phrase-only extraction."""
    if not answer:
        return None
    answer = answer.strip()
    
    # reject long answers (> 8 words)
    if len(answer.split()) > 8:
        return None

    # reject full sentences ending with '.'
    if "." in answer:
        return None

    return answer

def llm_extract_answer(
    question: str, final_chunks: List[Dict[str, Any]], 
) -> Dict[str, Any]:
    """
    Generate answer using LLM and apply strict phrase-based filtering.
    """
    if not final_chunks:
        return {"answer": "NOT FOUND", "answer_score": 0.0, "reasoning": "no_chunks"}
    
    # Generate raw answer
    raw_answer = generate_llm_answer(question, final_chunks)
    
    # Filter for phrase-only output
    filt_answer = postprocess_llm_answer(raw_answer)
    
    if not filt_answer:
        return {
            "answer": "NOT FOUND",
            "answer_score": 0.0,
            "reasoning": "llm_invalid_filter",
            "source_chunk_id": final_chunks[0]["chunk_id"],
            "source_segment": final_chunks[0]["segment_id"],
            "context": final_chunks[0].get("text", "")
        }
        
    return {
        "answer": filt_answer,
        "answer_score": 1.0, 
        "source_chunk_id": final_chunks[0]["chunk_id"],
        "source_segment": final_chunks[0]["segment_id"],
        "context": final_chunks[0].get("text", ""),
        "reasoning": "llm_generation_phrase"
    }
