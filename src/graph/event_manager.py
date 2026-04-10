
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from src.utils import load_json, save_json, load_chunks
from src.nlp.event_extraction import extract_structured_events

class EventManager:
    """
    Manages structured events and their temporal/causal dependencies across chunks.
    """
    def __init__(self, book_dir: Path):
        self.book_dir = book_dir
        self.events_file = book_dir / "structured_events.json"
        self.chains_file = book_dir / "event_chains.json"

    def process_book_events(self, chunks: List[Dict]):
        """Extract and link events across the entire book."""
        all_events = []
        for chunk in chunks:
            cid = chunk["chunk_id"]
            text = chunk.get("text", "")
            events = extract_structured_events(text)
            for ev in events:
                ev["chunk_id"] = cid
                ev["segment_id"] = chunk.get("segment_id", 0)
                all_events.append(ev)
        
        save_json(self.events_file, all_events)
        
        # Build temporal chains
        # Rule: Link event i to event j if |chunk_i - chunk_j| <= 5
        chains = []
        all_events.sort(key=lambda x: x["chunk_id"])
        
        for i in range(len(all_events)):
            for j in range(i + 1, len(all_events)):
                ev1 = all_events[i]
                ev2 = all_events[j]
                
                dist = ev2["chunk_id"] - ev1["chunk_id"]
                if dist > 5:
                    break
                
                # Check for potential causality: ev1 (Trigger) -> ev2 (Reaction)
                # Heuristic: if ev1 is a 'death' or 'attack' and ev2 is 'escape' or 'betrayal'
                is_causal = False
                trigger_types = ["death", "attack", "betrayal"]
                reaction_types = ["escape", "betrayal", "attack", "revelation"]
                
                if ev1["type"] in trigger_types and ev2["type"] in reaction_types:
                    is_causal = True
                
                chains.append({
                    "trigger_chunk": ev1["chunk_id"],
                    "reaction_chunk": ev2["chunk_id"],
                    "trigger_type": ev1["type"],
                    "reaction_type": ev2["type"],
                    "is_causal": is_causal,
                    "distance": dist
                })
        
        save_json(self.chains_file, chains)
        return all_events, chains

def load_event_context(book_dir: Path):
    """Load preprocessed event data for a book."""
    events = load_json(book_dir / "structured_events.json") or []
    chains = load_json(book_dir / "event_chains.json") or []
    
    # Map chunk_id to its events
    chunk_to_events = defaultdict(list)
    for ev in events:
        chunk_to_events[ev["chunk_id"]].append(ev)
        
    # Map reaction_chunk to its trigger_chunks
    reaction_to_triggers = defaultdict(list)
    for chain in chains:
        reaction_to_triggers[chain["reaction_chunk"]].append(chain["trigger_chunk"])
        
    return {
        "events": events,
        "chains": chains,
        "chunk_to_events": chunk_to_events,
        "reaction_to_triggers": reaction_to_triggers
    }
