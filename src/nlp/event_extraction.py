"""
Event Extraction for Narrative Graph.

Identifies key narrative events (betrayal, escape, death, revelation, etc.) 
within chunks using keyword lexicons and linguistic patterns.
"""

import re
from typing import Dict, List, Set

import re
import spacy
from typing import Dict, List, Set, Optional

# Load spaCy once
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except:
            _nlp = spacy.load("en_core_web_md")
    return _nlp

# Event lexicons mapped to core event types
EVENT_LEXICONS = {
    "death": {"kill", "murder", "slay", "die", "death", "execute", "execution", "slain", "perish", "suicide", "hang", "behead"},
    "escape": {"escape", "flee", "fled", "rescue", "liberate", "rescued", "run away", "slip", "broke away"},
    "attack": {"attack", "fight", "battle", "strike", "smite", "assault", "ambush", "vengeance", "revenge", "kill", "war"},
    "betrayal": {"betray", "traitor", "treachery", "deceive", "trick", "abandon", "forsake", "desert"},
    "revelation": {"reveal", "discover", "secret", "truth", "uncover", "confess", "learn", "realize", "know"},
    "loyalty": {"honor", "loyal", "loyalty", "oath", "vow", "pledge", "trust", "faithful", "refuse", "witness"},
    "affiliation": {"marry", "ally", "friend", "helper", "join", "help", "save", "serve", "allegiance"},
}

def extract_structured_events(text: str) -> List[Dict]:
    """
    Extract (Subject, Action, Object) triples associated with specific event types.
    Example output: {"who": "Phil", "action": "refused", "target": "testify", "type": "betrayal"}
    """
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp(text)
    events = []

    for sent in doc.sents:
        for token in sent:
            # Check if verb is in any of our lexicons
            verb_lemma = token.lemma_.lower()
            event_type = None
            for et, keywords in EVENT_LEXICONS.items():
                if verb_lemma in keywords:
                    event_type = et
                    break
            
            if event_type:
                # Find Subject (nsubj) and Object (dobj or pobj or prep)
                subj = ""
                obj = ""
                
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = child.text
                    elif child.dep_ in ("dobj", "pobj", "attr"):
                        obj = child.text
                    elif child.dep_ == "prep":
                        # Follow the preposition
                        for prep_child in child.children:
                            if prep_child.dep_ in ("pobj", "attr"):
                                obj = f"{child.text} {prep_child.text}"

                # Only include if we found at least a subject or object to anchor the event
                if subj or obj:
                    events.append({
                        "who": subj,
                        "action": token.text,
                        "target": obj,
                        "type": event_type,
                        "sentence": sent.text.strip(),
                        "pos": token.i
                    })
    
    return events

def extract_events_from_text(text: str) -> List[Dict]:
    """Maintain backward compatibility for keywords-only extraction."""
    structured = extract_structured_events(text)
    if not structured:
        return []
    
    # Group by type for legacy format
    from collections import defaultdict
    by_type = defaultdict(list)
    for ev in structured:
        by_type[ev["type"]].append(ev["action"])
        
    return [
        {"type": et, "keywords": sorted(set(acts)), "count": len(acts)}
        for et, acts in by_type.items()
    ]

def get_event_node_name(event_type: str, keywords: List[str] = None) -> str:
    """Standardize event node names."""
    return event_type.capitalize()
