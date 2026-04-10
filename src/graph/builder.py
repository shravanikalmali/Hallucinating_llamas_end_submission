"""
Knowledge Graph Construction — Co-occurrence Graph

Builds a simple, robust co-occurrence graph (NetworkX undirected Graph):
  - Nodes: entities from filtered_chunk_entities.json
  - Edges: two entities co-occurring in the same chunk
    - weight = number of chunks they co-occur in
    - segments = list of segment_ids where co-occurrence happens
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

import networkx as nx

from src.config import find_data_directory
from src.utils import load_json, save_json, load_chunks


from src.nlp.event_extraction import extract_events_from_text, get_event_node_name


def build_graph_for_book(book_dir: Path) -> tuple[Optional[nx.Graph], list]:
    """Build a narrative graph for a single book.

    - Nodes: entities AND event types.
    - Edges: 
        1. Entity-Entity co-occurrence
        2. Entity-Event association
        3. Event-Event co-occurrence
    """
    fce_file = book_dir / "filtered_chunk_entities.json"
    chunks_file = book_dir / "chunks.jsonl"

    if not fce_file.exists() or not chunks_file.exists():
        return None, []

    chunk_entities = load_json(fce_file)
    chunks = load_chunks(chunks_file)
    chunks.sort(key=lambda x: x["chunk_id"])

    chunk_ent_map = {item["chunk_id"]: item["entities"] for item in chunk_entities}
    chunk_seg_map = {c["chunk_id"]: c["segment_id"] for c in chunks}

    # Collect entity -> segment mapping
    entity_segments: dict[str, set[int]] = defaultdict(set)
    event_segments: dict[str, set[int]] = defaultdict(set)
    
    # 2. Extract and Sequence Events
    from src.graph.event_manager import EventManager
    em = EventManager(book_dir)
    structured_events, event_chains = em.process_book_events(chunks)

    # 3. Build Graph
    G = nx.Graph()
    co_occurrence_records = []
    edge_data: dict[tuple[str, str], dict] = {}

    # Node tracking
    entity_segments: dict[str, set[int]] = defaultdict(set)
    event_counts = defaultdict(int)

    for chunk in tqdm(chunks, desc="  Building narrative graph", leave=False):
        cid = chunk["chunk_id"]
        sid = chunk_seg_map.get(cid, 0)
        
        # 1. Add Entity-Entity edges (legacy co-occurrence)
        entities = sorted(set(chunk_ent_map.get(cid, [])))
        for e in entities:
            entity_segments[e].add(sid)
            
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                n1, n2 = entities[i], entities[j]
                key = (n1, n2)
                if key not in edge_data:
                    edge_data[key] = {"weight": 0, "segments": [], "type": "entity_cooccurrence"}
                edge_data[key]["weight"] += 1
                edge_data[key]["segments"].append(sid)

        # 2. Add Entity-Event associations
        c_events = [ev for ev in structured_events if ev["chunk_id"] == cid]
        for ev in c_events:
            ev_name = ev["type"].capitalize()
            event_counts[ev_name] += 1
            
            # 2a. Link the explicit subject ('who') with a high weight
            subj = ev.get("who", "")
            if subj and subj in entities:
                key = tuple(sorted([subj, ev_name]))
                if key not in edge_data:
                    edge_data[key] = {"weight": 0, "segments": [], "type": "direct_narrative_association"}
                edge_data[key]["weight"] += 3 # Strengthening direct subject-event link
                edge_data[key]["segments"].append(sid)

            # 2b. Link other entities in chunk with normal weight
            for ent in entities:
                if ent == subj: continue
                key = tuple(sorted([ent, ev_name]))
                if key not in edge_data:
                    edge_data[key] = {"weight": 0, "segments": [], "type": "narrative_association"}
                edge_data[key]["weight"] += 1
                edge_data[key]["segments"].append(sid)

    # 3. Add Causal/Temporal Event-Event edges
    for chain in event_chains:
        n1 = chain["trigger_type"].capitalize()
        n2 = chain["reaction_type"].capitalize()
        key = tuple(sorted([n1, n2]))
        
        if key not in edge_data:
            edge_data[key] = {"weight": 0, "segments": [], "type": "causal_chain" if chain["is_causal"] else "temporal_flow"}
        
        edge_data[key]["weight"] += 1

    # Finalise Nodes
    for entity, segs in entity_segments.items():
        G.add_node(entity, type="entity", segments=",".join(str(s) for s in sorted(segs)))
    
    for event, count in event_counts.items():
        G.add_node(event, type="event", count=count)

    # Finalise Edges
    for (n1, n2), data in edge_data.items():
        unique_segs = sorted(set(data["segments"]))
        G.add_edge(n1, n2,
                   weight=data["weight"],
                   type=data["type"],
                   segments=",".join(str(s) for s in unique_segs))
        
        co_occurrence_records.append({
            "node1": n1, "node2": n2,
            "weight": data["weight"],
            "relation": data["type"],
            "segments": unique_segs,
        })

    return G, co_occurrence_records


def process_book(book_dir: Path, overwrite: bool = False) -> bool:
    """Process a single book's narrative graph."""
    book_id = book_dir.name
    graph_file = book_dir / "graph.graphml"

    if graph_file.exists() and not overwrite:
        print(f"  ℹ️  Skipping {book_id} — graph exists")
        return True

    print(f"\n── {book_id[:40]}{'…' if len(book_id) > 40 else ''}")

    G, co_occurrences = build_graph_for_book(book_dir)
    if G is None:
        return False

    nx.write_graphml(G, str(graph_file))
    save_json(book_dir / "narrative_edges.json", co_occurrences)
    save_json(book_dir / "graph_stats.json", {
        "nodes": G.number_of_nodes(),
        "entities": sum(1 for n, d in G.nodes(data=True) if d.get("type") == "entity"),
        "events": sum(1 for n, d in G.nodes(data=True) if d.get("type") == "event"),
        "edges": G.number_of_edges(),
    })

    print(f"    Graph: {G.number_of_nodes()} nodes ({G.number_of_edges()} edges)")
    return True


def run(book_id: str = None, processed_dir: str = None, overwrite: bool = False):
    """Main graph building entry point."""
    processed = find_data_directory(processed_dir)

    print("=" * 60)
    print("  Knowledge Graph — Co-occurrence")
    print("=" * 60)

    if book_id:
        book_dirs = [processed / book_id]
    else:
        book_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
        print(f"\n  Found {len(book_dirs)} book directories")

    success = 0
    for book_dir in tqdm(book_dirs, desc="Building graphs"):
        if process_book(book_dir, overwrite):
            success += 1

    print(f"\n{'='*60}\n  Graph Complete! ✅ {success} books\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_id", type=str, help="Book ID to process")
    parser.add_argument("--processed_dir", type=str, help="Directory containing processed books")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing graph")
    
    args = parser.parse_args()
    run(book_id=args.book_id, processed_dir=args.processed_dir, overwrite=args.overwrite)
