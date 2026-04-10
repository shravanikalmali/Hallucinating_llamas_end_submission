#!/usr/bin/env python3
"""Graph audit + visualization utility.

Usage examples:
  python experiments/graph_audit.py \
    --processed-dir /path/to/processed --book <book_id>

  python experiments/graph_audit.py \
    --processed-dir /path/to/processed \
    --questions-json narrativeqa_evaluation_results.json

Optional outputs:
  --save-png   Save a 50-node spring-layout PNG per book
  --save-html  Save an interactive PyVis HTML per book (if pyvis installed)
"""

from __future__ import annotations

import argparse
import json
import re
import importlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx


NOISE_REGEXES = [
    re.compile(r"^(chapter|book|part|section)\b", re.I),
    re.compile(r"^[ivxlcdm]+$", re.I),
    re.compile(r"^\d+$"),
]

NOISE_TERMS = {
    "he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
    "i", "me", "my", "we", "us", "you", "your", "the man", "the woman",
    "man", "woman", "boy", "girl", "mr", "mrs", "ms", "sir", "lady", "gentleman",
    "room", "house", "street",
}


def is_noisy_node(name: str) -> bool:
    n = (name or "").strip()
    nl = n.lower().strip(" .,:;!?\"'`()[]{}")
    if not nl:
        return True
    if nl in NOISE_TERMS:
        return True
    if len(nl) <= 2 and nl not in {"uk", "us"}:
        return True
    return any(rx.match(nl) for rx in NOISE_REGEXES)


def _parse_books_from_questions_json(path: Path) -> list[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    books = []
    for item in items:
        b = item.get("book_id")
        if b and b not in books:
            books.append(b)
    return books


def _entity_label_map(book_dir: Path) -> dict[str, str]:
    """Map canonical entity -> dominant NER label using entities.json + coref map."""
    entities_file = book_dir / "entities.json"
    filtered_file = book_dir / "filtered_chunk_entities.json"
    coref_file = book_dir / "coref_map.json"

    if not entities_file.exists() or not filtered_file.exists():
        return {}

    coref = {}
    if coref_file.exists():
        try:
            coref = json.loads(coref_file.read_text(encoding="utf-8"))
        except Exception:
            coref = {}

    filtered = json.loads(filtered_file.read_text(encoding="utf-8"))
    filtered_by_chunk = {x["chunk_id"]: set(x.get("entities", [])) for x in filtered}

    raw = json.loads(entities_file.read_text(encoding="utf-8"))
    votes: dict[str, Counter] = defaultdict(Counter)

    for item in raw:
        cid = item.get("chunk_id")
        canonical_set = filtered_by_chunk.get(cid, set())
        if not canonical_set:
            continue

        for ent in item.get("entities", []):
            if not isinstance(ent, dict):
                continue
            txt = (ent.get("text") or "").strip()
            lbl = (ent.get("label") or "").strip().upper()
            if not txt or not lbl:
                continue

            cand = coref.get(txt, txt)
            if cand in canonical_set:
                votes[cand][lbl] += 1
                continue

            # fallback: best lexical match among canonical entities in this chunk
            tnorm = txt.lower()
            for c in canonical_set:
                cnorm = c.lower()
                if tnorm == cnorm or tnorm in cnorm or cnorm in tnorm:
                    votes[c][lbl] += 1

    label_map = {}
    for entity, counter in votes.items():
        if counter:
            label_map[entity] = counter.most_common(1)[0][0]
    return label_map


def _audit_graph(book_dir: Path) -> dict:
    graph_path = book_dir / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")

    G = nx.read_graphml(graph_path)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    isolates = list(nx.isolates(G))
    weak_edges = [(u, v, d) for u, v, d in G.edges(data=True) if float(d.get("weight", 1)) <= 1.0]
    noisy_nodes = [x for x in G.nodes() if is_noisy_node(str(x))]
    top_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    edge_rel_count = sum(1 for _, _, d in G.edges(data=True) if d.get("relation"))

    weights = [int(float(d.get("weight", 1))) for _, _, d in G.edges(data=True)]
    weight_hist = dict(sorted(Counter(weights).items()))

    components = list(nx.connected_components(G)) if n_nodes > 0 else []
    component_sizes = sorted((len(c) for c in components), reverse=True)

    return {
        "book_id": book_dir.name,
        "nodes": n_nodes,
        "edges": n_edges,
        "isolates": len(isolates),
        "isolates_ratio": (len(isolates) / n_nodes) if n_nodes else 0.0,
        "weak_edges": len(weak_edges),
        "weak_edges_ratio": (len(weak_edges) / n_edges) if n_edges else 0.0,
        "noisy_nodes": len(noisy_nodes),
        "noisy_nodes_ratio": (len(noisy_nodes) / n_nodes) if n_nodes else 0.0,
        "top_degree": top_degree,
        "edge_relation_coverage": (edge_rel_count / n_edges) if n_edges else 0.0,
        "edge_weight_hist": weight_hist,
        "components": len(components),
        "largest_component": (component_sizes[0] if component_sizes else 0),
        "weak_edge_examples": [(u, v) for u, v, _ in weak_edges[:30]],
        "isolated_node_examples": [str(x) for x in isolates[:30]],
        "noisy_node_examples": [str(x) for x in noisy_nodes[:30]],
    }


def _save_png(book_dir: Path, out_dir: Path, max_nodes: int = 50) -> Path | None:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return None

    G = nx.read_graphml(book_dir / "graph.graphml")
    labels = _entity_label_map(book_dir)

    nodes = list(G.nodes())[:max_nodes]
    H = G.subgraph(nodes).copy()

    colors = []
    for n in H.nodes():
        lbl = labels.get(str(n), "")
        if lbl == "PERSON":
            colors.append("red")
        elif lbl in {"GPE", "LOC", "FAC"}:
            colors.append("green")
        elif lbl:
            colors.append("blue")
        else:
            colors.append("gray")

    pos = nx.spring_layout(H, k=0.5, seed=42)
    plt.figure(figsize=(12, 12))
    nx.draw(H, pos, node_color=colors, with_labels=True, node_size=450, font_size=7)
    out_path = out_dir / f"{book_dir.name}_graph_preview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _save_pyvis(book_dir: Path, out_dir: Path) -> Path | None:
    try:
        Network = importlib.import_module("pyvis.network").Network
    except Exception:
        return None

    G = nx.read_graphml(book_dir / "graph.graphml")
    labels = _entity_label_map(book_dir)

    net = Network(height="900px", width="100%", notebook=False)
    for n, data in G.nodes(data=True):
        lbl = labels.get(str(n), "")
        color = "#ef4444" if lbl == "PERSON" else "#22c55e" if lbl in {"GPE", "LOC", "FAC"} else "#3b82f6"
        net.add_node(str(n), label=str(n), title=f"label={lbl or 'UNKNOWN'}", color=color)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        net.add_edge(str(u), str(v), value=w, title=f"weight={w}")

    out_path = out_dir / f"{book_dir.name}_graph.html"
    net.show(str(out_path))
    return out_path


def _iter_books(processed_dir: Path, books: list[str] | None) -> Iterable[Path]:
    if books:
        for b in books:
            yield processed_dir / b
    else:
        for d in sorted(processed_dir.iterdir()):
            if d.is_dir() and (d / "graph.graphml").exists():
                yield d


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate graph quality and optionally visualize it.")
    ap.add_argument("--processed-dir", required=True, help="Processed directory containing per-book graph.graphml")
    ap.add_argument("--book", action="append", default=[], help="Book ID (repeatable)")
    ap.add_argument("--questions-json", help="JSON file with items containing book_id; audits those books")
    ap.add_argument("--save-png", action="store_true", help="Save 50-node PNG previews")
    ap.add_argument("--save-html", action="store_true", help="Save PyVis HTML previews if pyvis installed")
    ap.add_argument("--out", default="/tmp/graph_audit_report.json", help="Output JSON path")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_path = Path(args.out)

    books = list(args.book)
    if args.questions_json:
        books = _parse_books_from_questions_json(Path(args.questions_json))

    viz_dir = out_path.parent / "graph_viz"
    if args.save_png or args.save_html:
        viz_dir.mkdir(parents=True, exist_ok=True)

    report = {"books": [], "aggregate": {}}

    for book_dir in _iter_books(processed, books if books else None):
        if not (book_dir / "graph.graphml").exists():
            continue
        metrics = _audit_graph(book_dir)
        metrics["png_preview"] = None
        metrics["html_preview"] = None

        if args.save_png:
            p = _save_png(book_dir, viz_dir)
            metrics["png_preview"] = str(p) if p else None

        if args.save_html:
            p = _save_pyvis(book_dir, viz_dir)
            metrics["html_preview"] = str(p) if p else None

        report["books"].append(metrics)

        print("\n---")
        print(book_dir.name)
        print(f"nodes={metrics['nodes']} edges={metrics['edges']} components={metrics['components']}")
        print(f"isolates={metrics['isolates']} ({metrics['isolates_ratio']:.2%})")
        print(f"weak_edges(w<=1)={metrics['weak_edges']} ({metrics['weak_edges_ratio']:.2%})")
        print(f"noisy_nodes={metrics['noisy_nodes']} ({metrics['noisy_nodes_ratio']:.2%})")
        print(f"edge_relation_coverage={metrics['edge_relation_coverage']:.2%}")
        print(f"top_degree={metrics['top_degree'][:5]}")

    rows = report["books"]
    if rows:
        report["aggregate"] = {
            "book_count": len(rows),
            "avg_isolates_ratio": sum(r["isolates_ratio"] for r in rows) / len(rows),
            "avg_weak_edges_ratio": sum(r["weak_edges_ratio"] for r in rows) / len(rows),
            "avg_noisy_nodes_ratio": sum(r["noisy_nodes_ratio"] for r in rows) / len(rows),
            "avg_relation_coverage": sum(r["edge_relation_coverage"] for r in rows) / len(rows),
        }
        print("\n=== AGGREGATE ===")
        print(report["aggregate"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
