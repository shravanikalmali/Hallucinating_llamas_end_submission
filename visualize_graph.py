
import networkx as nx
import os
from pathlib import Path

def generate_mermaid():
    path = "/home/shravanikalmali/Desktop/inlp_final/drive-download-20260408T131838Z-3-001/c8e25c068b7c8a00ba00096e73ce7ea893c69aba/graph.graphml"
    G = nx.read_graphml(path)
    
    # Top 15 nodes by degree
    sorted_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:15]]
    
    print("graph TD")
    # Styles
    print("    classDef char fill:#f9f,stroke:#333,stroke-width:2px;")
    print("    classDef loc fill:#dfd,stroke:#333,stroke-width:2px;")
    
    for node in sorted_nodes:
        data = G.nodes[node]
        clean = node.replace(" ", "_").replace('"', "").replace("'", "")
        ntype = str(data.get('d0', 'entity')).lower()
        
        icon = "👤" if ntype == "entity" else "🎬"
        print(f'    {clean}["{icon} {node}"]')
        
    seen_edges = set()
    for u, v, data in G.edges(data=True):
        if u in sorted_nodes and v in sorted_nodes:
            edge = tuple(sorted((u, v)))
            if edge not in seen_edges:
                clean_u = u.replace(" ", "_").replace('"', "").replace("'", "")
                clean_v = v.replace(" ", "_").replace('"', "").replace("'", "")
                print(f"    {clean_u} --- {clean_v}")
                seen_edges.add(edge)
                if len(seen_edges) > 30: # Limit edges for readability
                    break

if __name__ == "__main__":
    generate_mermaid()
