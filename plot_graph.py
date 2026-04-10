
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_graph(book_id, processed_dir, output_file):
    graph_path = Path(processed_dir) / book_id / "graph.graphml"
    if not graph_path.exists():
        print(f"Error: Graph not found at {graph_path}")
        return

    G = nx.read_graphml(str(graph_path))
    
    # Filter for top nodes by degree to keep the plot readable
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, degree in sorted_nodes[:40]]
    subgraph = G.subgraph(top_nodes)

    plt.figure(figsize=(15, 12))
    
    # Define colors based on type
    node_colors = []
    for node, data in subgraph.nodes(data=True):
        data_vals = [str(v).lower() for v in data.values()]
        if "event" in data_vals:
            node_colors.append('#ADD8E6') # Light Blue for events
        else:
            node_colors.append('#FFB6C1') # Light Pink for entities
            
    # Layout
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    
    # Draw
    nx.draw(subgraph, pos, 
            with_labels=True, 
            node_color=node_colors, 
            node_size=2500, 
            font_size=10, 
            font_weight='bold', 
            edge_color='#D3D3D3',
            width=1.5,
            alpha=0.9)
            
    plt.title(f"Knowledge Graph: {book_id}", fontsize=15)
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Graph saved as {output_file}")

if __name__ == "__main__":
    book_id = "c8e25c068b7c8a00ba00096e73ce7ea893c69aba"
    pdir = "/home/shravanikalmali/Desktop/inlp_final/drive-download-20260408T131838Z-3-001"
    output = "static_graph_visualization.png"
    plot_graph(book_id, pdir, output)
