#####
# This code was written by Josh Edwards for CS 4080 Group Project
# This code was generated using Claude 3.5 Sonnet 
# The code has been modified to implement random geometric graphs, print the graph and save it in three different formats
#   - Adjacency list
#   - JSON
#   - Edge list
#
# The code has also been modified to save the graph in the specified format and save it in the current directory
# The code has also been modified to generate graphs of various sizes, from 10 to 1000 nodes
#
#####  

import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from typing import Tuple, Literal
import pickle

OutputFormat = Literal['adj', 'json', 'edge']

# Create a random geometric graph with given number of nodes and radius.
def create_random_geometric_graph(n_nodes: int, radius: float) -> Tuple[nx.Graph, dict]:
    """
    Create a random geometric graph with given number of nodes and radius.
    
    Args:
        n_nodes: Number of nodes in the graph
        radius: Distance threshold for connecting nodes
        
    Returns:
        Tuple of (Graph, position dictionary)
    """
    # Create the random geometric graph
    G = nx.random_geometric_graph(n_nodes, radius)
    
    # Get the position dictionary
    pos = nx.get_node_attributes(G, 'pos')
    
    # Add weights based on Euclidean distances
    for (u, v) in G.edges():
        dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        G[u][v]['weight'] = dist
    
    return G, pos

# Save the graph data in the specified format
def save_graph_data(G: nx.Graph, pos: dict, base_filename: str, output_dir: str, format: OutputFormat = 'adj'):
    """
    Save the graph data in the specified format
    
    Args:
        G: NetworkX graph
        pos: Position dictionary
        base_filename: Base name for the files
        output_dir: Directory to save the files
        format: Output format ('adj', 'json', or 'edge')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if format == 'adj':
            # Save as adjacency list with weights
            output_path = os.path.join(output_dir, f"{base_filename}.adj")
            with open(output_path, 'w') as f:
                # First line: number of nodes and edges
                f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
                # Write node positions
                for node in G.nodes():
                    x, y = pos[node]
                    f.write(f"n {node} {x:.6f} {y:.6f}\n")
                # Write edges with weights
                for u, v, data in G.edges(data=True):
                    f.write(f"e {u} {v} {data['weight']:.6f}\n")
        
        elif format == 'json':
            # Save as JSON
            output_path = os.path.join(output_dir, f"{base_filename}.json")
            data = {
                'nodes': {str(node): {'pos': [float(x), float(y)]} 
                         for node, (x, y) in pos.items()},
                'edges': [{
                    'source': str(u),
                    'target': str(v),
                    'weight': float(data['weight'])
                } for u, v, data in G.edges(data=True)]
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'edge':
            # Save as edge list
            output_path = os.path.join(output_dir, f"{base_filename}.edge")
            with open(output_path, 'w') as f:
                # Header with node positions
                f.write("# Node positions:\n")
                for node, (x, y) in pos.items():
                    f.write(f"# {node}: {x:.6f} {y:.6f}\n")
                f.write("# Edge list (source target weight):\n")
                # Write edges
                for u, v, data in G.edges(data=True):
                    f.write(f"{u} {v} {data['weight']:.6f}\n")
        
    except Exception as e:
        print(f"Error saving graph {base_filename}: {str(e)}")
        raise

# Visualize and save the graph as an image
def visualize_and_save_graph(G: nx.Graph, pos: dict, base_filename: str, output_dir: str):
    """
    Visualize and save the graph as an image
    
    Args:
        G: NetworkX graph
        pos: Position dictionary
        base_filename: Base name for the file
        output_dir: Directory to save the file
    """
    plt.figure(figsize=(12, 12))
    plt.title(f"Random Geometric Graph (n={G.number_of_nodes()})")
    
    # Draw the graph
    nx.draw(G, pos,
           node_color='lightblue',
           node_size=100,
           edge_color='gray',
           width=0.5,
           with_labels=False)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Generate random geometric graphs of various sizes
def generate_graphs(format: OutputFormat = 'adj'):
    """
    Generate random geometric graphs of various sizes
    
    Args:
        format: Output format ('adj', 'json', or 'edge')
    """
    # Define the sizes of graphs to generate
    sizes = [10] + list(range(100, 1100, 100))
    
    # Create output directory in current directory
    output_dir = os.path.join(".", f"random_geometric_graphs_{format}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store graph information
    graph_info = []
    
    for n in sizes:
        print(f"Generating graph with {n} nodes...")
        
        # Adjust radius based on number of nodes to maintain reasonable density
        radius = np.sqrt(np.log(n) / n)
        
        # Create the graph
        G, pos = create_random_geometric_graph(n, radius)
        
        # Create base filename
        base_filename = f"random_geometric_{n}"
        
        # Save graph data in specified format
        save_graph_data(G, pos, base_filename, output_dir, format)
        
        # Visualize and save graph
        visualize_and_save_graph(G, pos, base_filename, output_dir)
        
        # Collect graph information
        info = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes(),
            'radius': radius
        }
        graph_info.append(info)
        
        print(f"Graph with {n} nodes saved successfully")
        print(f"Edges: {info['edges']}")
        print(f"Average degree: {info['avg_degree']:.2f}")
        print("-" * 40)
    
    # Save summary information
    with open(os.path.join(output_dir, "graph_summary.txt"), 'w') as f:
        f.write("Random Geometric Graphs Summary\n")
        f.write("=" * 30 + "\n\n")
        for info in graph_info:
            f.write(f"Nodes: {info['nodes']}\n")
            f.write(f"Edges: {info['edges']}\n")
            f.write(f"Average degree: {info['avg_degree']:.2f}\n")
            f.write(f"Radius parameter: {info['radius']:.4f}\n")
            f.write("-" * 20 + "\n\n")

if __name__ == "__main__":
    # Get format choice from user
    while True:
        format_choice = input("Choose output format (adj/json/edge): ").lower()
        if format_choice in ['adj', 'json', 'edge']:
            break
        print("Invalid choice. Please choose 'adj', 'json', or 'edge'.")
    
    generate_graphs(format_choice)