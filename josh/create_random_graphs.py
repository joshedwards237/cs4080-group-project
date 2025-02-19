import networkx as nx
import numpy as np
from typing import Tuple
import random
import matplotlib.pyplot as plt
import os
import pickle

def create_highway_dimension_graph(n_nodes: int, is_high_dim: bool = False) -> Tuple[nx.Graph, dict]:
    """
    Create a graph with either high or low highway dimension.
    Low highway dimension: Similar to road networks with clear hierarchical structure
    High highway dimension: More random connections across different scales
    
    Args:
        n_nodes: Number of nodes
        is_high_dim: If True, creates high highway dimension graph
    
    Returns:
        Tuple of (Graph, position dictionary)
    """
    G = nx.Graph()
    
    # Create positions in 2D space
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    
    if is_high_dim:
        # High highway dimension: More random long-range connections
        # First create a base connected graph
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
                # Add edges with probability inversely proportional to distance
                if random.random() < 0.1 / (dist + 0.1):
                    G.add_edge(i, j, weight=dist)
    else:
        # Low highway dimension: Hierarchical structure like road networks
        # First create local connections
        for i in range(n_nodes):
            nearest = []
            for j in range(n_nodes):
                if i != j:
                    dist = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
                    nearest.append((dist, j))
            # Connect to closest neighbors
            nearest.sort()
            for dist, j in nearest[:3]:  # Connect to 3 nearest neighbors
                G.add_edge(i, j, weight=dist)
        
        # Add some "highway" connections between clusters
        centers = random.sample(range(n_nodes), n_nodes // 20)  # 5% of nodes are hubs
        for i in centers:
            for j in centers:
                if i != j:
                    dist = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
                    G.add_edge(i, j, weight=dist)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
            G.add_edge(node1, node2, weight=dist)
    
    return G, pos

def create_doubling_dimension_graph(n_nodes: int, is_high_dim: bool = False) -> Tuple[nx.Graph, dict]:
    """
    Create a graph with either high or low doubling dimension.
    Low doubling dimension: Points well-separated in metric space
    High doubling dimension: Points clustered with varying densities
    
    Args:
        n_nodes: Number of nodes
        is_high_dim: If True, creates high doubling dimension graph
    
    Returns:
        Tuple of (Graph, position dictionary)
    """
    G = nx.Graph()
    
    if is_high_dim:
        # High doubling dimension: Create clusters with varying densities
        clusters = []
        remaining_nodes = n_nodes
        while remaining_nodes > 0:
            size = min(random.randint(10, 50), remaining_nodes)
            center = (random.uniform(0, 1), random.uniform(0, 1))
            radius = random.uniform(0.05, 0.2)
            clusters.append((center, radius, size))
            remaining_nodes -= size
        
        # Generate positions based on clusters
        pos = {}
        node_idx = 0
        for center, radius, size in clusters:
            for _ in range(size):
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(0, radius)
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                pos[node_idx] = (x, y)
                node_idx += 1
    else:
        # Low doubling dimension: More uniform distribution
        pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    
    # Add edges based on distances
    for i in range(n_nodes):
        nearest = []
        for j in range(n_nodes):
            if i != j:
                dist = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
                nearest.append((dist, j))
        nearest.sort()
        # Connect to more neighbors in high doubling dimension
        k = 8 if is_high_dim else 4
        for dist, j in nearest[:k]:
            G.add_edge(i, j, weight=dist)
    
    return G, pos

def create_density_graph(n_nodes: int, is_dense: bool = False) -> Tuple[nx.Graph, dict]:
    """
    Create either a dense or sparse graph.
    Dense: Many edges per node
    Sparse: Few edges per node
    
    Args:
        n_nodes: Number of nodes
        is_dense: If True, creates dense graph
    
    Returns:
        Tuple of (Graph, position dictionary)
    """
    # Create positions in 2D space
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    
    if is_dense:
        # Dense graph: probability of edge ~ 1/log(n)
        G = nx.random_geometric_graph(n_nodes, radius=1/np.log(n_nodes), pos=pos)
    else:
        # Sparse graph: probability of edge ~ 1/n
        G = nx.random_geometric_graph(n_nodes, radius=1/n_nodes, pos=pos)
    
    # Add weights based on distances
    for (u, v) in G.edges():
        dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        G[u][v]['weight'] = dist
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
            G.add_edge(node1, node2, weight=dist)
    
    return G, pos

def save_graph(G: nx.Graph, pos: dict, base_filename: str, output_dir: str):
    """
    Save graph and positions in multiple formats
    
    Args:
        G: NetworkX graph
        pos: Position dictionary
        base_filename: Base name for the files (without extension)
        output_dir: Directory to save the files
    """
    try:
        # Save as pickle (with positions)
        pickle_path = os.path.join(output_dir, f"{base_filename}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump((G, pos), f)
        
        # Add positions as node attributes for NetworkX formats
        G_with_pos = G.copy()
        nx.set_node_attributes(G_with_pos, pos, 'pos')
        
        # Save as GraphML
        graphml_path = os.path.join(output_dir, f"{base_filename}.graphml")
        nx.write_graphml(G_with_pos, graphml_path)
        
        # Save as GML
        gml_path = os.path.join(output_dir, f"{base_filename}.gml")
        nx.write_gml(G_with_pos, gml_path)
    except Exception as e:
        print(f"Error saving graph {base_filename}: {str(e)}")
        raise

def create_all_graphs(n_nodes: int, output_dir: str = "graphs"):
    """
    Create all types of graphs and save them to files.
    
    Args:
        n_nodes: Number of nodes for each graph
        output_dir: Directory to save the graph files
    """
    if n_nodes < 1:
        raise ValueError("Number of nodes must be positive")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all graph types
    graph_generators = [
        (create_highway_dimension_graph, "highway", [False, True]),
        (create_doubling_dimension_graph, "doubling", [False, True]),
        (create_density_graph, "density", [False, True])
    ]
    
    graph_info = []  # Store information about generated graphs
    
    for generator, name, options in graph_generators:
        for option in options:
            G, pos = generator(n_nodes, option)
            desc = "high" if option else "low"
            if name == "density":
                desc = "dense" if option else "sparse"
            
            # Create base filename
            base_filename = f"{desc}_{name}_{n_nodes}"
            
            # Save graph in multiple formats
            save_graph(G, pos, base_filename, output_dir)
            
            # Collect graph information
            info = {
                'type': f"{desc} {name}",
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
                'base_filename': base_filename
            }
            graph_info.append(info)
            
            # Visualize a small subset and save the plot
            if G.number_of_nodes() > 100:
                subset_nodes = random.sample(list(G.nodes()), 100)
                subset_G = G.subgraph(subset_nodes)
                subset_pos = {node: pos[node] for node in subset_nodes}
                plt.figure(figsize=(10, 10))
                plt.title(f"{desc} {name} graph (100 node sample)")
                nx.draw(subset_G, subset_pos, 
                       node_size=50, 
                       node_color='lightblue',
                       edge_color='gray',
                       width=0.5,
                       with_labels=False)
                plt.savefig(os.path.join(output_dir, f"{base_filename}_sample.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Generated and saved {desc} {name} graph")
    
    # Save graph information to a summary file
    summary_file = os.path.join(output_dir, "graph_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Generated Graphs (n={n_nodes})\n")
        f.write("=" * 50 + "\n\n")
        for info in graph_info:
            f.write(f"Type: {info['type']}\n")
            f.write(f"Nodes: {info['nodes']}\n")
            f.write(f"Edges: {info['edges']}\n")
            f.write(f"Average degree: {info['avg_degree']:.2f}\n")
            f.write(f"Files:\n")
            f.write(f"  - {info['base_filename']}.pkl (Python pickle)\n")
            f.write(f"  - {info['base_filename']}.graphml (GraphML format)\n")
            f.write(f"  - {info['base_filename']}.gml (GML format)\n")
            f.write("-" * 30 + "\n\n")
    
    return graph_info

def load_graph(filename: str) -> Tuple[nx.Graph, dict]:
    """
    Load graph and positions from a file
    Supports .pkl, .graphml, and .gml formats
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Graph file not found: {filename}")
        
    try:
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        elif filename.endswith('.graphml'):
            G = nx.read_graphml(filename)
            pos = nx.get_node_attributes(G, 'pos')
            return G, pos
        elif filename.endswith('.gml'):
            G = nx.read_gml(filename)
            pos = nx.get_node_attributes(G, 'pos')
            return G, pos
        else:
            raise ValueError("Unsupported file format. Use .pkl, .graphml, or .gml")
    except Exception as e:
        print(f"Error loading graph from {filename}: {str(e)}")
        raise

if __name__ == "__main__":
    # Create graphs of different sizes
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for size in sizes:
        print(f"\nGenerating graphs with {size} nodes...")
        output_dir = f"graphs_{size}"
        graph_info = create_all_graphs(size, output_dir)
        
        print(f"\nGenerated graphs saved in {output_dir}/")
        print("Summary of generated graphs:")
        for info in graph_info:
            print(f"\n{info['type']}:")
            print(f"  Nodes: {info['nodes']}")
            print(f"  Edges: {info['edges']}")
            print(f"  Average degree: {info['avg_degree']:.2f}")
