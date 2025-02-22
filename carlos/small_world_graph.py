import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import json
import os


def generate_graphs(node_set=None, output_format=None):
    if output_format is None:
        output_format = ['adj', 'json', 'edge', 'graph']

    if node_set is None:
        node_set = [10, 100, 200, 300, 400, 500,
                    600, 700, 800, 900, 1000]

    output_dir = os.path.join(".", "random_small_world_graphs")
    os.makedirs(output_dir, exist_ok=True)

    graphs = generate_set_of_graphs(node_set)  # returns a tuple: [(graph, nodes, k, p, pos)]
    graph_info = []
    print("saving graphs \n")
    for graph in graphs:
        print(f"Generating graph with {graph[1]} nodes...")
        print(f"Saving graph with {graph[1]} nodes...")
        filename = f"random_small_world_{graph[1]}"

        save_graph_data(graph[0], graph[4], filename, output_dir, output_format)

        # Collect graph information
        info = {
            'nodes': graph[0].number_of_nodes(),
            'edges': graph[0].number_of_edges(),
            'avg_degree': 2 * graph[0].number_of_edges() / graph[0].number_of_nodes(),
            'connections_k': graph[2],
            'rewriting_p': graph[3]
        }
        graph_info.append(info)

        print(f"Graph with {info['nodes']} nodes saved successfully")
        print(f"Edges: {info['edges']}")
        print(f"Average degree: {info['avg_degree']:.2f}")
        print(f"Max number of connections: {info['connections_k']:.2f}")
        print(f"Rewriting probability: {info['rewriting_p']:.2f}")
        print("-" * 40)

    # Save summary information
    with open(os.path.join(output_dir, "graph_summary.txt"), 'w') as f:
        f.write("Random Small World Graphs Summary\n")
        f.write("=" * 30 + "\n\n")
        for info in graph_info:
            f.write(f"Nodes: {info['nodes']}\n")
            f.write(f"Edges: {info['edges']}\n")
            f.write(f"Average degree: {info['avg_degree']:.2f}\n")
            f.write(f"Max number of connections: {info['connections_k']:.2f}\n")
            f.write(f"Rewriting probability: {info['rewriting_p']:.2f}\n")
            f.write("-" * 20 + "\n\n")


def generate_set_of_graphs(node_set, random_k=None, random_p=None, random_weights=False, seed=None):
    """
    Generate Watts-Strogatz graphs with given n values.

    :param node_set: List of integers representing node counts for each graph
    :param random_weights: decide to use random weights or euclidian distance
    :param random_k: decide to use random value for connection number
    :param random_p: decide to use random value for rewriting probability
    :param seed: decide to use random value for a seed
    :return: List of generated graphs along with their parameters
    """
    graphs = []

    if seed is not None:
        random.seed(seed)

    for nodes in node_set:
        print(f"Generating graph with {nodes} nodes...\n")
        if nodes < 3:  # A valid Watts-Strogatz graph requires at least 3 nodes
            print(f"Skipping n={nodes}, must be at least 3.\n")
            continue

        if random_k is None:
            k = random.choice(range(2, int(nodes / 2), 2))  # Ensure k is even and < n
        else:
            k = random_k

        if random_p is None:
            p = random.uniform(0, 1)  # Rewiring probability
        else:
            p = random_p

        if seed is None:
            graph = nx.connected_watts_strogatz_graph(nodes, k, p)
            pos = nx.spring_layout(G=graph)
        else:
            graph = nx.connected_watts_strogatz_graph(nodes, k, p, seed=seed)
            pos = nx.spring_layout(G=graph, seed=seed)

        # Add weights to edges
        if random_weights:
            for u, v in graph.edges():
                # Random weight between 1 and total nodes divided by 2
                graph[u][v]['weight'] = random.uniform(1, nodes/2)
        else:
            for (u, v) in graph.edges():
                dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                graph[u][v]['weight'] = dist

        graphs.append((graph, nodes, k, p, pos))
        #graphs.append(G)

    return graphs


def draw_graphs_and_save(graph: nx.Graph, positions, filename, directory):
    plt.figure(figsize=(12, 12))
    plt.title(f"Random Small World Graph (n={graph.number_of_nodes()})")

    nx.draw(graph, pos=positions,
            with_labels=False, node_size=3, node_color="lightblue", edge_color="gray")
    # Save the plot
    plt.savefig(os.path.join(directory, f"{filename}.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def save_graph_data(graph: nx.Graph, positions, filename, directory, formats):
    """

    :param graph:
    :param positions:
    :param filename:
    :param directory:
    :param formats:
    :return:
    """

    os.makedirs(directory, exist_ok=True)
    pos = positions

    if 'adj' in formats:
        # Save as adjacency list with weights
        output_path = os.path.join(directory, f"{filename}.adj")
        with open(output_path, 'w') as f:
            # First line: number of nodes and edges
            f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
            # Write node positions
            for node in graph.nodes():
                x, y = pos[node]
                f.write(f"n {node} {x:.6f} {y:.6f}\n")
            # Write edges with weights
            for u, v, data in graph.edges(data=True):
                f.write(f"e {u} {v} {data['weight']:.6f}\n")

    if 'json' in formats:
        # Save as JSON
        output_path = os.path.join(directory, f"{filename}.json")
        data = {
            'nodes': {str(node): {'pos': [float(x), float(y)]}
                      for node, (x, y) in pos.items()},
            'edges': [{
                'source': str(u),
                'target': str(v),
                'weight': float(data['weight'])
            } for u, v, data in graph.edges(data=True)]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    if 'edge' in formats:
        # Save as edge list
        output_path = os.path.join(directory, f"{filename}.edge")
        with open(output_path, 'w') as f:
            # Header with node positions
            f.write("# Node positions:\n")
            for node, (x, y) in pos.items():
                f.write(f"# {node}: {x:.6f} {y:.6f}\n")
            f.write("# Edge list (source target weight):\n")
            # Write edges
            for u, v, data in graph.edges(data=True):
                f.write(f"{u} {v} {data['weight']:.6f}\n")

    if 'graph' in formats:
        # Save as edge list
        output_path = os.path.join(directory, f"{filename}.pickle")
        pickle.dump(graph, open(output_path, 'wb'))
        # load graph object from file
        # G = pickle.load(open('filename.pickle', 'rb'))

    draw_graphs_and_save(graph, pos, filename, directory)
