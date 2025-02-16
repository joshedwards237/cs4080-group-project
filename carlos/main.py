#####
# This code was generated using ChatGpt and a python module
# The model was trained using python.org and stackoverflow.com
# The code has been modified to implement larger graphs, print the graph and show visual traversal
#
# Three graphs are generated:
#   - The initial graph with nodes and weights
#   - The same graph with the fastest path between the selected nodes
#   - The first graph but as it is being traversed, the nodes "searched" and the path checked is highlighted in blue.
#       When the fastest path to the targe node is found, the path is highlighted in green.
#####
import matplotlib.pyplot as plt
import networkx as nx
import random

import dijkstra
#import bi_dijkstra
#import a_star


###
# Generates a random weighted adjacency list representation of an undirected graph.
###
def generate_random_adjacency_list(num_nodes, max_edges_per_node, max_weight=20):
    adj_list = {str(i): {} for i in range(num_nodes)}  # Create empty adjacency list

    for node in adj_list:
        num_edges = random.randint(1, max_edges_per_node)  # Random number of edges
        possible_neighbors = list(adj_list.keys())  # All possible neighbors
        possible_neighbors.remove(node)  # Remove self-loops

        # Select random neighbors
        neighbors = random.sample(possible_neighbors, min(num_edges, len(possible_neighbors)))

        for neighbor in neighbors:
            weight = random.randint(1, max_weight)  # Random weight
            adj_list[node][neighbor] = weight
            adj_list[neighbor][node] = weight  # Ensure undirected edges

    return adj_list


def draw_graph(graph, shortest_path=None, positions=None):
    """
    Draws the graph using networkx and matplotlib.

    Parameters:
        graph (networkx graph): The adjacency list representation of the graph.
        shortest_path (list): The shortest path to highlight (if provided).
        positions (dic):  node positions to make sure they are graphs are all the same
    """

    plt.figure(figsize=(12, 10))
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, positions, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=12, arrows=True)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=labels, font_size=12)

    # Highlight the shortest path if provided
    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(graph, positions, edgelist=path_edges, edge_color="red", width=2)

    plt.show()


if __name__ == '__main__':
    # gen large random graph
    num_nodes = 20  # number of nodes
    max_edges_per_node = 4  # graph sparsity
    max_weight = 20  # max weight between nodes
    start_node = '0'  # source node
    end_node = str(random.randint(1, num_nodes))  # target node
    adj_list = generate_random_adjacency_list(num_nodes, max_edges_per_node, max_weight)  # adjacency list

    # create Directed Graph
    graph = nx.DiGraph(adj_list)
    for node, neighbors in adj_list.items():
        for neighbor, weight in neighbors.items():
            graph.add_edge(node, neighbor, weight=weight)

    # create a static position of nodes
    pos = nx.spring_layout(graph, seed=42)
    fixed_nodes = adj_list.keys()
    positions = nx.spring_layout(graph, pos=pos, fixed=fixed_nodes, seed=42)

    # Print adjacency list
    for node, edges in list(adj_list.items())[:num_nodes]:  # Print first 10 nodes
        print(f"{node}: {edges}")

    # Draw the original graph
    draw_graph(graph, positions=positions)
    # Run Dijkstra's algorithm
    distances, previous_nodes = dijkstra.run_dijkstra(adj_list, start_node)
    shortest_path = dijkstra.reconstruct_path(previous_nodes, end_node)

    draw_graph(graph, shortest_path=shortest_path, positions=positions)

    dijkstra.dijkstra_with_animation(graph, start_node, end_node, positions)
