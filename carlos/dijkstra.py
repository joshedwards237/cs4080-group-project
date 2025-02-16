import networkx as nx
import matplotlib.pyplot as plt
import heapq


def dijkstra_with_animation(graph, source, target, positions=None):
    # Priority queue (min-heap)
    queue = [(0, source)]
    # Distance dictionary
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    # Path tracking
    previous_nodes = {}

    # Visualization setup
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos=positions, with_labels=True, node_color='lightgray', edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(graph, pos=positions, edge_labels=labels, font_size=12, ax=ax)

    while queue:
        # Get the node with the smallest distance
        current_dist, current_node = heapq.heappop(queue)

        # Visualization: Highlight the current node
        nx.draw_networkx_nodes(graph, pos=positions, nodelist=[current_node], node_color='blue', ax=ax)
        plt.pause(0.5)

        # Stop if we reach the target
        if current_node == target:
            break

        for neighbor, data in graph[current_node].items():
            new_dist = current_dist + data['weight']
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (new_dist, neighbor))

                # Visualization: Highlight edges being traversed
                nx.draw_networkx_edges(graph, pos=positions, edgelist=[(current_node, neighbor)], edge_color='blue', ax=ax)
                plt.pause(0.05)

    # Reconstruct shortest path
    path = []
    node = target
    while node in previous_nodes:
        path.append((previous_nodes[node], node))
        node = previous_nodes[node]

    # Highlight the shortest path
    nx.draw_networkx_edges(graph, pos=positions, edgelist=path, edge_color='green', width=2, ax=ax)
    plt.show()

    return distances[target], path[::-1]  # Return shortest distance and path


def run_dijkstra(adj_list, start):
    """
    Implements Dijkstra's algorithm to find the shortest path from a given start node.

    Parameters:
        adj_list (dict): Adjacency list representation of the graph with weights.
        start (str): The starting node.

    Returns:
        tuple: (distances, previous_nodes) where distances is a dict of the shortest distances from start node,
               and previous_nodes is a dict to reconstruct the shortest path.
    """
    # Priority queue for the min-heap
    pq = [(0, start)]  # (distance, node)
    distances = {node: float('inf') for node in adj_list}
    distances[start] = 0
    previous_nodes = {node: None for node in adj_list}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Skip processing if we have already found a shorter path
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in adj_list[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous_nodes


def reconstruct_path(previous_nodes, end):
    """Reconstructs the shortest path from end to start."""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    return path[::-1]  # Reverse the path
