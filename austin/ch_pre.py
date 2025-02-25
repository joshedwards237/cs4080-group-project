import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt


def aux_path_search(G, source, target, removed):
    heap = [(0, source)]
    distances = {source: 0}
    while heap:
        cost, node = heapq.heappop(heap)
        if node == target:
            return cost
        for neighbor in G.neighbors(node):
            if neighbor == removed:
                continue
            edge_weight = G[node][neighbor].get('weight')
            new_cost = cost + edge_weight
            if new_cost < distances.get(neighbor, math.inf):
                distances[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor))

    return math.inf


def compute_edge_diff(G, v):
    neighbors = list(G.neighbors(v))
    shortcuts_added = 0
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            w = neighbors[j]
            cost_uv = G[u][v].get('weight')
            cost_vw = G[v][w].get('weight')
            via_cost = cost_uv + cost_vw
            aux_path_cost = aux_path_search(G, u, w, v)
            if aux_path_cost > via_cost:
                shortcuts_added += 1
    edges_removed = len(neighbors)
    return shortcuts_added - edges_removed, shortcuts_added


def contract_node(G_work, v, current_rank, shortcuts, rank_dict, aux_g):
    neighbors = list(G_work.neighbors(v))
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            w = neighbors[j]

            if not (G_work.has_edge(u, v) and G_work.has_edge(v, w)):
                continue

            cost_uv = G_work[u][v].get('weight')
            cost_vw = G_work[v][w].get('weight')
            via_cost = cost_uv + cost_vw
            aux_path_cost = aux_path_search(G_work, u, w, v)
            if aux_path_cost > via_cost:
                if shortcuts.get((u, w)) is not None:
                    if shortcuts[(u, w)] > via_cost:
                        aux_g[u][w].update({'weight': via_cost})
                        G_work[u][w].update({'weight': via_cost})
                        shortcuts[(u, w)] = via_cost
                else:
                    shortcuts[(u, w)] = via_cost
                    aux_g.add_edge(u, w, weight=via_cost, shortcut=True)
                    G_work.add_edge(u, w, weight=via_cost, shortcut=True)

    rank_dict[v] = current_rank
    G_work.remove_node(v)


def build_contraction_hierarchy_offline(G):
    G_work = G.copy()
    G_aux = G.copy()
    shortcuts = {}
    rank_dict = {}

    order = sorted(list(G.nodes()), key=lambda v: compute_edge_diff(G, v)[0])
    current_rank = 0
    for v in order:
        contract_node(G_work, v, current_rank, shortcuts, rank_dict, G_aux)
        current_rank += 1

    return G_aux, rank_dict, order, shortcuts


def build_contraction_hierarchy_online(G):
    G_work = G.copy()
    G_aux = G.copy()
    shortcuts = {}
    rank_dict = {}
    current_rank = 0
    process_order = []

    edge_diffs = {v: compute_edge_diff(G_work, v)[0] for v in list(G_work.nodes())}

    while G_work.nodes():

        # Select the node with the smallest edge difference.
        v_min = min(edge_diffs, key=edge_diffs.get)
        # save the neighbors of v_min to update edge difference dict
        v_neighbors = list(G_work.neighbors(v_min))
        # remove v_min from the rank list
        edge_diffs.pop(v_min)

        process_order.append(v_min)
        contract_node(G_work, v_min, current_rank, shortcuts, rank_dict, G_aux)
        current_rank += 1

        # update edge difference dict
        for node in v_neighbors:
            if node in edge_diffs:
                edge_diffs[node] = compute_edge_diff(G_work, node)[0]

    return G_aux, rank_dict, process_order, shortcuts


def draw_graph(graph, positions=None, online=True):
    plt.figure(figsize=(12, 10))

    if online:
        plt.title("Online")
    else:
        plt.title("Offline")

    nx.draw_networkx_nodes(graph, positions, node_color="lightblue", node_size=1500)
    nx.draw_networkx_labels(graph, positions, font_size=12)

    labels = nx.get_edge_attributes(graph, 'weight')

    shortcut_edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get('shortcut', False)]
    regular_edges = [(u, v) for u, v, data in graph.edges(data=True) if not data.get('shortcut', False)]

    nx.draw_networkx_edges(graph, positions, edgelist=regular_edges, edge_color="gray")

    nx.draw_networkx_edges(graph, positions, edgelist=shortcut_edges, edge_color="green",
                           connectionstyle='arc3, rad=0.33', arrows=True)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=labels, font_size=12)

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('A', 'B', weight=4)
    G.add_edge('B', 'C', weight=2)
    G.add_edge('C', 'D', weight=1)
    G.add_edge('D', 'E', weight=3)
    G.add_edge('B', 'G', weight=1)
    G.add_edge('D', 'I', weight=1)
    G.add_edge('E', 'J', weight=3)
    G.add_edge('F', 'G', weight=1)
    G.add_edge('G', 'H', weight=2)
    G.add_edge('I', 'J', weight=1)
    G.add_edge('G', 'L', weight=1)
    G.add_edge('I', 'N', weight=3)
    G.add_edge('J', 'O', weight=3)
    G.add_edge('K', 'L', weight=1)
    G.add_edge('L', 'M', weight=3)
    G.add_edge('M', 'N', weight=3)
    G.add_edge('N', 'O', weight=3)
    G.add_edge('K', 'P', weight=1)
    G.add_edge('P', 'Q', weight=1)
    G.add_edge('Q', 'R', weight=3)
    G.add_edge('R', 'S', weight=3)
    G.add_edge('S', 'T', weight=3)
    G.add_edge('Q', 'V', weight=1)
    G.add_edge('T', 'Y', weight=3)
    G.add_edge('U', 'V', weight=3)
    G.add_edge('V', 'W', weight=2)
    G.add_edge('W', 'X', weight=2)
    G.add_edge('X', 'Y', weight=2)
    positions = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXY"

    for idx, letter in enumerate(letters):
        col = idx // 5
        row = idx % 5
        positions[letter] = (col, 4 - row)

    # --- Offline Contraction Hierarchy ---
    aux_g_offline, rank_offline, process_order, shortcuts = build_contraction_hierarchy_offline(G)
    print("Offline node order: ", process_order)
    draw_graph(aux_g_offline, positions=positions, online=False)
    print("Offline shortcuts: ", shortcuts)
    print("Total Offline Shortcuts: ", len(shortcuts))

    # --- Online Contraction Hierarchy ---
    aux_g_online, rank_online, process_order, shortcuts = build_contraction_hierarchy_online(G)
    print("Online CH node order: ", process_order)
    draw_graph(aux_g_online, positions=positions, online=True)
    print("Online shortcuts: ", shortcuts)
    print("Total Online Shortcuts: ", len(shortcuts))
