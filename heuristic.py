import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List, Dict, Tuple

def generate_complete_priority_graph(n: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.complete_graph(n)
    pos = {i: np.random.rand(2) * 100 for i in G.nodes}
    priorities = {i: random.randint(1, 100) for i in G.nodes}
    for u, v in G.edges:
        dist = int(np.linalg.norm(pos[u] - pos[v]) + 1)
        G[u][v]['weight'] = dist
    return G, pos, priorities

def tsp_nearest_neighbor(G: nx.Graph, nodes: List[int]) -> Tuple[List[int], float]:
    if not nodes:
        return [], 0
    if len(nodes) == 1:
        return [nodes[0]], 0.0
    unvisited = set(nodes)
    tour = [nodes[0]]
    unvisited.remove(nodes[0])
    total_length = 0
    while unvisited:
        last = tour[-1]
        next_node = min(unvisited, key=lambda x: G[last][x]['weight'])
        total_length += G[last][next_node]['weight']
        tour.append(next_node)
        unvisited.remove(next_node)
    total_length += G[tour[-1]][tour[0]]['weight']
    tour.append(tour[0])
    return tour, total_length

def partition_graph_mst(G: nx.Graph, k: int) -> List[List[int]]:
    mst = nx.minimum_spanning_tree(G)
    edges_sorted = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    mst_copy = mst.copy()
    for i in range(k - 1):
        if i < len(edges_sorted):
            u, v, _ = edges_sorted[i]
            mst_copy.remove_edge(u, v)
    components = list(nx.connected_components(mst_copy))
    return [list(c) for c in components]

def corrected_robot_assignment(tour_lengths: List[float], node_groups: List[List[int]],
                               priorities: Dict[int, float], m: int) -> List[int]:
    k = len(node_groups)
    robots = [1] * k
    remaining = m - k

    def latency(i):
        phi_vals = sorted([priorities[v] for v in node_groups[i]], reverse=True)
        assigned = robots[i]
        effective = phi_vals[:assigned]
        return tour_lengths[i] / sum(1 / p for p in effective)

    for _ in range(remaining):
        latencies = [latency(i) for i in range(k)]
        max_idx = np.argmax(latencies)
        robots[max_idx] += 1

    return robots

def find_best_k_latency_fixed(G, pos, priorities, m: int):
    best_latency = float("inf")
    best_result = None
    for k in range(1, m + 1):
        groups = partition_graph_mst(G, k)
        tsp_tours = []
        tsp_lengths = []
        for group in groups:
            tour, length = tsp_nearest_neighbor(G, group)
            tsp_tours.append(tour)
            tsp_lengths.append(length)
        robot_counts = corrected_robot_assignment(tsp_lengths, groups, priorities, m)

        latencies = []
        for i in range(k):
            phi_vals = sorted([priorities[v] for v in groups[i]], reverse=True)
            effective = phi_vals[:robot_counts[i]]
            latency = tsp_lengths[i] / sum(1 / p for p in effective)
            latencies.append(latency)
        max_latency = max(latencies)

        if max_latency < best_latency:
            best_latency = max_latency
            best_result = (groups, robot_counts, tsp_tours)

    return best_result, best_latency

def visualize_heuristic_solution(G, pos, priorities, groups, tsp_tours, latency):
    plt.figure(figsize=(10, 8))
    k = len(groups)
    colors = plt.cm.get_cmap('tab10', k)

    nx.draw_networkx_nodes(G, pos, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, labels={i: f"{i}\nϕ={priorities[i]}" for i in G.nodes()})
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3)

    for i, tour in enumerate(tsp_tours):
        edges = [(tour[j], tour[j + 1]) for j in range(len(tour) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[colors(i)] * len(edges), width=2)

    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title(f"Heuristic Solution\nMax Latency = {latency:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

