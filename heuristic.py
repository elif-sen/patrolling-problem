import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List, Dict, Tuple

def generate_graph(n: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.complete_graph(n)
    pos = {i: np.random.rand(2) * 100 for i in G.nodes}
    priorities = {i: random.randint(1, 100) for i in G.nodes}
    for u, v in G.edges:
        dist = int(np.linalg.norm(pos[u] - pos[v]) + 1)  # ensure positive integer distances
        G[u][v]['weight'] = dist
    return G, pos, priorities

def tsp_nearest_neighbor_fixed(G: nx.Graph, nodes: List[int]) -> Tuple[List[int], float]:
    if not nodes:
        return [], 0
    if len(nodes) == 1:
        return [nodes[0]], 0.0
    G_sub = G.subgraph(nodes).copy()
    unvisited = set(nodes)
    tour = [nodes[0]]
    unvisited.remove(nodes[0])
    total_length = 0
    while unvisited:
        last = tour[-1]
        next_node = min(unvisited, key=lambda x: G_sub[last][x]['weight'])
        total_length += G_sub[last][next_node]['weight']
        tour.append(next_node)
        unvisited.remove(next_node)
    total_length += G_sub[tour[-1]][tour[0]]['weight']  # return to start
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

def greedy_robot_assignment(tour_lengths: List[float], node_groups: List[List[int]],
                            priorities: Dict[int, float], m: int) -> Dict[int, List[float]]:
    k = len(node_groups)
    robot_assignments = {i: [] for i in range(k)}
    robot_pool = sorted([v for v in priorities.values()], reverse=True)
    for i in range(k):
        robot_assignments[i].append(robot_pool.pop(0))
    while robot_pool:
        latencies = [
            tour_lengths[i] / sum(1 / p for p in robot_assignments[i])
            for i in range(k)
        ]
        max_group = np.argmax(latencies)
        robot_assignments[max_group].append(robot_pool.pop(0))
    return robot_assignments

def run_priority_k_tsp(n: int, k: int, m: int, seed: int = 42):
    G, pos, priorities = generate_graph(n, seed)
    groups = partition_graph_mst(G, k)

    tsp_tours = []
    tsp_lengths = []
    for group in groups:
        tour, length = tsp_nearest_neighbor_fixed(G, group)
        tsp_tours.append(tour)
        tsp_lengths.append(length)

    assignments = greedy_robot_assignment(tsp_lengths, groups, priorities, m)

    latencies = []
    for i in range(k):
        denom = sum(1 / p for p in assignments[i])
        latency = tsp_lengths[i] / denom
        latencies.append(latency)

    return max(latencies), G, pos, priorities, groups, tsp_tours, assignments, tsp_lengths

def visualize_solution(G, pos, priorities, groups, tsp_tours, assignments, tsp_lengths, latency_value):
    k = len(groups)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', k)

    nx.draw_networkx_nodes(G, pos, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, labels={i: f"{i}\nϕ={priorities[i]}" for i in G.nodes()})
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3)

    for i, tour in enumerate(tsp_tours):
        edges = [(tour[j], tour[j + 1]) for j in range(len(tour) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[colors(i)] * len(edges), width=2)

    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(f"Best k-TSP Partition (Latency = {latency_value:.2f})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def find_best_k_latency(n: int, m: int, seed: int = 42):
    results = {}
    best_data = None
    for k in range(1, m + 1):
        latency, G, pos, priorities, groups, tsp_tours, assignments, tsp_lengths = run_priority_k_tsp(n, k, m, seed)
        results[k] = latency
        print(f"k = {k}: Max latency = {latency:.2f}")
        if best_data is None or latency < results[best_data[0]]:
            best_data = (k, latency, G, pos, priorities, groups, tsp_tours, assignments, tsp_lengths)

    best_k, best_latency, G, pos, priorities, groups, tsp_tours, assignments, tsp_lengths = best_data
    print(f"\nBest k = {best_k} with Min-Max Latency = {best_latency:.2f}")
    visualize_solution(G, pos, priorities, groups, tsp_tours, assignments, tsp_lengths, best_latency)

if __name__ == "__main__":
    find_best_k_latency(n=14, m=3, seed=1)
