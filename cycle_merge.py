import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import random
from typing import List, Tuple

class Graph:
    def __init__(self, n: int, coordinates: np.ndarray = None, distances: np.ndarray = None):
        self.n = n
        if coordinates is not None:
            self.coordinates = coordinates
            self.distances = squareform(pdist(coordinates, metric='euclidean'))
        elif distances is not None:
            self.distances = distances
            self.coordinates = np.random.rand(n, 2) * 10
        else:
            self.coordinates = np.random.rand(n, 2) * 10
            self.distances = squareform(pdist(self.coordinates, metric='euclidean'))
        self.priorities = np.ones(n)
    
    def set_priorities(self, priorities: np.ndarray):
        self.priorities = priorities / np.max(priorities)

def create_random_complete_metric_graph(n: int, seed: int = None) -> Graph:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    coordinates = np.random.rand(n, 2) * 20
    graph = Graph(n, coordinates=coordinates)
    priorities = np.random.uniform(0.1, 1.0, n)
    priorities[np.argmax(priorities)] = 1.0
    graph.set_priorities(priorities)
    return graph

def nearest_neighbor_tsp(graph: Graph, vertices: List[int]) -> Tuple[List[int], float]:
    if len(vertices) <= 1:
        return vertices, 0.0
    if len(vertices) == 2:
        return vertices + [vertices[0]], 2 * graph.distances[vertices[0], vertices[1]]
    tour = [vertices[0]]
    remaining = set(vertices[1:])
    current = vertices[0]
    total_length = 0
    while remaining:
        nearest = min(remaining, key=lambda v: graph.distances[current, v])
        total_length += graph.distances[current, nearest]
        tour.append(nearest)
        current = nearest
        remaining.remove(nearest)
    total_length += graph.distances[current, tour[0]]
    tour.append(tour[0])
    return tour, total_length

def compute_cycle_latency(graph: Graph, cycle_vertices: List[int], num_agents: int = 1) -> float:
    if len(cycle_vertices) <= 1:
        return 0.0
    _, tsp_length = nearest_neighbor_tsp(graph, cycle_vertices)
    priorities = [graph.priorities[v] for v in cycle_vertices]
    priorities.sort(reverse=True)
    denominator = sum(1.0 / priorities[i] for i in range(min(num_agents, len(priorities))))
    if denominator == 0:
        return 0.0
    return tsp_length / denominator

def kmeans_cycle_cover(graph: Graph, k: int) -> List[List[int]]:
    if k >= graph.n:
        return [[i] for i in range(graph.n)]
    coordinates = graph.coordinates
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coordinates)
    clusters = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return clusters

def are_cycles_adjacent(graph: Graph, cycle1: List[int], cycle2: List[int]) -> bool:
    if not cycle1 or not cycle2:
        return False
    min_dist = float('inf')
    for v1 in cycle1:
        for v2 in cycle2:
            min_dist = min(min_dist, graph.distances[v1, v2])
    all_distances = graph.distances[np.triu_indices(graph.n, k=1)]
    threshold = np.median(all_distances)
    return min_dist <= threshold

def merge_cycles(graph: Graph, cycle1: List[int], cycle2: List[int]) -> List[int]:
    return cycle1 + cycle2

def greedy_cycle_merge(graph: Graph, m: int) -> Tuple[List[List[int]], List[int]]:
    cycles = kmeans_cycle_cover(graph, m)
    cycles = [cycle for cycle in cycles if len(cycle) > 0]
    num_agents = [1] * len(cycles)
    latencies = [compute_cycle_latency(graph, cycle, 1) for cycle in cycles]
    iteration = 0
    max_iterations = 100
    while len(cycles) > 1 and iteration < max_iterations:
        iteration += 1
        max_latency_idx = np.argmax(latencies)
        max_latency = latencies[max_latency_idx]
        best_merge = None
        best_new_max_latency = max_latency
        for j in range(len(cycles)):
            if j == max_latency_idx:
                continue
            if not are_cycles_adjacent(graph, cycles[max_latency_idx], cycles[j]):
                continue
            merged_cycle = merge_cycles(graph, cycles[max_latency_idx], cycles[j])
            merged_agents = num_agents[max_latency_idx] + num_agents[j]
            merged_latency = compute_cycle_latency(graph, merged_cycle, merged_agents)
            temp_latencies = latencies[:]
            temp_latencies[max_latency_idx] = merged_latency
            temp_latencies.pop(j if j > max_latency_idx else j)
            new_max_latency = max(temp_latencies)
            if new_max_latency < best_new_max_latency:
                best_new_max_latency = new_max_latency
                best_merge = j
        if best_merge is not None:
            j = best_merge
            merged_cycle = merge_cycles(graph, cycles[max_latency_idx], cycles[j])
            merged_agents = num_agents[max_latency_idx] + num_agents[j]
            if j > max_latency_idx:
                cycles[max_latency_idx] = merged_cycle
                cycles.pop(j)
                num_agents[max_latency_idx] = merged_agents
                num_agents.pop(j)
                latencies[max_latency_idx] = compute_cycle_latency(graph, merged_cycle, merged_agents)
                latencies.pop(j)
            else:
                cycles[j] = merged_cycle
                cycles.pop(max_latency_idx)
                num_agents[j] = merged_agents
                num_agents.pop(max_latency_idx)
                latencies[j] = compute_cycle_latency(graph, merged_cycle, merged_agents)
                latencies.pop(max_latency_idx)
        else:
            break
    return cycles, num_agents

def visualize_solution(graph: Graph, cycles: List[List[int]], num_agents: List[int], 
                      title: str = "Multi-Agent Patrolling Solution"):
    plt.figure(figsize=(12, 8))
    x_coords = graph.coordinates[:, 0]
    y_coords = graph.coordinates[:, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cycles)))
    for i, (cycle, agents, color) in enumerate(zip(cycles, num_agents, colors)):
        if len(cycle) == 0:
            continue
        cycle_x = [graph.coordinates[v, 0] for v in cycle]
        cycle_y = [graph.coordinates[v, 1] for v in cycle]
        plt.scatter(cycle_x, cycle_y, c=[color], s=100, alpha=0.7, 
                   label=f'Cycle {i+1} ({agents} agents)')
        if len(cycle) > 1:
            tour, _ = nearest_neighbor_tsp(graph, cycle)
            tour_x = [graph.coordinates[v, 0] for v in tour]
            tour_y = [graph.coordinates[v, 1] for v in tour]
            plt.plot(tour_x, tour_y, c=color, alpha=0.6, linewidth=2)
    for i in range(graph.n):
        plt.annotate(f'{i}\n({graph.priorities[i]:.2f})', 
                    (x_coords[i], y_coords[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def test_algorithm():
    n = 10
    m = 6
    print(f"Testing Greedy Cycle Merge with {n} vertices and {m} agents")
    graph = create_random_complete_metric_graph(n, seed=42)
    print(f"Graph priorities: {graph.priorities}")
    initial_cycles = kmeans_cycle_cover(graph, m)
    initial_cycles = [c for c in initial_cycles if len(c) > 0]
    initial_agents = [1] * len(initial_cycles)
    print("\nInitial k-cycle cover:")
    max_latency = 0
    for i, (cycle, agents) in enumerate(zip(initial_cycles, initial_agents)):
        latency = compute_cycle_latency(graph, cycle, agents)
        max_latency = max(max_latency, latency)
        print(f"Cycle {i+1}: nodes {cycle}, agents={agents}, latency={latency:.3f}")
    print(f"Initial max latency: {max_latency:.3f}")
    visualize_solution(graph, initial_cycles, initial_agents, "Initial KMeans Cycle Cover")
    merged_cycles, agent_alloc = greedy_cycle_merge(graph, m)
    print("\nMerged solution:")
    max_latency = 0
    for i, (cycle, agents) in enumerate(zip(merged_cycles, agent_alloc)):
        latency = compute_cycle_latency(graph, cycle, agents)
        max_latency = max(max_latency, latency)
        print(f"Cycle {i+1}: nodes {cycle}, agents={agents}, latency={latency:.3f}")
    print(f"Merged max latency: {max_latency:.3f}")
    visualize_solution(graph, merged_cycles, agent_alloc, "Greedy Cycle Merge Solution")

if __name__ == "__main__":
    test_algorithm()
