
"""
pbmlp_nx.py

NetworkX-based implementation of the theoretical PBMLP algorithms:
  - ClusterAssignment (Alg. 1)
  - BinarySearchL (Alg. 2)

Distances are taken to be shortest-path distances on the given (weighted) graph.
TSP approximations operate on the metric closure induced by those distances.

Author: SJTU Codes
License: MIT
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass
import math
import random
import networkx as nx


# ------------------------------ Distances (metric closure) ------------------------------

def all_pairs_metric_distances(G: nx.Graph, weight: str = "weight") -> Dict[int, Dict[int, float]]:
    """Compute shortest-path distances between all node pairs (metric closure)."""
    # Uses Dijkstra; assumes nonnegative weights
    return {u: dict(d) for u, d in nx.all_pairs_dijkstra_path_length(G, weight=weight)}


# ------------------------------ MST weight on metric closure ------------------------------

def mst_weight_metric_closure(dist: Dict[int, Dict[int, float]], nodes: Optional[Sequence[int]] = None) -> float:
    """Prim-style MST weight over a complete graph whose edge weights are given by 'dist'."""
    if nodes is None:
        nodes = list(dist.keys())
    n = len(nodes)
    if n <= 1:
        return 0.0

    idx_of = {v: i for i, v in enumerate(nodes)}
    in_tree = [False] * n
    key = [math.inf] * n

    key[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = -1
        best = math.inf
        for i in range(n):
            if not in_tree[i] and key[i] < best:
                best = key[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        total += 0.0 if best is math.inf else best

        u_node = nodes[u]
        for v in range(n):
            if not in_tree[v]:
                v_node = nodes[v]
                w = dist[u_node][v_node]
                if w < key[v]:
                    key[v] = w
    return total


# ------------------------------ TSP approximations on metric closure ------------------------------

def tsp_length_mst_preorder_dist(dist: Dict[int, Dict[int, float]], nodes: Sequence[int]) -> float:
    """2-approx TSP using MST preorder on the metric closure defined by 'dist'."""
    n = len(nodes)
    if n <= 1:
        return 0.0

    # Prim to get parent array
    idx_of = {v: i for i, v in enumerate(nodes)}
    parent = [-1] * n
    key = [math.inf] * n
    in_tree = [False] * n
    key[0] = 0.0

    for _ in range(n):
        u = -1
        best = math.inf
        for i in range(n):
            if not in_tree[i] and key[i] < best:
                best = key[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        u_node = nodes[u]
        for v in range(n):
            if not in_tree[v]:
                v_node = nodes[v]
                w = dist[u_node][v_node]
                if w < key[v]:
                    key[v] = w
                    parent[v] = u

    # Build MST adjacency
    adj = [[] for _ in range(n)]
    for v in range(1, n):
        p = parent[v]
        if p >= 0:
            adj[p].append(v)
            adj[v].append(p)

    # Preorder DFS
    order: List[int] = []
    def dfs(i: int, p: int) -> None:
        order.append(i)
        for j in adj[i]:
            if j != p:
                dfs(j, i)
    dfs(0, -1)

    # Sum cycle length
    total = 0.0
    for i in range(n - 1):
        a = nodes[order[i]]
        b = nodes[order[i + 1]]
        total += dist[a][b]
    total += dist[nodes[order[-1]]][nodes[order[0]]]
    return total


def tsp_length_nn_dist(dist: Dict[int, Dict[int, float]], nodes: Sequence[int]) -> float:
    """Nearest-neighbor heuristic tour length on metric closure."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    remaining = set(nodes)
    current = min(remaining)  # deterministic start
    remaining.remove(current)
    tour = [current]
    while remaining:
        nxt = min(remaining, key=lambda v: dist[current][v])
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt
    total = 0.0
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


# ------------------------------ Algorithm 1: ClusterAssignment ------------------------------

def cluster_assignment_nx(
    G: nx.Graph,
    priorities: Dict[int, float],
    m: int,
    L: float,
    dist: Optional[Dict[int, Dict[int, float]]] = None,
    weight: str = "weight",
) -> Optional[List[Set[int]]]:
    """
    Build at most m clusters using radius r_L(x)=L/phi(x) over metric shortest-path distances.
    Return None if > m clusters are required (infeasible for this L).
    """
    if dist is None:
        dist = all_pairs_metric_distances(G, weight=weight)

    V = list(G.nodes())
    if not V:
        return []

    # Sort by nonincreasing priority (tie-break by id for determinism)
    V_sorted = sorted(V, key=lambda v: (-priorities[v], v))
    U: Set[int] = set(V)
    clusters: List[Set[int]] = []

    for vi in V_sorted:
        if vi not in U:
            continue
        phi = priorities[vi]
        if phi <= 0:
            raise ValueError(f"Priority must be > 0 for node {vi}. Got {phi}.")
        radius = L / phi
        Ci: Set[int] = set()
        for v in list(U):
            if dist[vi][v] <= radius:
                Ci.add(v)
        clusters.append(Ci)
        U.difference_update(Ci)
        if len(clusters) > m:
            return None
    return clusters


# ------------------------------ Algorithm 2: BinarySearchL ------------------------------

def binary_search_L_nx(
    G: nx.Graph,
    priorities: Dict[int, float],
    m: int,
    eps: float,
    dist: Optional[Dict[int, Dict[int, float]]] = None,
    weight: str = "weight",
    L_min: float = 0.0,
    L_max: Optional[float] = None,
) -> Tuple[float, List[Set[int]], Dict[int, Dict[int, float]]]:
    """
    Binary search the minimum feasible latency L* within tolerance eps.
    Returns (L_hat, clusters, dist) where dist is the metric-closure distance dict used.
    """
    if dist is None:
        dist = all_pairs_metric_distances(G, weight=weight)

    # Upper bound: 2 * MST(V) on the metric closure
    if L_max is None:
        mst_total = mst_weight_metric_closure(dist, list(G.nodes()))
        L_max = 2.0 * mst_total

    last_clusters: Optional[List[Set[int]]] = None
    while (L_max - L_min) > eps:
        L_mid = 0.5 * (L_min + L_max)
        clusters = cluster_assignment_nx(G, priorities, m, L_mid, dist=dist, weight=weight)
        if clusters is not None:
            L_max = L_mid
            last_clusters = clusters
        else:
            L_min = L_mid

    if last_clusters is None:
        # Try at L_max
        clusters = cluster_assignment_nx(G, priorities, m, L_max, dist=dist, weight=weight)
        if clusters is None:
            raise RuntimeError("No feasible clustering found at chosen L_max.")
        last_clusters = clusters

    return L_max, last_clusters, dist


# ------------------------------ Allocation & Latency ------------------------------

def allocate_robots_proportional(
    clusters: List[Set[int]],
    priorities: Dict[int, float],
    m: int,
) -> List[int]:
    """At least 1 per nonempty cluster, then Hamilton apportionment by sum of priorities."""
    k = len(clusters)
    if k == 0:
        return []
    nonempty = [i for i, C in enumerate(clusters) if len(C) > 0]
    k_nonempty = len(nonempty)
    if k_nonempty > m:
        raise ValueError("More nonempty clusters than robots.")

    alloc = [0] * k
    for i in nonempty:
        alloc[i] = 1
    remaining = m - k_nonempty
    if remaining <= 0:
        return alloc

    weights = [sum(priorities[v] for v in C) for C in clusters]
    W = sum(weights[i] for i in nonempty)
    if W <= 0:
        # uniform
        base = remaining // k_nonempty
        rem = remaining - base * k_nonempty
        for i in nonempty:
            alloc[i] += base
        for i in nonempty[:rem]:
            alloc[i] += 1
        return alloc

    quotas = [(i, remaining * (weights[i] / W)) for i in nonempty]
    for i, q in quotas:
        alloc[i] += int(math.floor(q))
    given = sum(int(math.floor(q)) for _, q in quotas)
    remainders = sorted(((i, q - math.floor(q)) for i, q in quotas), key=lambda x: -x[1])
    for i in range(remaining - given):
        alloc[remainders[i][0]] += 1
    return alloc


def cluster_latency_dist(
    dist: Dict[int, Dict[int, float]],
    cluster: Sequence[int],
    priorities: Dict[int, float],
    agents: int,
    tsp_method: str = "mst",
) -> float:
    """Compute latency L_j for a cluster using metric-closure distances and a TSP heuristic."""
    n = len(cluster)
    if n == 0:
        return 0.0
    if agents <= 0:
        return float("inf")
    if agents >= n:
        return 0.0

    if tsp_method == "mst":
        tsp_len = tsp_length_mst_preorder_dist(dist, list(cluster))
    elif tsp_method == "nn":
        tsp_len = tsp_length_nn_dist(dist, list(cluster))
    else:
        raise ValueError("tsp_method must be 'mst' or 'nn'")

    phis = sorted((priorities[v] for v in cluster), reverse=True)
    denom = sum(1.0 / phis[i] for i in range(min(agents, len(phis))))
    return tsp_len / denom


def schedule_from_clusters_nx(
    G: nx.Graph,
    clusters: List[Set[int]],
    priorities: Dict[int, float],
    m: int,
    dist: Optional[Dict[int, Dict[int, float]]] = None,
    tsp_method: str = "mst",
    weight: str = "weight",
) -> Dict[str, object]:
    """
    Allocate robots, compute TSP lengths and latencies per cluster, return summary.
    """
    if dist is None:
        dist = all_pairs_metric_distances(G, weight=weight)

    alloc = allocate_robots_proportional(clusters, priorities, m)
    tsp_lengths: List[float] = []
    latencies: List[float] = []

    for C, a in zip(clusters, alloc):
        C_list = list(C)
        if len(C_list) == 0:
            tsp_lengths.append(0.0)
            latencies.append(0.0)
            continue
        tsp_len = tsp_length_mst_preorder_dist(dist, C_list) if tsp_method == "mst" else tsp_length_nn_dist(dist, C_list)
        tsp_lengths.append(tsp_len)
        latencies.append(cluster_latency_dist(dist, C_list, priorities, a, tsp_method=tsp_method))

    return dict(
        clusters=[set(C) for C in clusters],
        allocation=alloc,
        tsp_lengths=tsp_lengths,
        latencies=latencies,
        max_latency=max(latencies) if latencies else 0.0,
    )


# ------------------------------ Demo ------------------------------

def _demo_random_geometric(n: int = 50, m: int = 6, seed: int = 7, radius: float = 0.35) -> None:
    """Demo on a random geometric graph; edge weights are Euclidean; priorities in (0,1]."""
    random.seed(seed)
    G = nx.random_geometric_graph(n, radius, seed=seed)
    # ensure connectivity (increase radius if needed in practice). For demo, proceed as-is.
    pos = nx.get_node_attributes(G, "pos")
    # assign Euclidean weights
    for u, v in G.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)
    # priorities
    priorities = {v: random.random() * 0.9 + 0.1 for v in G.nodes()}  # avoid zeros

    # metric closure distances
    dist = all_pairs_metric_distances(G, weight="weight")

    # Binary search for L*
    eps = 1e-3
    L_hat, clusters, dist = binary_search_L_nx(G, priorities, m, eps, dist=dist, weight="weight")
    sched = schedule_from_clusters_nx(G, clusters, priorities, m, dist=dist, tsp_method="mst", weight="weight")

    print("=== PBMLP (NetworkX) Demo ===")
    print(f"n={n}, m={m}, eps={eps}, radius={radius}")
    print(f"L_hat = {L_hat:.4f}, #clusters = {len(clusters)}")
    print("Allocation:", sched["allocation"])
    print("Per-cluster latency:")
    for j, (C, L_j) in enumerate(zip(sched["clusters"], sched["latencies"])):
        print(f"  C{j}: |V|={len(C):>3}  L_j={L_j:>10.4f}")
    print(f"Max latency: {sched['max_latency']:.4f}")


if __name__ == "__main__":
    _demo_random_geometric()
