import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

# ---- Brute-force TSP ----
def brute_force_tsp(G_sub):
    nodes = list(G_sub.nodes)
    min_cost = float('inf')
    best_path = None
    for perm in itertools.permutations(nodes):
        try:
            cost = 0
            for i in range(len(perm)):
                u = perm[i]
                v = perm[(i + 1) % len(perm)]
                cost += G_sub[u][v]['weight']
            if cost < min_cost:
                min_cost = cost
                best_path = perm
        except KeyError:
            continue
    if len(nodes) == 1:
        return 0, (nodes[0],)
    return min_cost, best_path

# ---- Agent Assignments ----
def generate_agent_assignments(k, m):
    for alloc in itertools.product(range(1, m + 1), repeat=k):
        if sum(alloc) == m:
            yield list(alloc)

# ---- Partition Generator ----
def k_partitions(collection, k):
    def helper(partitions, rest):
        if not rest and len(partitions) == k:
            yield [set(p) for p in partitions]
        elif rest:
            for i in range(len(partitions)):
                new_partitions = [list(p) for p in partitions]
                new_partitions[i].append(rest[0])
                yield from helper(new_partitions, rest[1:])
            if len(partitions) < k:
                yield from helper(partitions + [[rest[0]]], rest[1:])
    return helper([], list(collection))

# ---- Cost Function ----
def compute_cost_corrected(G, partitions, agent_counts, priorities):
    costs = []
    for part, agents in zip(partitions, agent_counts):
        G_sub = G.subgraph(part).copy()
        tsp_len, _ = brute_force_tsp(G_sub)
        top_priorities = sorted([priorities[v] for v in part], reverse=True)[:agents]
        if any(p == 0 for p in top_priorities):
            cost = float('inf')
        else:
            inv_sum = sum(1 / p for p in top_priorities)
            cost = tsp_len / inv_sum if inv_sum != 0 else float('inf')
        costs.append(cost)
    return max(costs)

def f_cost(G, subset, agent_count, priorities):
    if not subset:
        return 0
    G_sub = G.subgraph(subset).copy()
    tsp_len, _ = brute_force_tsp(G_sub)
    top_priorities = sorted([priorities[v] for v in subset], reverse=True)[:agent_count]
    if any(p == 0 for p in top_priorities):
        return float('inf')
    inv_sum = sum(1 / p for p in top_priorities)
    return tsp_len / inv_sum if inv_sum != 0 else float('inf')

# ---- Visualizer with TSP Paths ----
def visualize_partition_with_tours(G, partitions, priorities, cost, pos, agent_counts):
    color_map = ['green', 'blue', 'red', 'orange', 'purple', 'brown']
    label_dict = {v: f"{v}\nϕ={priorities[v]:.2f}" for v in G.nodes()}

    # Determine bottleneck partition
    bottleneck_idx = -1
    max_latency = -1

    for idx, (part, agents) in enumerate(zip(partitions, agent_counts)):
        subG = G.subgraph(part).copy()
        tsp_len, _ = brute_force_tsp(subG)
        top_priorities = sorted([priorities[v] for v in part], reverse=True)[:agents]
        inv_sum = sum(1 / p for p in top_priorities) if all(p > 0 for p in top_priorities) else 0
        latency = tsp_len / inv_sum if inv_sum != 0 else float('inf')
        if latency > max_latency:
            max_latency = latency
            bottleneck_idx = idx

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.0)

    # Draw MST ----
    mst = nx.minimum_spanning_tree(G)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=mst.edges(),
        edge_color="black",
        width=3.5,
        style="solid",
        alpha=0.8
    )

    for idx, part in enumerate(partitions):
        subG = G.subgraph(part).copy()
        _, tsp_path = brute_force_tsp(subG)
        if tsp_path:
            tour_edges = [(tsp_path[i], tsp_path[(i + 1) % len(tsp_path)]) for i in range(len(tsp_path))]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=tour_edges,
                edge_color=color_map[idx % len(color_map)],
                width=3.0 if idx == bottleneck_idx else 2.0,
                style='-' if idx != bottleneck_idx else 'solid',
                alpha=0.9,
                arrows=True,
                arrowstyle='-|>',
                connectionstyle="arc3,rad=0.15"
            )

        # Highlight top priority nodes
        agents = agent_counts[idx]
        top_nodes = sorted(part, key=lambda v: priorities[v], reverse=True)[:agents]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=top_nodes,
            node_color='gold',
            edgecolors='black',
            node_size=900,
            linewidths=2.0,
            label="Top priority nodes"
        )

    # Draw all nodes normally
    nx.draw_networkx_nodes(G, pos, node_color="lightyellow", node_size=700, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=10, font_weight='bold')

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(
        f"Optimal Partition with TSP Tours + MST Overlay\n"
        f"Max Latency = {cost:.2f} (Bottleneck: Partition {bottleneck_idx + 1})"
    )
    plt.axis("off")
    plt.show()

# ---- Solver ----
def solve_patrolling_problem_final(G, priorities, m_agents, max_k=None):
    nodes = list(G.nodes)
    n = len(nodes)
    if max_k is None:
        max_k = m_agents

    best_solution = None
    best_cost = float('inf')

    for k in range(1, min(n, m_agents) + 1):
        for partitions in k_partitions(nodes, k):
            for agent_counts in generate_agent_assignments(k, m_agents):
                cost = compute_cost_corrected(G, partitions, agent_counts, priorities)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (partitions, agent_counts)

    return best_solution, best_cost

# ---- Example Usage ----
if __name__ == "__main__":
    # Create a complete graph with fixed coordinates
    G = nx.complete_graph(8)
    pos = {v: (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)) for v in G.nodes()}

    # Set edge weights to Euclidean distance
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dist = math.hypot(x1 - x2, y1 - y2)
        G.edges[u, v]['weight'] = dist

    # Assign random priorities in (0.1, 1.0]
    priorities = {v: round(random.uniform(0.1, 1.0), 2) for v in G.nodes()}

    # Solve with 3 agents
    (partitions, agent_counts), cost = solve_patrolling_problem_final(G, priorities, m_agents=3)

    # Print overall solution summary
    print("Best Partitions:", partitions)
    print("Agent Assignment per Partition:", agent_counts)
    print("Objective Cost (Max Latency):", round(cost, 2))

    # ---- Detailed info per partition ----
    print("\nDetailed Info for Each Partition:")
    for idx, (part, agent_count) in enumerate(zip(partitions, agent_counts)):
        subG = G.subgraph(part).copy()
        tsp_len, tsp_path = brute_force_tsp(subG)
        top_priorities = sorted([priorities[v] for v in part], reverse=True)[:agent_count]
        inv_sum = sum(1 / p for p in top_priorities)
        corrected_latency = tsp_len / inv_sum if inv_sum != 0 else float('inf')
        print(f"Partition {idx + 1}:")
        print(f"  Nodes: {sorted(part)}")
        print(f"  TSP Path: {tsp_path}")
        print(f"  TSP Length: {tsp_len:.2f}")
        print(f"  Latency (TSP / Harmonic Sum): {corrected_latency:.2f}")
        print(f"  Agents Assigned: {agent_count}\n")

    # Visualize result with true Cartesian layout
    visualize_partition_with_tours(G, partitions, priorities, cost, pos, agent_counts)
