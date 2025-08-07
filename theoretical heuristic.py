import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

# ---- Brute-force TSP ----
def brute_force_tsp(G_sub):
    nodes = list(G_sub.nodes)
    if len(nodes) == 1:
        return 0, tuple(nodes)
    min_cost = float('inf')
    best_path = None
    for perm in itertools.permutations(nodes):
        cost = 0
        valid = True
        for i in range(len(perm)):
            u = perm[i]
            v = perm[(i + 1) % len(perm)]
            if G_sub.has_edge(u, v):
                cost += G_sub[u][v]['weight']
            else:
                valid = False
                break
        if valid and cost < min_cost:
            min_cost = cost
            best_path = perm
    return min_cost, best_path

# ---- Agent Assignments ----
def generate_agent_assignments(k, m):
    for alloc in itertools.product(range(1, m + 1), repeat=k):
        if sum(alloc) == m:
            yield list(alloc)

# ---- Partition Generator ----
def k_partitions(collection, k):
    def helper(partitions, rest):
        if not rest:
            if len(partitions) == k:
                yield [set(p) for p in partitions]
            return
        if len(partitions) < k:
            # start a new partition with rest[0]
            yield from helper(partitions + [[rest[0]]], rest[1:])
        # add rest[0] to existing partitions
        for i in range(len(partitions)):
            new_partitions = [list(p) for p in partitions]
            new_partitions[i].append(rest[0])
            yield from helper(new_partitions, rest[1:])
    return helper([], list(collection))

# ---- Cost Function ----
def compute_cost_corrected(G, partitions, agent_counts, priorities):
    costs = []
    for part, agents in zip(partitions, agent_counts):
        if not part:
            costs.append(0)
            continue
        G_sub = G.subgraph(part).copy()
        tsp_len, _ = brute_force_tsp(G_sub)
        top_priorities = sorted((priorities[v] for v in part), reverse=True)[:agents]
        if any(p == 0 for p in top_priorities):
            costs.append(float('inf'))
            continue
        inv_sum = sum(1 / p for p in top_priorities)
        cost = tsp_len / inv_sum if inv_sum > 0 else float('inf')
        costs.append(cost)
    return max(costs)

# ---- Brute-Force Solver ----
def solve_patrolling_problem_final(G, priorities, m_agents):
    best_solution = None
    best_cost = float('inf')
    nodes = list(G.nodes())
    for k in range(1, min(len(nodes), m_agents) + 1):
        for parts in k_partitions(nodes, k):
            for alloc in generate_agent_assignments(k, m_agents):
                cost = compute_cost_corrected(G, parts, alloc, priorities)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (parts, alloc)
    return best_solution, best_cost

# ---- Heuristic Algorithms ----
def binary_search_L(G, phi, k, L):
    V = sorted(G.nodes, key=lambda v: -phi[v])
    U = set(V)
    clusters = []
    for vi in V:
        if vi not in U:
            continue
        C = {vi}
        for v in list(U):
            if v != vi and G[vi][v]['weight'] <= L / (2 * phi[vi]):
                C.add(v)
        clusters.append(C)
        U -= C
        if len(clusters) > k:
            return False
    return clusters

def tsp_construction(G, cluster, phi, L, epsilon, n_max):
    phi_max = max(phi[v] for v in cluster)
    partitions = [set(cluster)]
    queue = [set(cluster)]
    used_agents = 1
    while queue:
        S = queue.pop(0)
        split_occurred = False
        for m in range(2, int(1/epsilon) + 1):
            for subset in itertools.combinations(S, m+1):
                if all(G[u][v]['weight'] > L / (m * phi_max) for u, v in itertools.combinations(subset, 2)):
                    if used_agents >= n_max:
                        return False
                    v_star = max(subset, key=lambda v: min(G[v][u]['weight'] for u in subset if u != v))
                    S.remove(v_star)
                    new_cluster = {v_star}
                    partitions.append(new_cluster)
                    queue.append(new_cluster)
                    used_agents += 1
                    split_occurred = True
                    break
            if split_occurred:
                break
    return partitions

def integrated_patrol(G, phi, k, epsilon=0.1, delta_tol=1e-2):
    def mst_length(G):
        T = nx.minimum_spanning_tree(G)
        return sum(d['weight'] for _, _, d in T.edges(data=True))
    L_low, L_high = 0, 2 * mst_length(G)
    best_solution, best_L = None, float('inf')
    while L_high - L_low > delta_tol:
        L_mid = (L_low + L_high) / 2
        clusters = binary_search_L(G, phi, k, L_mid)
        if clusters is False:
            L_low = L_mid
            continue
        total_parts = []
        agents_used = 0
        feasible = True
        for C in clusters:
            remain = k - agents_used
            refined = tsp_construction(G, C, phi, L_mid, epsilon, remain)
            if refined is False:
                feasible = False
                break
            total_parts.extend(refined)
            agents_used += len(refined)
        if feasible and agents_used <= k:
            best_solution = (total_parts, [1]*len(total_parts))
            best_L = L_mid
            L_high = L_mid
        else:
            L_low = L_mid
    return best_solution, best_L

# ---- Visualization ----
def visualize_named_solution(G, partitions, priorities, cost, pos, agent_counts, title="Patrolling Solution"):
    colors = ['green','blue','red','orange','purple','brown']
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')
    for idx, (part, agents) in enumerate(zip(partitions, agent_counts)):
        subG = G.subgraph(part)
        tsp_len, tsp_path = brute_force_tsp(subG)
        if tsp_path:
            edges = [(tsp_path[i], tsp_path[(i+1)%len(tsp_path)]) for i in range(len(tsp_path))]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors[idx%len(colors)], width=2)
        top = sorted(part, key=lambda v: priorities[v], reverse=True)[:agents]
        nx.draw_networkx_nodes(G, pos, nodelist=top, node_color='gold', node_size=300)
    nx.draw_networkx_nodes(G, pos, node_color='lightyellow', node_size=200)
    labels = {v: f"{v}\nϕ={priorities[v]:.2f}" for v in G.nodes}
    nx.draw_networkx_labels(G, pos, labels)
    plt.title(f"{title}\nMax Latency = {cost:.2f}")
    plt.axis('off')
    plt.show()

# ---- Example Usage ----
if __name__ == "__main__":
    # Create random complete graph
    G = nx.complete_graph(20)
    pos = {i: (random.uniform(0,10), random.uniform(0,10)) for i in G.nodes()}
    for u,v in G.edges():
        G.edges[u,v]['weight'] = math.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1])
    priorities = {i: round(random.uniform(0.1,1.0), 2) for i in G.nodes()}

    # Solve heuristic
    (h_parts, h_agents), h_cost = integrated_patrol(G, priorities, k=3)
    print("Heuristic cost:", h_cost)
    # visualize_named_solution(G, h_parts, priorities, h_cost, pos, h_agents, title="Heuristic Solution")

    # # Solve brute-force
    # (bf_parts, bf_agents), bf_cost = solve_patrolling_problem_final(G, priorities, m_agents=3)
    # print("Brute-force cost:", bf_cost)
    # visualize_named_solution(G, bf_parts, priorities, bf_cost, pos, bf_agents, title="Brute-force Optimal")



