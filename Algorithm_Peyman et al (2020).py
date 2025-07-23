import networkx as nx
import math
import itertools
import matplotlib.pyplot as plt
import random
import time

class KMinMaxTreeCover:
    def __init__(self, graph: nx.Graph, k: int):
        """
        Initialize the k-tree cover problem instance.

        Args:
            graph: A NetworkX graph.
            k: The maximum number of trees allowed in the cover.
        """
        self.graph = graph
        self.k = k

    '''
    @staticmethod
    def plot_graph(G, name, layout=None, highlight_color='red', base_color='lightgray'):
        """
        Plot the full graph G
        """
        if layout is None:
            layout = nx.spring_layout(G)

        plt.figure(figsize=(8, 8))

        # Draw base graph in light color
        nx.draw_networkx_nodes(G, layout, node_color=base_color, node_size=200)
        nx.draw_networkx_edges(G, layout, edge_color=base_color, width=0.5)
        nx.draw_networkx_labels(G, layout, font_size=8)

        plt.title("Graph: " + name)
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_CC_in_full_graph(G, CC, name, layout=None, highlight_color='red', base_color='lightgray'):
        """
        Plot the full graph G, highlighting the subgraph CC.

        Args:
            G: The full NetworkX graph.
            CC: The connected component subgraph (nodes must be in G).
            layout: Optional dict of positions. If None, uses spring_layout.
            highlight_color: color for CC edges/nodes.
            base_color: color for non-CC edges/nodes.
        """
        if layout is None:
            layout = nx.spring_layout(G)

        plt.figure(figsize=(8, 8))

        # Draw base graph in light color
        nx.draw_networkx_nodes(G, layout, node_color=base_color, node_size=200)
        nx.draw_networkx_edges(G, layout, edge_color=base_color, width=0.5)
        nx.draw_networkx_labels(G, layout, font_size=8)

        # Draw CC nodes & edges on top
        nx.draw_networkx_nodes(CC, layout, node_color=highlight_color, node_size=300)
        nx.draw_networkx_edges(CC, layout, edge_color=highlight_color, width=2)

        plt.title("Highlighted Connected Component: " + name)
        plt.axis('off')
        plt.show()

    '''

    @staticmethod
    def minimum_spanning_tree_weight(G):
        """
        Find MST and return weight of MST
        """
        T = nx.minimum_spanning_tree(G, weight='weight')
        return sum(d['weight'] for _, _, d in T.edges(data=True))

    @staticmethod
    def cheapest_cross_edge(G, CC_u, CC_v):
        """
        Return cheapest edge connecting two graphs
        """
        min_w = float('inf')
        min_edge = None
        for u in CC_u:
            for v in CC_v:
                if G.has_edge(u, v):
                    w = G[u][v]['weight']
                    if w < min_w:
                        min_w, min_edge = w, (u, v)
        return min_edge, min_w
    
    @staticmethod
    def plot_tree_cover(G, trees, j, layout=None, base_color='lightgrey'):
        """
        Visualize the tree cover on the full graph,
        showing node weights and edge weights.

        Args:
            G: Original graph.
            trees: list of trees to highlight.
            j: The dyadic class index.
        """
        if layout is None:
            layout = nx.spring_layout(G)

        plt.figure(figsize=(10, 10))

        # Draw base graph in light color
        nx.draw_networkx_nodes(G, layout, node_color=base_color, node_size=200)
        nx.draw_networkx_edges(G, layout, edge_color=base_color, width=0.5)

        # Node weights as labels
        node_labels = {v: f"{v}\n({G.nodes[v]['weight']:.2f})" for v in G.nodes}
        nx.draw_networkx_labels(G, layout, labels=node_labels, font_size=8)

        # Edge weights
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels, font_size=6)

        # Highlight each tree with different color
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        for i, T in enumerate(trees):
            c = colors[i % len(colors)]
            nx.draw_networkx_nodes(T, layout, node_color=c, node_size=300)
            nx.draw_networkx_edges(T, layout, edge_color=c, width=2)

        plt.title(f"Tree Cover for W_j = {j}")
        plt.axis('off')
        plt.show()

    def split_tree_by_weight(self, T, beta):
        """
        Split a tree if its weight exceeds beta
        """
        pieces = []
        work = T.copy()
        while self.minimum_spanning_tree_weight(work) > beta:
            # find leaf whose incident edge is heaviest
            leaf_edges = [(u, v, d['weight'])
                          for u, v, d in work.edges(data=True)
                          if work.degree(u) == 1 or work.degree(v) == 1]
            u, v, w = max(leaf_edges, key=lambda x: x[2])

            # remove that leaf edge, take the leaf node piece as a subtree
            work.remove_edge(u, v)

            # identify connected component containing the leaf
            comp_nodes = next(c for c in nx.connected_components(work) if (u in c) ^ (v in c))
            piece = work.subgraph(comp_nodes).copy()
            pieces.append(piece)

            # remove those nodes from work
            work.remove_nodes_from(comp_nodes)

        if work.number_of_nodes() > 0:
            pieces.append(work)

        return pieces

    def rootless_tree_cover(self, B):
        """
        k min max tree cover as defined by W. Xu, W. Liang, and X. Lin.
        Approximation algorithms for min-max cycle cover problems.
        IEEE Transactions on Computers, 64(3):600–613, 2013.

        Input:
            B: parameter controlling max tree weight

        Output:
            Returns a list of trees or raises ValueError if B too low.
        """
        G = self.graph
        k = self.k

        # Line 1-4: Remove edges > B/3 and find CCs
        threshold = B / 3
        G_cut = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d['weight'] <= threshold)
        CCs = [G.subgraph(c).copy() for c in nx.connected_components(G_cut)]

        all_nodes_in_cut = set(G_cut.nodes)
        for v in G.nodes:
            if v not in all_nodes_in_cut:
                CCs.append(G.subgraph([v]).copy())

        # Classify light / heavy by MST weight vs B
        light_CCs = []
        heavy_CCs = []
        for CC in CCs:
            w_mst = self.minimum_spanning_tree_weight(CC)
            # print(w_mst)
            if w_mst <= B:
                light_CCs.append(CC)
            else:
                heavy_CCs.append(CC)

        l, h = len(light_CCs), len(heavy_CCs)

        if l + h >= 8 * k:
            raise ValueError("B is too low")

        # Precompute cheapest cross-edges (reg-reg and reg-heavy) for merging
        cheapest = {}
        for i, CCi in enumerate(light_CCs):
            for j, CCj in enumerate(light_CCs):
                if i < j:
                    edge, w = self.cheapest_cross_edge(G, CCi.nodes, CCj.nodes)
                    cheapest[(i, j)] = cheapest[(j, i)] = (edge, w)
        for i, CCi in enumerate(light_CCs):
            for j, CCj in enumerate(heavy_CCs):
                edge, w = self.cheapest_cross_edge(G, CCi.nodes, CCj.nodes)
                cheapest[(i, l + j)] = (edge, w)

        # Precompute A(CCi) for every light node
        wmin = []
        for i, CCi in enumerate(light_CCs):
            min_edge_w = float('inf')
            for CCj in heavy_CCs:
                for u in CCi.nodes:
                    for v in CCj.nodes:
                        if G.has_edge(u, v) and G[u][v]['weight'] <= B / 2:
                            min_edge_w = min(min_edge_w, G[u][v]['weight'])
            wmin.append(min_edge_w if min_edge_w != float('inf') else 1.0)

        A = [self.minimum_spanning_tree_weight(CCi) + wmin_i for CCi, wmin_i in zip(light_CCs, wmin)]

        # Precompute existence of light–light cross‐edge \leq B/2:
        light_connect = [[False] * l for _ in range(l)]
        for i, j in itertools.combinations(range(l), 2):
            exists = False
            for u in light_CCs[i].nodes:
                for v in light_CCs[j].nodes:
                    if G.has_edge(u, v) and G[u][v]['weight'] <= B / 2:
                        exists = True
                        break
                if exists:
                    break
            light_connect[i][j], light_connect[j][i] = exists, exists

        # Line 5–20: iterate through a, b
        for a in range(0, l + 1):
            for b in range(0, l - a + 1):
                H = nx.Graph()

                regs = list(range(l))
                for i in regs:
                    H.add_node(f"r{i}")

                heavies = list(range(l, l + h))
                selected_heavy = heavies[:a]
                for j in selected_heavy:
                    H.add_node(f"h{j}")

                nulls = [f"n{idx}" for idx in range(b)]
                for n in nulls:
                    H.add_node(n)

                for i, j in itertools.combinations(range(l), 2):
                    if light_connect[i][j]:
                        H.add_edge(f"r{i}", f"r{j}", weight=0.0)

                for i in range(l):
                    for n in nulls:
                        H.add_edge(f"r{i}", n, weight=0.0)

                for i in range(l):
                    for hj in selected_heavy:
                        H.add_edge(f"r{i}", hj, weight=A[i])

                if (H.number_of_nodes() % 2) != 0:
                    continue
                M = nx.algorithms.matching.min_weight_matching(H)
                if len(M) * 2 != H.number_of_nodes():
                    continue

                T_cover = []
                merged_heavy = {j: heavy_CCs[j - l].copy() for j in selected_heavy}

                for u, v in M:
                    if u.startswith('r') and v.startswith('r'):
                        i, j = int(u[1:]), int(v[1:])
                        Ti = nx.minimum_spanning_tree(light_CCs[i], weight='weight')
                        Tj = nx.minimum_spanning_tree(light_CCs[j], weight='weight')
                        (eu, ev), _ = cheapest[(i, j)]
                        Tij = nx.union(Ti, Tj)
                        Tij.add_edge(eu, ev, weight=G[eu][ev]['weight'])
                        T_cover.append(Tij)

                    elif u.startswith('r') and v.startswith('n') or (v.startswith('r') and u.startswith('n')):
                        ri = int(u[1:]) if u.startswith('r') else int(v[1:])
                        Ti = nx.minimum_spanning_tree(light_CCs[ri], weight='weight')
                        T_cover.append(Ti)

                    else:
                        ri = int(u[1:]) if u.startswith('r') else int(v[1:])
                        hj = int(v[1:]) if v.startswith('h') else int(u[1:])
                        CCj_graph = merged_heavy[hj]
                        (eu, ev), _ = cheapest[(ri, hj)]
                        CCj_graph.add_edge(eu, ev, weight=G[eu][ev]['weight'])
                        merged_heavy[hj] = nx.minimum_spanning_tree(CCj_graph, weight='weight')

                matched_nodes = {u for pair in M for u in pair}
                for i in range(l):
                    if f"r{i}" not in matched_nodes:
                        Ti = nx.minimum_spanning_tree(light_CCs[i], weight='weight')
                        T_cover.append(Ti)

                for hj, Tj_modified in merged_heavy.items():
                    wT = sum(d['weight'] for _, _, d in Tj_modified.edges(data=True))
                    if wT < 8 / 3 * B:
                        T_cover.append(Tj_modified)
                    else:
                        pieces = self.split_tree_by_weight(Tj_modified, 8 / 3 * B)
                        T_cover.extend(pieces)

                if len(T_cover) <= k:
                    covered_nodes = {n for T in T_cover for n in T.nodes}
                    if covered_nodes == set(G.nodes):
                        return T_cover

        raise ValueError("B is too low")

    def find_feasible_B(self, step=0.05):
        """
        Try to find a feasible B by gradually scaling the MST weight.
        """
        mst_wt = self.minimum_spanning_tree_weight(self.graph)
        factors = [round(i * step, 2) for i in range(1, int(1.0 / step) + 1)]
        for factor in factors:
            B = factor * mst_wt
            try:
                trees = self.rootless_tree_cover(B)
                return trees, B
            except ValueError:
                continue
        raise ValueError
                
class KRobotAssignment:
    def __init__(self, graph, k, L, beta = 8/3):
        '''
        Initialize the k-robot assignment algorithm
        
        Args:
            graph: NetworkX graph representing the environment
            k: Number of robots
            L: Parameter for distance bounds
            beta: Approximation factor (we set to currently best known approximation factor for minmax)
        '''

        self.graph = graph
        self.k = k
        self.L = L
        self.beta = beta
        
    def get_edge_weight(self, u, v):
        """
        Get weight of edge between u and v
        """

        if self.graph.has_edge(u, v):
            return self.graph[u][v].get('weight', 1)
        return float('inf')
    
    def compute_distance(self, u, v):
        """
        Compute shortest path distance between u and v
        """

        try:
            return nx.shortest_path_length(self.graph, u, v, weight='weight')
        except:
            return float('inf')
    
    @staticmethod
    def round_to_dyadic(w):
        """
        Round weight to the least dyadic value greather than or equal with w
        """
        return 2 ** math.ceil(math.log2(w))
    
    
    @staticmethod
    def plot_robot_assignment(G, robots_trees, layout=None):
        """
        Visualize trees assigned to each robot.
        Each robot's trees are shown in a distinct color.
        No dyadic W_j color-coding.

        Args:
            G: Original NetworkX graph.
            robots_trees: {robot_id: [T, ...]} — each robot’s list of trees.
            layout: Optional layout dict.
        """
        if layout is None:
            layout = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(12, 12))

        # Draw base graph
        nx.draw_networkx_nodes(G, layout, node_color='lightgrey', node_size=200)
        nx.draw_networkx_edges(G, layout, edge_color='lightgrey', width=0.5)

        # Node weights
        node_labels = {
            v: f"{v}\n({G.nodes[v].get('weight', 0):.2f})" for v in G.nodes
        }
        nx.draw_networkx_labels(G, layout, labels=node_labels, font_size=8)

        # Edge weights
        edge_labels = {
            (u, v): f"{G[u][v].get('weight', 0):.2f}" for u, v in G.edges
        }
        nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels, font_size=6)

        # Distinct color per robot
        colors = plt.cm.get_cmap('tab10', len(robots_trees))
        robot_colors = {robot_id: colors(i) for i, robot_id in enumerate(robots_trees)}

        for robot_id, trees in robots_trees.items():
            color = robot_colors[robot_id]
            for T in trees:
                nx.draw_networkx_nodes(
                    T, layout, node_color=[color], node_size=300, alpha=0.9
                )
                nx.draw_networkx_edges(
                    T, layout, edge_color=[color], width=2, alpha=0.9
                )

        for robot_id, color in robot_colors.items():
            plt.scatter([], [], color=color, label=f"Robot {robot_id}")
        plt.legend(title="Robots", loc='upper left')

        plt.title("Robot Assignment")
        plt.axis('off')
        plt.show()
    
    def group_by_dyadic_weights(self):
        """
        Group nodes by dyadic weight class.
        Returns:
            W: {j: set of nodes with dyadic weight 1/2^j}
            m: max dyadic exponent.
        """
        weights = {v: self.graph.nodes[v]['weight'] for v in self.graph.nodes()}
        rounded = {v: self.round_to_dyadic(w) for v, w in weights.items()}

        wmax = max(rounded.values())
        wmin = min(rounded.values())
        ratio = wmax / wmin if wmin > 0 else 1
        m = max(1, int(round(math.log2(ratio))))

        W = {}
        for v, w in rounded.items():
            j = int(round(math.log2(wmax / w)))
            if j not in W:
                W[j] = set()
            W[j].add(v)
        # print(m)
        return W, m
    
    def k_robot_assignment(self, L):
        """
        Implements the k-robot assignment.
        Returns:
            {robot_id: [trees]}, or False if fails.
            m: the floor of log(largest_node_weight / smallest_node_weight)
        """
        W, m = self.group_by_dyadic_weights()

        T_Wj = {}       # Trees for Wj

        # Step 1: For each Wj, find a feasible tree cover with the smallest T using the approximation algorithm
        for j, Wj_nodes in W.items():
            subgraph = self.graph.subgraph(Wj_nodes).copy()
            
            found = False

            for t in range(1, self.k + 1):
                treecover = KMinMaxTreeCover(subgraph, t)
                try:
                    trees, B = treecover.find_feasible_B()
                    if B <= self.beta * (2 ** j) * L:
                        # KMinMaxTreeCover.plot_tree_cover(self.graph, trees, j)
                        T_Wj[j] = (trees, t)        # (trees, number of trees)
                        found = True
                        break
                except ValueError:                  # error raised if B too small
                    continue

            if not found:
                return False

        # Step 2: Assign trees to robots
        robots = {r: {"trees": [], "depot": None} for r in range(self.k)}
        free_robots = set(robots.keys())

        for j in sorted(W.keys()):
            trees, _ = T_Wj[j]           
            for T in trees:    
                Q = set(T.nodes)
                # print(Q)
                for r in robots:
                    if robots[r]["depot"] is not None:  # if robot has depot, find j s.t. depot in V_j
                        depot = robots[r]["depot"]
                        print(depot)

                        j0 = None
                        for jj, Wjj in W.items():
                            if depot in Wjj:
                                j0 = jj
                                break
                        assert j0 is not None           # depot must be in one of j!

                        print("Q:")
                        print(Q)
                        print("Threshold")
                        print(self.k * (2 ** j0) * L)

                        for v in Q:
                            print(nx.shortest_path_length(self.graph, v, depot, weight='weight'))

                        Q0 = {v for v in Q if nx.shortest_path_length(self.graph, v, depot,  weight='weight') <= self.k * (2 ** j0) * L}

                        if Q0:
                            MST_Q0 = nx.minimum_spanning_tree(self.graph.subgraph(Q0))
                            robots[r]["trees"].append(MST_Q0)
                            Q -= Q0

                if Q:
                    if not free_robots:                 # assign the rest of the vertices to a new robot
                        return False
                    new_r = free_robots.pop()
                    MST_Q = nx.minimum_spanning_tree(self.graph.subgraph(Q))
                    robots[new_r]["trees"].append(MST_Q)
                    robots[new_r]["depot"] = list(MST_Q.nodes)[0]

        return {r: robots[r]["trees"] for r in robots if robots[r]["trees"]}, m

    def schedule(self):
        """
        Find the smallest feasible L* using binary search.
        Returns:
            robots_trees: {robot_id: [trees]}
            L*: final feasible L
            m: floor of log(largest_node_weight / smallest_node_weight)
        """
        # Compute initial upper bound and lower bound for L
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')
        dmax = sum(d['weight'] for _, _, d in mst.edges(data=True))
        L_low = 1e-6  # or small positive epsilon
        L_high = dmax

        best_result = None
        best_m = None
        best_L = None

        # Binary search loop
        for _ in range(50):  # limit iterations to prevent infinite loop
            L_mid = (L_low + L_high) / 2
            result = self.k_robot_assignment(L_mid)

            if result is False:
                # L too small, need larger L
                L_low = L_mid
            else:
                # L feasible — tighten upper bound
                robots_trees, m = result
                best_result = robots_trees
                best_m = m
                best_L = L_mid
                L_high = L_mid
            
            print(L_mid)

            # Early stopping if bounds converge
            if abs(L_high - L_low) < 1e-6:
                break

        if best_result is None:
            raise ValueError("Could not find feasible L in search range.")

        return best_result, best_L, best_m
        
class SingleRobotSchedule:
    def __init__(self, trees, L, k, m):
        """
        Args:
            trees: list of trees for this robot, [T0, T1, ...], T0 is depot.
            L: the guessed bound L
            k: number of robots (used for epsilon)
            m: the floor of log(largest_node_weight / smallest_node_weight)
        """
        self.trees = trees
        self.L = L
        self.k = k
        self.m = m
        self.paths_per_tree = []
        self.epsilon = None
        self.tours = []
        self.paths = []
        self.idx = []
    
    def compute_schedule(self):
        """
        Compute tours, split to paths, index for round robin.
        """
        T0 = self.trees[0]
        w0 = sum(T0.nodes[v]['weight'] for v in T0.nodes)
        self.epsilon = 2 * self.k * self.L / w0

        for T in self.trees:
            # Step 1: Approximate a TSP tour over T
            if T.number_of_nodes() == 0:
                continue 
            elif T.number_of_nodes() == 1:  
                node = next(iter(T.nodes))
                self.tours.append((node, 0))
                self.paths_per_tree.append([[node]])
                self.idx.append(0)
                continue
            else:
                tour_nodes = list(nx.approximation.traveling_salesman_problem(T, weight='weight'))
                tour_length = sum(
                    T[u][v]['weight'] for u, v in zip(tour_nodes, tour_nodes[1:] + tour_nodes[:1])
                    if T.has_edge(u, v)
                )
                self.tours.append((tour_nodes, tour_length))

            # Step 2: Partition the tour into paths of length at most epsilon
            current_path = []
            current_len = 0
            paths = []
            for i in range(len(tour_nodes) - 1):
                u, v = tour_nodes[i], tour_nodes[i + 1]
                w = T[u][v]['weight']
                if current_len + w > self.epsilon:          # start new path
                    paths.append(current_path)
                    current_path = [v]
                    current_len = w
                else:                                       # old path
                    current_path.append(v)
                    current_len += w
            if current_path:                                
                paths.append(current_path)
            if paths:
                # print("Path: ")
                # print(paths)
                self.paths_per_tree.append(paths)
                self.idx.append(0)
        print("Path:")
        print(self.paths_per_tree)

    def round_robin_sequence(self, steps):
        """
        Generate sequence of (tree_index, segment) for given steps.
        """
        seq = []
        h = len(self.paths_per_tree)
        if h == 0:
            return seq

        i = 0
        idx = self.idx[:]  # safe copy

        for _ in range(steps):
            seq.append(self.paths_per_tree[i][idx[i]])
            while not self.paths_per_tree[i]:  # skip empty
                i = (i + 1) % h
            idx[i] = (idx[i] + 1) % len(self.paths_per_tree[i])
            i = (i + 1) % h
            # print(seq)
        return seq

    def full_round_robin_sequence(self):
        """
        Return the entire cycle of segments once.
        """
        total = int(self.k ** 2 * math.log(max(self.k ** 2 * self.m, 2)) * self.L)
        return self.round_robin_sequence(total)

    def get_routes(self):
        """
        Return list of segments (node lists) in full cycle order.
        """
        # print(self.full_round_robin_sequence())
        return self.full_round_robin_sequence()

def schedule_all_routes(graph: nx.Graph, k: int):
    assignment = KRobotAssignment(graph, k, L=0)
    robots_trees, L, m = assignment.schedule()
    routes = {}
    for r, trees in robots_trees.items():
        sched = SingleRobotSchedule(trees, L, k, m)
        sched.compute_schedule()
        routes[r] = sched.get_routes()
    # print_all_routes(routes)
    return routes, L, robots_trees

def print_all_routes(routes):
    for robot_id, segments in routes.items():
        flat_path = []
        for seg in segments:
            if isinstance(seg, (list, tuple)):
                flat_path.extend(seg)
            else:
                flat_path.append(seg)

        # Remove consecutive duplicates
        cleaned_path = []
        for node in flat_path:
            if not cleaned_path or cleaned_path[-1] != node:
                cleaned_path.append(node)

        print(f"\nRobot {robot_id} full route: {cleaned_path}")

def visualize_routes(graph: nx.Graph, routes: dict, robots_trees: dict, L, k, coords=None):
    """
    Visualize each robot's route clusters (trees) with their paths and latencies.

    Args:
        graph: The original graph.
        routes: {robot_id: [segments]} full segmented routes per robot.
        robots_trees: {robot_id: [trees]} assigned trees per robot.
        L: The guessed bound L.
        k: Number of robots.
        coords: dict {node: (x,y)} fixed node positions (optional).

    Prints cluster info per robot and plots routes.
    """
    if coords is None:
        pos = nx.spring_layout(graph, seed=42)
    else:
        pos = coords

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    plt.figure(figsize=(10, 10))

    # Draw base graph with nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_color='lightgrey', node_size=500)
    nx.draw_networkx_edges(graph, pos, edge_color='lightgrey', width=0.5)

    node_labels = {
        n: f"{n}\n(w={graph.nodes[n].get('weight', 0):.2f})"
        for n in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)

    edge_labels = {
        (u, v): f"{graph[u][v].get('weight', 0):.2f}"
        for u, v in graph.edges
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    single_node_loop_handles = []
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for r, segments in routes.items():
        c = colors[r % len(colors)]

        flat_path = []
        for seg in segments:
            if isinstance(seg, int):
                flat_path.append(seg)
            else:
                flat_path.extend(seg)

        cleaned_path = []
        for node in flat_path:
            if not cleaned_path or cleaned_path[-1] != node:
                cleaned_path.append(node)

        print("hi")
        # Identify if path is a single node loop
        if len(cleaned_path) == 1:
            node = cleaned_path[0]
            DG = nx.DiGraph()
            DG.add_edge(node, node)  # self-loop edge
            
            nx.draw_networkx_edges(
                DG, pos,
                edgelist=[(node, node)],
                edge_color=c, width=2,
                arrows=True, arrowsize=25,
                connectionstyle='arc3,rad=0.3'  # slightly curved loop
            )
            plt.scatter([], [], color=c, label=f"Robot {r} single-node loop")

        else:
            # Normal multi-node path edges
            directed_edges = list(zip(cleaned_path[:-1], cleaned_path[1:]))

            DG = nx.DiGraph()
            DG.add_edges_from(directed_edges)

            nx.draw_networkx_edges(
                DG, pos,
                edgelist=directed_edges,
                edge_color=c, width=2,
                arrows=True, arrowsize=25,
                connectionstyle='arc3,rad=0.1'
            )
            plt.scatter([], [], color=c, label=f"Robot {r}")

    plt.title(f"Routes for {len(routes)} robots")

    plt.axis('off')

    max_latency, _, max_latency_node, max_latency_cycle, max_cycle_length = compute_max_latency_weighted(routes, graph)

    plt.text(
        0, 1.05,
        f"Max Latency: {max_latency:.3f}\nNode: {max_latency_node}",
        transform=plt.gca().transAxes,
        fontsize=12,
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )

    plt.show()

    # Print details
    print(f"Max Latency: {max_latency:.3f}")
    print(f"Node causing max latency: {max_latency_node}")
    print(f"Cycle length: {max_cycle_length:.3f}")
    print(f"Cycle path: {max_latency_cycle}")


def compute_max_latency_weighted(routes, graph, node_weight_attr='weight', edge_weight_attr='weight'):
    """
    Compute max weighted latency:
      For each node, find its max time between visits in the route.
      Multiply by node weight.
      Return the maximum across all nodes.

    Args:
        routes: {robot_id: [segments]}
        graph: nx.Graph
    Returns:
        max_latency: float
        node_latency: {node: weighted latency}
        max_latency_node: node with max
        max_latency_cycle: cycle where max occurs
        max_cycle_length: cycle length where max occurs
    """
    max_latency = 0
    node_latency = {}
    max_latency_node = None
    max_latency_cycle = None
    max_cycle_length = 0

    for r, segments in routes.items():
        # Flatten segments
        flat_path = []
        for seg in segments:
            if isinstance(seg, int):
                flat_path.append(seg)
            else:
                flat_path.extend(seg)

        if len(flat_path) < 2:
            continue

        # Remove consecutive duplicates AND handle cycle wrap-around
        cleaned_path = []
        for n in flat_path:
            if not cleaned_path or cleaned_path[-1] != n:
                cleaned_path.append(n)
        # Also check head/tail
        if len(cleaned_path) > 1 and cleaned_path[0] == cleaned_path[-1]:
            cleaned_path.pop()

        # print(cleaned_path)
        # If only one unique vertex, latency is 0
        if len(set(cleaned_path)) == 1:
            return 0, {}, None, None, 0

        if len(cleaned_path) < 2:
            continue

        # Compute edge lengths
        edge_lengths = []
        for i in range(len(cleaned_path)):
            u = cleaned_path[i]
            v = cleaned_path[(i + 1) % len(cleaned_path)]
            w = graph[u][v].get(edge_weight_attr, 1)
            edge_lengths.append(w)

        total_cycle_length = sum(edge_lengths)

        print("hiii")
        # For each node, compute max gap between visits
        node_positions = {}
        for idx, node in enumerate(cleaned_path):
            node_positions.setdefault(node, []).append(idx)

        for node, positions in node_positions.items():
            gaps = []
            for i in range(len(positions)):
                a = positions[i]
                b = positions[(i + 1) % len(positions)]
                if b > a:
                    gap_edges = edge_lengths[a:b]
                else:
                    gap_edges = edge_lengths[a:] + edge_lengths[:b]
                gap_length = sum(gap_edges)
                gaps.append(gap_length)

            max_gap = max(gaps)
            w_node = graph.nodes[node].get(node_weight_attr, 1)
            latency = w_node * max_gap

            node_latency[node] = max(node_latency.get(node, 0), latency)

            if latency > max_latency:
                max_latency = latency
                max_latency_node = node
                max_latency_cycle = cleaned_path
                max_cycle_length = total_cycle_length

    return max_latency, node_latency, max_latency_node, max_latency_cycle, max_cycle_length

# === Testing Ground ===

# Example 1: General Graph with uniform random placement in square [-1,1]^2
G = nx.complete_graph(30)
coords = {i: (random.uniform(-1, 1), random.uniform(-1, 1)) for i in G.nodes()}
for u, v in G.edges():
    G[u][v]['weight'] = math.dist(coords[u], coords[v])
for v in G.nodes():
    G.nodes[v]['weight'] = random.random()

k = 4
start_time = time.time()
routes, L, trees = schedule_all_routes(G, k)
end_time = time.time()
print(f"Algorithm took {end_time - start_time:.6f} seconds for Graph 1.")

visualize_routes(G, routes, trees, L, k, coords)

# Example 2: Clustered graph with clusters randomly placed in separated squares on x-axis
G2 = nx.Graph()
num_clusters = 5
cluster_size = 6

coords2 = {}
square_size = 2
gap = 3  # spacing between cluster squares on x-axis

for c in range(num_clusters):
    base_x = c * (square_size + gap)
    base_y = 0
    nodes = [c * cluster_size + i for i in range(cluster_size)]
    for node in nodes:
        G2.add_node(node, weight=random.uniform(0.2, 1.0))
        x = random.uniform(base_x, base_x + square_size)
        y = random.uniform(base_y, base_y + square_size)
        coords2[node] = (x, y)

    # Edges within cluster
    for i in nodes:
        for j in nodes:
            if i < j:
                G2.add_edge(i, j, weight=math.dist(coords2[i], coords2[j]))

all_nodes = list(G2.nodes)
for i in all_nodes:
    for j in all_nodes:
        if (i // cluster_size) != (j // cluster_size) and i < j:
            G2.add_edge(i, j, weight=10.0)  # big penalty for cross-cluster

k = 4
start_time = time.time()
routes2, L2, trees2 = schedule_all_routes(G2, k)
end_time = time.time()
print(f"Algorithm took {end_time - start_time:.6f} seconds for Graph 2.")

visualize_routes(G2, routes2, trees2, L2, k, coords2)

# === Example 3: Forced multi-robot test ===

G3 = nx.Graph()
num_clusters = 4      
cluster_size = 2

coords3 = {}
square_size = 1
gap = 1000  # VERY large gap to ensure separation forces multiple robots

# Place 4 clusters far apart
for c in range(num_clusters):
    base_x = c * (square_size + gap)
    base_y = 0
    nodes = [c * cluster_size + i for i in range(cluster_size)]
    for node in nodes:
        G3.add_node(node, weight=random.uniform(0.5, 1.0))  # non-trivial weight
        x = random.uniform(base_x, base_x + square_size)
        y = random.uniform(base_y, base_y + square_size)
        coords3[node] = (x, y)

    # Fully connect inside each cluster
    for i in nodes:
        for j in nodes:
            if i < j:
                G3.add_edge(i, j, weight=math.dist(coords3[i], coords3[j]))

# Add sparse cross-cluster edges with very high cost to ensure separation
for i in range(num_clusters * cluster_size):
    for j in range(num_clusters * cluster_size):
        if (i // cluster_size) != (j // cluster_size) and i < j:
            G3.add_edge(i, j, weight=10000)

k = 4

start_time = time.time()
routes3, L3, trees3 = schedule_all_routes(G3, k)
end_time = time.time()
print(f"Algorithm took {end_time - start_time:.6f} seconds for Graph 3.")

visualize_routes(G3, routes3, trees3, L3, k, coords3)
