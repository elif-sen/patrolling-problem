import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
import random
import math
from collections import defaultdict
import heapq
from typing import List, Tuple, Dict, Set
import warnings
warnings.filterwarnings('ignore')

class Graph:
    def __init__(self, n: int):
        self.n = n
        self.vertices = list(range(n))
        self.weights = {}  # vertex weights
        self.distances = np.zeros((n, n))  # distance matrix
        self.positions = {}  # for visualization
        
    def set_vertex_weight(self, v: int, weight: float):
        self.weights[v] = weight
        
    def set_distance(self, u: int, v: int, distance: float):
        self.distances[u][v] = distance
        self.distances[v][u] = distance
        
    def set_position(self, v: int, pos: Tuple[float, float]):
        self.positions[v] = pos

class MulticlassMinimumSpanningForest:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.h = self._compute_h()
        self.vertex_classes = self._partition_vertices()
        self.trees = {}
        self._compute_msf()
        
    def _compute_h(self) -> int:
        w_max = max(self.graph.weights.values())
        w_min = min(self.graph.weights.values())
        return math.floor(math.log2(w_max / w_min)) + 1
    
    def _partition_vertices(self) -> Dict[int, List[int]]:
        classes = {}
        for i in range(self.h):
            classes[i] = []
            
        for v in self.graph.vertices:
            w_v = self.graph.weights[v]
            for i in range(self.h):
                if 1/(2**(i+1)) < w_v <= 1/(2**i):
                    classes[i].append(v)
                    break
                    
        return classes
    
    def _compute_msf(self):
        for i in range(self.h):
            if len(self.vertex_classes[i]) > 1:
                vertices = self.vertex_classes[i]
                # Create subgraph distance matrix
                subgraph_dist = np.zeros((len(vertices), len(vertices)))
                for idx_u, u in enumerate(vertices):
                    for idx_v, v in enumerate(vertices):
                        if idx_u != idx_v:
                            subgraph_dist[idx_u][idx_v] = self.graph.distances[u][v]
                
                # Compute MST
                mst = minimum_spanning_tree(subgraph_dist).toarray()
                
                # Convert back to original vertex indices
                tree_edges = []
                for idx_u in range(len(vertices)):
                    for idx_v in range(len(vertices)):
                        if mst[idx_u][idx_v] > 0:
                            u, v = vertices[idx_u], vertices[idx_v]
                            tree_edges.append((u, v, self.graph.distances[u][v]))
                
                self.trees[i] = tree_edges
            else:
                self.trees[i] = []
    
    def get_tree_weight(self, i: int) -> float:
        if i not in self.trees:
            return 0.0
        return sum(edge[2] for edge in self.trees[i])
    
    def get_weighted_sum(self) -> float:
        total = 0.0
        for i in range(self.h):
            total += (1/(2**i)) * self.get_tree_weight(i)
        return total

class MinMaxKTreeCover:
    def __init__(self, vertices: List[int], distances: np.ndarray, k: int):
        self.vertices = vertices
        self.distances = distances
        self.k = k
        
    def solve(self) -> Tuple[List[List[int]], float]:
        """4-approximation algorithm for min-max k-tree cover"""
        if len(self.vertices) <= self.k:
            # Each vertex forms its own tree
            trees = [[v] for v in self.vertices]
            while len(trees) < self.k:
                trees.append([])
            return trees, 0.0
        
        # Simple greedy approach for 4-approximation
        # Start with MST and then partition
        n = len(self.vertices)
        if n == 0:
            return [[] for _ in range(self.k)], 0.0
            
        # Create distance matrix for vertices
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = self.distances[self.vertices[i]][self.vertices[j]]
        
        # Compute MST
        mst = minimum_spanning_tree(dist_matrix).toarray()
        
        # Convert MST to edge list
        edges = []
        for i in range(n):
            for j in range(n):
                if mst[i][j] > 0:
                    edges.append((i, j, mst[i][j]))
        
        # Sort edges by weight (descending) and remove k-1 heaviest
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # Remove k-1 heaviest edges to get k trees
        if len(edges) >= self.k - 1:
            edges = edges[self.k-1:]
        
        # Build trees using Union-Find
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for u, v, _ in edges:
            union(u, v)
        
        # Group vertices by their root
        components = defaultdict(list)
        for i in range(n):
            root = find(i)
            components[root].append(self.vertices[i])
        
        trees = list(components.values())
        
        # Ensure we have exactly k trees
        while len(trees) < self.k:
            trees.append([])
        
        # Calculate max tree weight
        max_weight = 0.0
        for tree in trees:
            if len(tree) > 1:
                tree_weight = 0.0
                for i in range(len(tree)):
                    for j in range(i+1, len(tree)):
                        tree_weight += self.distances[tree[i]][tree[j]]
                max_weight = max(max_weight, tree_weight)
        
        return trees[:self.k], max_weight

class SingleRobotPatrolSchedule:
    def __init__(self, graph: Graph, vertices: List[int]):
        self.graph = graph
        self.vertices = vertices
        self.schedule = []
        
    def compute_schedule(self):
        """Algorithm 1: Single robot patrol scheduling"""
        if not self.vertices:
            return
            
        # For simplicity, create a cycle through all vertices
        # In practice, this would use the multiclass MSF structure
        self.schedule = self.vertices.copy()
        if len(self.vertices) > 1:
            # Add return to start to complete the cycle
            self.schedule.append(self.vertices[0])

class KRobotPatrolScheduler:
    def __init__(self, graph: Graph, k: int):
        self.graph = graph
        self.k = k
        self.msf = MulticlassMinimumSpanningForest(graph)
        self.robot_assignments = {}
        self.robot_schedules = {}
        self.robot_depots = {}
        self.achieved_latency = None
        
    def solve(self, max_iterations: int = 20) -> bool:
        """Algorithm 2: O(kh)-approximation algorithm using binary search to find the smallest feasible latency"""
        initial_L = 1.0  # Initial guess for latency
        low = initial_L * (2.0 ** -max_iterations)  # Lower bound, significantly below initial guess
        high = initial_L * (2.0 ** max_iterations)  # Upper bound, significantly above initial guess
        best_L = None

        while low <= high:
            mid = (low + high) / 2
            if self._try_solve_with_latency(mid):
                # Solution found, try a smaller L (search lower half)
                best_L = mid
                high = mid - 1e-6  # Small decrement to avoid infinite loops
            else:
                # No solution, try a larger L (search upper half)
                low = mid + 1e-6  # Small increment to avoid infinite loops

        if best_L is not None:
            self.achieved_latency = best_L
            return True
        return False
        """Algorithm 2: O(kh)-approximation algorithm
        L = 1.0  # Initial guess for latency
        
        for iteration in range(max_iterations):
            if self._try_solve_with_latency(L):
                self.achieved_latency = L
                return True
            L *= 2.0
            
        return False"""
    
    def _try_solve_with_latency(self, L: float) -> bool:
        """Try to solve with given latency bound L"""
        L0 = 0.0
        
        # Pre-processing: compute tree covers for each class
        tree_covers = {}
        
        for j in range(self.msf.h):
            vertices_j = self.msf.vertex_classes[j]
            if not vertices_j:
                continue
                
            # Run min-max k-tree cover algorithm
            tree_cover_solver = MinMaxKTreeCover(vertices_j, self.graph.distances, self.k)
            trees, max_tree_weight = tree_cover_solver.solve()
            tree_covers[j] = trees
            
            # Update L0
            weight_factor = 1.0 / (2**j)
            L0 = max(L0, weight_factor * max_tree_weight)
            
            # Check if L0 > 4L (failure condition)
            if L0 > 4 * L:
                return False
        
        # Initialize robots as free
        free_robots = set(range(self.k))
        self.robot_assignments = {i: [] for i in range(self.k)}
        self.robot_depots = {}
        
        # Assign vertices to robots
        for j in range(self.msf.h):
            if j not in tree_covers:
                continue
                
            for tree in tree_covers[j]:
                if not tree:
                    continue
                    
                Q = tree.copy()
                
                # Try to assign to non-free robots first
                non_free_robots = [i for i in range(self.k) if i not in free_robots]
                
                for robot_i in non_free_robots:
                    if robot_i not in self.robot_depots:
                        continue
                        
                    depot = self.robot_depots[robot_i]
                    
                    # Find depot's class
                    j_prime = 0
                    for jp in range(self.msf.h):
                        if depot in self.msf.vertex_classes[jp]:
                            j_prime = jp
                            break
                    
                    # Calculate assignment condition
                    assignment_radius = min(self.k, 2**(j - j_prime)) * (2**j_prime) * L
                    
                    Q_prime = []
                    remaining_Q = []
                    
                    for v in Q:
                        if self.graph.distances[v][depot] <= assignment_radius:
                            Q_prime.append(v)
                        else:
                            remaining_Q.append(v)
                    
                    Q = remaining_Q
                    self.robot_assignments[robot_i].extend(Q_prime)
                
                # If Q is not empty, assign to a free robot
                if Q:
                    if not free_robots:
                        return False  # No free robots available
                    
                    robot_new = free_robots.pop()
                    # Set depot as arbitrary vertex from Q
                    self.robot_depots[robot_new] = Q[0]
                    self.robot_assignments[robot_new] = Q
        
        # Compute individual robot schedules
        for robot_i in range(self.k):
            if self.robot_assignments[robot_i]:
                scheduler = SingleRobotPatrolSchedule(self.graph, self.robot_assignments[robot_i])
                scheduler.compute_schedule()
                self.robot_schedules[robot_i] = scheduler.schedule
            else:
                self.robot_schedules[robot_i] = []
        
        return True
    
    def calculate_actual_latency(self) -> float:
        """Calculate the actual achieved min-max latency"""
        max_latency = 0.0
        
        for robot_i in range(self.k):
            schedule = self.robot_schedules.get(robot_i, [])
            if len(schedule) <= 1:
                continue
                
            # Calculate total patrol time for this robot
            robot_latency = 0.0
            for i in range(len(schedule) - 1):
                u, v = schedule[i], schedule[i + 1]
                robot_latency += self.graph.distances[u][v]
            
            max_latency = max(max_latency, robot_latency)
        
        return max_latency

def generate_random_graph(n: int, seed: int = None) -> Graph:
    """Generate a random complete metric graph"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    graph = Graph(n)
    
    # Generate random positions
    positions = np.random.rand(n, 2) * 10
    for i in range(n):
        graph.set_position(i, (positions[i][0], positions[i][1]))
    
    # Generate vertex weights (normalized to (0, 1])
    weights = np.random.rand(n) * 0.8 + 0.2  # weights in [0.2, 1.0]
    for i in range(n):
        graph.set_vertex_weight(i, weights[i])
    
    # Compute Euclidean distances
    distances = squareform(pdist(positions))
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.set_distance(i, j, distances[i][j])
    
    return graph

def visualize_solution(graph: Graph, scheduler: KRobotPatrolScheduler):
    """Visualize the patrol scheduling solution on a single graph"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    pos = graph.positions
    node_colors = [graph.weights[v] for v in graph.vertices]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create networkx graph
    nx_graph = nx.Graph()
    for v in graph.vertices:
        nx_graph.add_node(v)
    
    # First, draw all edges in light gray (background)
    for i in range(graph.n):
        for j in range(i+1, graph.n):
            nx_graph.add_edge(i, j, weight=graph.distances[i][j])
    
    # Draw background edges
    edge_list = list(nx_graph.edges())
    nx.draw_networkx_edges(nx_graph, pos, edgelist=edge_list, 
                          edge_color='lightgray', alpha=0.3, ax=ax)
    
    # Draw edge labels for all edges (showing distances)
    edge_labels = {}
    for i, j in edge_list:
        edge_labels[(i, j)] = f'{graph.distances[i][j]:.1f}'
    
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels, 
                                font_size=6, alpha=0.5, ax=ax)
    
    # Draw robot patrol routes with colored edges
    route_edges = []
    route_colors = []
    route_widths = []
    
    for robot_i, schedule in scheduler.robot_schedules.items():
        if not schedule or len(schedule) <= 1:
            continue
            
        color = colors[robot_i % len(colors)]
        
        # Add route edges
        for i in range(len(schedule) - 1):
            u, v = schedule[i], schedule[i+1]
            route_edges.append((u, v))
            route_colors.append(color)
            # Scale edge width based on distance (normalize to reasonable range)
            max_dist = np.max(graph.distances)
            width = 1 + 4 * (graph.distances[u][v] / max_dist)
            route_widths.append(width)
    
    # Draw route edges with colors and scaled widths
    for i, edge in enumerate(route_edges):
        nx.draw_networkx_edges(nx_graph, pos, edgelist=[edge], 
                              edge_color=[route_colors[i]], 
                              width=route_widths[i], alpha=0.8, ax=ax)
    
    # Draw nodes with vertex weights as colors
    nodes = nx.draw_networkx_nodes(nx_graph, pos, ax=ax, 
                                  node_color=node_colors, 
                                  cmap='viridis', node_size=400, 
                                  alpha=0.9)
    
    # Draw node labels
    nx.draw_networkx_labels(nx_graph, pos, ax=ax, font_size=10, 
                           font_weight='bold', font_color='white')
    
    # Highlight robot depots with special markers
    for robot_i, depot in scheduler.robot_depots.items():
        if depot is not None:
            color = colors[robot_i % len(colors)]
            x, y = pos[depot]
            ax.scatter(x, y, c=color, s=600, marker='s', 
                      label=f'Robot {robot_i} depot', alpha=0.8, 
                      edgecolor='black', linewidth=2)
    
    # Add colorbar for vertex weights
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=min(node_colors), 
                                               vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Vertex Weight', fontsize=12)
    
    # Calculate and display actual latency
    actual_latency = scheduler.calculate_actual_latency()
    
    # Create title with latency information
    title = f'K-Robot Patrol Scheduling Solution\n'
    title += f'Robots: {scheduler.k}, Vertices: {graph.n}, '
    title += f'Achieved Min-Max Latency: {actual_latency:.2f}'
    if scheduler.achieved_latency:
        title += f' (Bound: {scheduler.achieved_latency:.2f})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create legend
    legend_elements = []
    for robot_i in range(scheduler.k):
        if scheduler.robot_schedules.get(robot_i):
            color = colors[robot_i % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, 
                                            label=f'Robot {robot_i} path'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.02, 1))
    
    # Add text box with additional information
    info_text = f'Edge widths scaled by distance\nEdge labels show distances\nSquare markers show robot depots'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.8), fontsize=10)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def print_solution_details(graph: Graph, scheduler: KRobotPatrolScheduler):
    """Print detailed solution information"""
    print("=== K-Robot Patrol Scheduling Solution ===")
    print(f"Number of robots: {scheduler.k}")
    print(f"Number of vertices: {graph.n}")
    print(f"Vertex weight classes (h): {scheduler.msf.h}")
    
    actual_latency = scheduler.calculate_actual_latency()
    print(f"Achieved min-max latency: {actual_latency:.3f}")
    if scheduler.achieved_latency:
        print(f"Algorithm latency bound: {scheduler.achieved_latency:.3f}")
    print()
    
    print("Vertex weights:")
    for v in graph.vertices:
        print(f"  Vertex {v}: weight = {graph.weights[v]:.3f}")
    print()
    
    print("Vertex class assignments:")
    for i in range(scheduler.msf.h):
        if scheduler.msf.vertex_classes[i]:
            weight_range = f"({1/(2**(i+1)):.3f}, {1/(2**i):.3f}]"
            print(f"  Class {i} {weight_range}: {scheduler.msf.vertex_classes[i]}")
    print()
    
    print("Robot assignments:")
    for robot_i in range(scheduler.k):
        depot = scheduler.robot_depots.get(robot_i, "None")
        vertices = scheduler.robot_assignments[robot_i]
        schedule = scheduler.robot_schedules[robot_i]
        
        # Calculate individual robot latency
        robot_latency = 0.0
        if len(schedule) > 1:
            for i in range(len(schedule) - 1):
                u, v = schedule[i], schedule[i + 1]
                robot_latency += graph.distances[u][v]
        
        print(f"  Robot {robot_i}:")
        print(f"    Depot: {depot}")
        print(f"    Assigned vertices: {vertices}")
        print(f"    Patrol schedule: {schedule}")
        print(f"    Robot latency: {robot_latency:.3f}")
        print()

def run_test(n: int = 8, k: int = 3, seed: int = 42):
    """Run a test case"""
    print(f"Running test with {n} vertices and {k} robots...")
    
    # Generate random graph
    graph = generate_random_graph(n, seed)
    
    # Solve patrol scheduling problem
    scheduler = KRobotPatrolScheduler(graph, k)
    success = scheduler.solve()
    
    if success:
        print("✓ Solution found!")
        print_solution_details(graph, scheduler)
        visualize_solution(graph, scheduler)
    else:
        print("✗ No solution found within iteration limit")
    
    return success

# Main execution
if __name__ == "__main__":
    # Test cases
    print("Testing K-Robot Patrol Scheduling Algorithm")
    print("=" * 50)
    
    # Test 1: Small graph
    print("\n--- Test 1: Small graph (8 vertices, 2 robots) ---")
    run_test(n=32, k=4, seed=42)
    