"""
mst.py - Minimum Spanning Tree algorithm and MST-based agent.

Implements Prim's MST algorithm for computing optimal resource visitation
order, and an MSTAgent that follows the MST traversal path to collect
resources efficiently. Models network/infrastructure exploration.
"""

import heapq
from collections import defaultdict
from agent import MemoryAgent
from config import DIRECTIONS, CellType
from utils import manhattan_distance, bfs_path, direction_from_positions


def compute_mst(nodes: list) -> list:
    """Compute the Minimum Spanning Tree over a set of positions.

    Uses Prim's algorithm with Manhattan distance as edge weights.
    Connects all given nodes into a tree with minimum total weight.

    Args:
        nodes: List of (row, col) positions (resource locations + goal).

    Returns:
        List of (weight, node1, node2) tuples representing MST edges.
        Returns empty list if fewer than 2 nodes.
    """
    if len(nodes) < 2:
        return []

    # Build complete graph with Manhattan distances
    in_mst = set()
    mst_edges = []
    start = nodes[0]
    in_mst.add(start)

    # Priority queue: (weight, from_node_idx, to_node_idx)
    heap = []
    node_set = set(range(len(nodes)))

    for i in range(1, len(nodes)):
        dist = manhattan_distance(start, nodes[i])
        heapq.heappush(heap, (dist, 0, i))

    while heap and len(in_mst) < len(nodes):
        weight, from_idx, to_idx = heapq.heappop(heap)
        if to_idx in in_mst:
            continue

        in_mst.add(to_idx)
        mst_edges.append((weight, nodes[from_idx], nodes[to_idx]))

        # Add edges from new node to all nodes not yet in MST
        for j in node_set - in_mst:
            dist = manhattan_distance(nodes[to_idx], nodes[j])
            heapq.heappush(heap, (dist, to_idx, j))

    return mst_edges


def mst_traversal_order(mst_edges: list, start: tuple, all_nodes: list) -> list:
    """Generate a DFS traversal order of the MST starting from a given node.

    Converts MST edges into an adjacency list and performs DFS to determine
    the order in which nodes should be visited.

    Args:
        mst_edges: List of (weight, node1, node2) from compute_mst.
        start: (row, col) starting position for traversal.
        all_nodes: All nodes in the MST.

    Returns:
        List of (row, col) positions in DFS visitation order.
    """
    if not mst_edges:
        return [start] if start in all_nodes else list(all_nodes)

    # Build adjacency list
    adj = defaultdict(list)
    for weight, n1, n2 in mst_edges:
        adj[n1].append((weight, n2))
        adj[n2].append((weight, n1))

    # DFS traversal
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        order.append(node)
        # Sort neighbors by weight for deterministic order
        for weight, neighbor in sorted(adj[node]):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)

    # Add any disconnected nodes
    for node in all_nodes:
        if node not in visited:
            order.append(node)

    return order


class MSTAgent(MemoryAgent):
    """Phase 2: Agent that follows MST traversal to collect resources efficiently.

    Computes the Minimum Spanning Tree over all resource positions and the
    goal, then follows the MST DFS traversal order. Uses BFS grid pathfinding
    to navigate between MST nodes.

    Additional Attributes:
        target_nodes: List of nodes to visit (resources + goal).
        mst_edges: Computed MST edges.
        visit_order: DFS traversal order of MST nodes.
        current_target_idx: Index into visit_order for current target.
        path_to_target: BFS path to current target node.
    """

    def __init__(self, env, start_pos: tuple = None):
        """Initialize the MST agent and compute traversal plan.

        Args:
            env: GridEnvironment instance.
            start_pos: Optional starting position.
        """
        super().__init__(env, start_pos)
        self.target_nodes = []
        self.mst_edges = []
        self.visit_order = []
        self.current_target_idx = 0
        self.path_to_target = []
        self._plan_route()

    def _plan_route(self):
        """Compute MST and determine visitation order."""
        # Collect target nodes: resources + goal
        self.target_nodes = list(self.env.resources.keys())
        if self.env.goal_pos and self.env.goal_pos not in self.target_nodes:
            self.target_nodes.append(self.env.goal_pos)

        if not self.target_nodes:
            return

        # Include start position as a node for MST computation
        all_nodes = [self.position] + self.target_nodes
        # Remove duplicates while preserving order
        seen = set()
        unique_nodes = []
        for n in all_nodes:
            if n not in seen:
                seen.add(n)
                unique_nodes.append(n)

        self.mst_edges = compute_mst(unique_nodes)
        self.visit_order = mst_traversal_order(
            self.mst_edges, self.position, unique_nodes)

        # Skip the start position in visit order (we're already there)
        if self.visit_order and self.visit_order[0] == self.position:
            self.visit_order = self.visit_order[1:]

        self.current_target_idx = 0
        self._compute_path_to_current_target()

    def _compute_path_to_current_target(self):
        """Compute BFS path from current position to the current target node."""
        if self.current_target_idx < len(self.visit_order):
            target = self.visit_order[self.current_target_idx]
            self.path_to_target = bfs_path(
                self.env.grid, self.position, target,
                self.env.rows, self.env.cols)
        else:
            self.path_to_target = []

    def choose_action(self, percept: dict = None) -> str:
        """Follow MST traversal path to visit all target nodes.

        Navigates toward the current target in the MST visit order.
        When a target is reached, advances to the next one.

        Args:
            percept: Perception dict (optional).

        Returns:
            Direction string for the next step.
        """
        # Check if current target is reached
        while (self.current_target_idx < len(self.visit_order) and
               self.position == self.visit_order[self.current_target_idx]):
            self.current_target_idx += 1
            self._compute_path_to_current_target()

        # If all targets visited, fall back to memory agent behavior
        if self.current_target_idx >= len(self.visit_order):
            return super().choose_action(percept)

        # Navigate along BFS path to current target
        if self.path_to_target and len(self.path_to_target) > 1:
            # Find our position in the path
            if self.position in self.path_to_target:
                idx = self.path_to_target.index(self.position)
                if idx + 1 < len(self.path_to_target):
                    next_pos = self.path_to_target[idx + 1]
                    d = direction_from_positions(self.position, next_pos)
                    if d:
                        return d

            # Recompute path if position not on current path
            self._compute_path_to_current_target()
            if self.path_to_target and len(self.path_to_target) > 1:
                next_pos = self.path_to_target[1]
                d = direction_from_positions(self.position, next_pos)
                if d:
                    return d

        # Fallback: use memory agent logic
        return super().choose_action(percept)

    def get_stats(self) -> dict:
        """Collect MST agent statistics.

        Returns:
            Dict with memory agent stats plus mst_nodes_visited and
            mst_total_nodes.
        """
        stats = super().get_stats()
        stats.update({
            'mst_nodes_visited': self.current_target_idx,
            'mst_total_nodes': len(self.visit_order),
        })
        return stats
