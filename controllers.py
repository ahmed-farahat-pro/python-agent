"""
controllers.py - Decision-making controllers for Phase 3 agents.

Implements four controller strategies with increasing sophistication:
  - ExplorerController: Pure random exploration (baseline).
  - ReactiveController: Random movement avoiding walls.
  - DeliberativeController: Greedy goal-directed navigation.
  - HierarchicalController: Two-level hierarchy with sub-goal selection
    and energy-aware strategy switching.
"""

import random
from config import DIRECTIONS, CellType, ENERGY_THRESHOLD_RATIO
from utils import manhattan_distance, bfs_path, direction_from_positions


class ExplorerController:
    """Pure random exploration controller (baseline).

    Selects uniformly at random from all four directions, including
    potentially invalid ones. This means the agent may waste moves
    bumping into walls. Serves as the lowest-performing baseline.
    """

    def decide(self, percept: dict, agent) -> str:
        """Choose a random direction (may include invalid moves).

        Args:
            percept: Agent's perception dict.
            agent: Reference to the agent.

        Returns:
            Random direction string (from all 4 directions).
        """
        return random.choice(list(DIRECTIONS.keys()))


class ReactiveController:
    """Reactive controller that avoids walls but otherwise moves randomly.

    Filters out directions leading to walls or boundaries, then picks
    randomly from the remaining valid options.
    """

    def decide(self, percept: dict, agent) -> str:
        """Choose a random valid direction (avoiding walls).

        Args:
            percept: Agent's perception dict.
            agent: Reference to the agent.

        Returns:
            Direction string for a valid move.
        """
        valid = agent.env.get_valid_actions()
        if valid:
            return random.choice(valid)
        return random.choice(list(DIRECTIONS.keys()))


class DeliberativeController:
    """Greedy goal-directed controller with stuck detection.

    Moves toward the goal by choosing the direction that minimizes
    Manhattan distance. Detects oscillation (getting stuck between
    two positions) and injects a random move to escape.

    Attributes:
        recent_positions: List of recent positions for stuck detection.
    """

    def __init__(self):
        """Initialize the deliberative controller."""
        self.recent_positions = []

    def _is_stuck(self, current_pos: tuple) -> bool:
        """Detect if the agent is oscillating between positions.

        Args:
            current_pos: Current agent position.

        Returns:
            True if agent appears stuck in a cycle.
        """
        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > 6:
            self.recent_positions.pop(0)
        if len(self.recent_positions) >= 4:
            # Check for A-B-A-B oscillation pattern
            if (self.recent_positions[-1] == self.recent_positions[-3] and
                    self.recent_positions[-2] == self.recent_positions[-4]):
                return True
        return False

    def decide(self, percept: dict, agent) -> str:
        """Choose the direction that moves closest to the goal.

        Uses BFS pathfinding for reliable navigation around walls.
        Falls back to greedy direction selection, and random movement
        when stuck in an oscillation cycle.

        Args:
            percept: Agent's perception dict.
            agent: Reference to the agent.

        Returns:
            Direction string toward the goal.
        """
        goal = agent.env.goal_pos
        if goal is None:
            return random.choice(list(DIRECTIONS.keys()))

        valid = agent.env.get_valid_actions()
        if not valid:
            return random.choice(list(DIRECTIONS.keys()))

        # Escape stuck cycles with a random move
        if self._is_stuck(agent.position):
            self.recent_positions.clear()
            return random.choice(valid)

        # Try BFS pathfinding to goal
        path = bfs_path(agent.env.grid, agent.position, goal,
                        agent.env.rows, agent.env.cols)
        if path and len(path) > 1:
            next_pos = path[1]
            d = direction_from_positions(agent.position, next_pos)
            if d and d in valid:
                return d

        # Fallback: greedy direction selection
        scored = []
        for action in valid:
            dr, dc = DIRECTIONS[action]
            nr, nc = agent.position[0] + dr, agent.position[1] + dc
            if agent.env.boundary_mode == 'wrap':
                nr = nr % agent.env.rows
                nc = nc % agent.env.cols
            dist = manhattan_distance((nr, nc), goal)
            scored.append((dist, action))
        scored.sort()

        return scored[0][1]


class HierarchicalController:
    """Two-level hierarchical controller with energy-aware strategy switching.

    High-level decision: selects a sub-goal based on the agent's state.
    Low-level decision: navigates toward the chosen sub-goal.

    Strategy priorities:
    1. SURVIVE: If energy is low, head directly to the goal.
    2. GATHER: If resources are nearby, collect the nearest one.
    3. EXPLORE: Move toward unvisited cells to maximize coverage.

    Attributes:
        energy_threshold_ratio: Fraction of max energy below which
            the agent switches to survival mode.
        current_subgoal: Current target position.
        current_strategy: Name of the active strategy.
    """

    def __init__(self, energy_threshold_ratio: float = ENERGY_THRESHOLD_RATIO):
        """Initialize the hierarchical controller.

        Args:
            energy_threshold_ratio: Energy fraction threshold for survival mode.
        """
        self.energy_threshold_ratio = energy_threshold_ratio
        self.current_subgoal = None
        self.current_strategy = 'explore'

    def _select_strategy(self, percept: dict, agent) -> tuple:
        """High-level decision: choose strategy and sub-goal.

        Args:
            percept: Agent's perception dict.
            agent: Reference to the agent.

        Returns:
            Tuple of (strategy_name, sub_goal_position).
        """
        energy = percept['energy']
        max_energy = percept['max_energy']
        threshold = max_energy * self.energy_threshold_ratio

        # Strategy 1: SURVIVE - head to goal when energy is low
        if energy <= threshold and agent.env.goal_pos:
            return ('survive', agent.env.goal_pos)

        # Strategy 2: GATHER - collect nearby resources
        resource_positions = agent.env.get_resource_positions()
        if resource_positions:
            nearest_resource = min(resource_positions,
                                   key=lambda p: manhattan_distance(agent.position, p))
            dist = manhattan_distance(agent.position, nearest_resource)
            # Only gather if we have enough energy to reach resource and then goal
            if agent.env.goal_pos:
                dist_to_goal = manhattan_distance(nearest_resource, agent.env.goal_pos)
                if energy > dist + dist_to_goal + 5:  # safety margin
                    return ('gather', nearest_resource)
            else:
                return ('gather', nearest_resource)

        # Strategy 3: EXPLORE - move to unvisited areas
        if agent.candidate_edges:
            nearest_candidate = min(agent.candidate_edges,
                                    key=lambda p: manhattan_distance(agent.position, p))
            return ('explore', nearest_candidate)

        # Default: head to goal
        if agent.env.goal_pos:
            return ('survive', agent.env.goal_pos)

        return ('explore', None)

    def _navigate_toward(self, target: tuple, agent) -> str:
        """Low-level decision: navigate toward a target position.

        Uses BFS pathfinding to find the shortest path, then returns
        the direction of the first step. Falls back to greedy direction
        if BFS fails.

        Args:
            target: (row, col) position to navigate toward.
            agent: Reference to the agent.

        Returns:
            Direction string for the next step.
        """
        if target is None:
            valid = agent.env.get_valid_actions()
            return random.choice(valid) if valid else random.choice(list(DIRECTIONS.keys()))

        # Try BFS path first
        path = bfs_path(agent.env.grid, agent.position, target,
                        agent.env.rows, agent.env.cols)
        if path and len(path) > 1:
            next_pos = path[1]
            d = direction_from_positions(agent.position, next_pos)
            if d:
                return d

        # Fallback: greedy direction
        valid = agent.env.get_valid_actions()
        if not valid:
            return random.choice(list(DIRECTIONS.keys()))

        best_action = None
        best_dist = float('inf')
        for action in valid:
            dr, dc = DIRECTIONS[action]
            nr = agent.position[0] + dr
            nc = agent.position[1] + dc
            dist = manhattan_distance((nr, nc), target)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action if best_action else random.choice(valid)

    def decide(self, percept: dict, agent) -> str:
        """Make a hierarchical decision: strategy selection then navigation.

        Args:
            percept: Agent's perception dict.
            agent: Reference to the agent.

        Returns:
            Direction string for the chosen action.
        """
        strategy, subgoal = self._select_strategy(percept, agent)
        self.current_strategy = strategy
        self.current_subgoal = subgoal
        return self._navigate_toward(subgoal, agent)
