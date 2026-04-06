"""
agent.py - Agent classes for the autonomous navigation project.

Implements three tiers of agents with increasing capability:
  - ReactiveAgent (Phase 1): Random reactive navigation.
  - MemoryAgent (Phase 2): Memory-based exploration with candidate edges.
  - HierarchicalAgent (Phase 3): Hierarchical decision-making with energy
    management and stochastic action execution.
"""

import random
from config import CellType, DIRECTIONS, PERPENDICULAR_DIRECTIONS
from utils import manhattan_distance, bfs_path, direction_from_positions


class ReactiveAgent:
    """Phase 1: Simple reactive agent that moves randomly among valid directions.

    The agent senses adjacent cells and selects a random valid move each step.
    It tracks movement statistics for experimental analysis.

    Attributes:
        env: Reference to the GridEnvironment.
        position: Current (row, col) position.
        steps_taken: Total number of steps executed.
        resources_collected: Count of resources picked up.
        goal_reached: Whether the agent has reached the goal.
        action_history: List of (action, event) tuples.
        action_counts: Dict mapping direction to count of times chosen.
        visited_cells: Set of (row, col) positions visited.
    """

    def __init__(self, env, start_pos: tuple = None):
        """Initialize the reactive agent.

        Args:
            env: GridEnvironment instance.
            start_pos: Optional starting position. Uses env.agent_pos if None.
        """
        self.env = env
        self.position = start_pos if start_pos else env.agent_pos
        self.steps_taken = 0
        self.resources_collected = 0
        self.goal_reached = False
        self.action_history = []
        self.action_counts = {d: 0 for d in DIRECTIONS}
        self.visited_cells = {self.position}
        self.revisit_count = 0

    def sense(self) -> dict:
        """Perceive the adjacent cells around the agent.

        Returns:
            Dict mapping direction names to the CellType of the
            neighboring cell, or 'boundary' if out of bounds.
        """
        percept = {}
        r, c = self.position
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if self.env.is_within_bounds(nr, nc):
                percept[direction] = self.env.get_cell(nr, nc)
            else:
                percept[direction] = 'boundary'
        percept['current'] = self.env.get_cell(r, c)
        percept['position'] = self.position
        return percept

    def choose_action(self, percept: dict = None) -> str:
        """Select a random valid action.

        Args:
            percept: Perception dict (unused by reactive agent but
                     included for interface consistency).

        Returns:
            Direction string for the chosen action.
        """
        valid = self.env.get_valid_actions()
        if not valid:
            return random.choice(list(DIRECTIONS.keys()))
        return random.choice(valid)

    def act(self, action: str) -> str:
        """Execute the chosen action in the environment.

        Args:
            action: Direction string to move.

        Returns:
            Event string from the environment.
        """
        success, new_pos, event = self.env.move_agent(action)
        self.position = self.env.agent_pos
        self.steps_taken += 1
        self.action_counts[action] += 1
        self.action_history.append((action, event))
        self.visited_cells.add(self.position)

        if event == 'resource_collected':
            self.resources_collected += 1
        elif event == 'goal_reached':
            self.goal_reached = True

        return event

    def step(self) -> str:
        """Execute one full perception-action cycle.

        Returns:
            Event string from the action taken.
        """
        old_visited_count = len(self.visited_cells)
        percept = self.sense()
        action = self.choose_action(percept)
        event = self.act(action)
        if len(self.visited_cells) == old_visited_count:
            self.revisit_count += 1
        return event

    def get_stats(self) -> dict:
        """Collect performance statistics.

        Returns:
            Dict with keys: steps_taken, resources_collected, goal_reached,
            coverage, visited_count, action_counts.
        """
        non_wall = self.env.count_non_wall_cells()
        coverage = len(self.visited_cells) / non_wall if non_wall > 0 else 0.0
        revisit_ratio = (self.revisit_count / self.steps_taken
                         if self.steps_taken > 0 else 0.0)
        return {
            'steps_taken': self.steps_taken,
            'resources_collected': self.resources_collected,
            'goal_reached': self.goal_reached,
            'coverage': coverage,
            'visited_count': len(self.visited_cells),
            'action_counts': dict(self.action_counts),
            'revisit_count': self.revisit_count,
            'revisit_ratio': revisit_ratio,
        }


class MemoryAgent(ReactiveAgent):
    """Phase 2: Agent with memory for systematic exploration.

    Extends ReactiveAgent with a visited-cell memory and candidate edge
    frontier. Prefers exploring unvisited cells and can push obstacles.

    Additional Attributes:
        candidate_edges: Set of unvisited (row, col) adjacent to visited cells.
        memory: Dict mapping (row, col) to observed CellType.
        obstacles_pushed: Count of obstacles successfully pushed.
        revisit_count: Number of times the agent revisited an already-visited cell.
    """

    def __init__(self, env, start_pos: tuple = None):
        """Initialize the memory agent.

        Args:
            env: GridEnvironment instance.
            start_pos: Optional starting position.
        """
        super().__init__(env, start_pos)
        self.candidate_edges = set()
        self.memory = {self.position: self.env.get_cell(*self.position)}
        self.obstacles_pushed = 0
        self._update_candidates()

    def _update_candidates(self):
        """Update the candidate edge frontier based on current position.

        Adds unvisited, non-wall neighbors of visited cells to the
        candidate set, and removes the current cell if present.
        """
        r, c = self.position
        self.candidate_edges.discard((r, c))
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if self.env.is_within_bounds(nr, nc):
                cell = self.env.get_cell(nr, nc)
                self.memory[(nr, nc)] = cell
                if (nr, nc) not in self.visited_cells and cell != CellType.WALL:
                    self.candidate_edges.add((nr, nc))

    def choose_action(self, percept: dict = None) -> str:
        """Select action preferring unvisited neighbors.

        Decision priority:
        1. Move to an adjacent unvisited, non-wall cell.
        2. If all neighbors visited, navigate toward the nearest candidate edge.
        3. If no candidates remain, fall back to random movement.

        Args:
            percept: Perception dict (optional).

        Returns:
            Direction string for the chosen action.
        """
        r, c = self.position

        # Priority 1: unvisited adjacent cells
        unvisited_dirs = []
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if self.env.is_within_bounds(nr, nc):
                cell = self.env.get_cell(nr, nc)
                if (nr, nc) not in self.visited_cells and cell not in (CellType.WALL,):
                    unvisited_dirs.append(direction)

        if unvisited_dirs:
            return random.choice(unvisited_dirs)

        # Priority 2: path to nearest candidate edge
        if self.candidate_edges:
            nearest = min(self.candidate_edges,
                          key=lambda p: manhattan_distance(self.position, p))
            path = bfs_path(self.env.grid, self.position, nearest,
                            self.env.rows, self.env.cols)
            if path and len(path) > 1:
                next_pos = path[1]
                d = direction_from_positions(self.position, next_pos)
                if d:
                    return d

        # Priority 3: random valid action
        valid = self.env.get_valid_actions()
        if valid:
            return random.choice(valid)
        return random.choice(list(DIRECTIONS.keys()))

    def act(self, action: str) -> str:
        """Execute action and update memory state.

        Args:
            action: Direction string.

        Returns:
            Event string from the environment.
        """
        event = super().act(action)

        if event == 'obstacle_pushed':
            self.obstacles_pushed += 1

        self._update_candidates()
        return event

    def get_stats(self) -> dict:
        """Collect extended performance statistics.

        Returns:
            Dict with base stats plus obstacles_pushed, revisit_count,
            revisit_ratio, candidate_edges_remaining.
        """
        stats = super().get_stats()
        revisit_ratio = (self.revisit_count / self.steps_taken
                         if self.steps_taken > 0 else 0.0)
        stats.update({
            'obstacles_pushed': self.obstacles_pushed,
            'revisit_count': self.revisit_count,
            'revisit_ratio': revisit_ratio,
            'candidate_edges_remaining': len(self.candidate_edges),
        })
        return stats


class HierarchicalAgent(MemoryAgent):
    """Phase 3: Agent with hierarchical decision-making and energy management.

    Uses a controller object for decision-making. Supports stochastic action
    execution (actions may deviate with a given noise probability) and energy
    constraints (each step costs energy, collecting resources restores some).

    Additional Attributes:
        energy: Current energy level.
        max_energy: Maximum energy capacity.
        controller: Controller object with a decide() method.
        noise: Probability of action deviation (stochastic execution).
        total_reward: Accumulated reward score.
        energy_history: List of energy values over time.
    """

    def __init__(self, env, start_pos: tuple = None, energy: int = 50,
                 controller=None, noise: float = 0.0):
        """Initialize the hierarchical agent.

        Args:
            env: GridEnvironment instance.
            start_pos: Optional starting position.
            energy: Starting energy budget.
            controller: Controller object implementing decide(percept, agent).
            noise: Action noise probability (0.0 to 1.0).
        """
        super().__init__(env, start_pos)
        self.energy = energy
        self.max_energy = energy
        self.controller = controller
        self.noise = noise
        self.total_reward = 0
        self.energy_history = [energy]

    def sense(self) -> dict:
        """Enhanced perception including energy and goal information.

        Returns:
            Dict with base perception plus energy, goal_distance,
            resources_nearby, and unvisited_neighbors count.
        """
        percept = super().sense()
        percept['energy'] = self.energy
        percept['max_energy'] = self.max_energy
        if self.env.goal_pos:
            percept['goal_distance'] = manhattan_distance(
                self.position, self.env.goal_pos)
        else:
            percept['goal_distance'] = float('inf')
        percept['resources_nearby'] = self._count_nearby_resources()
        percept['unvisited_neighbors'] = self._count_unvisited_neighbors()
        percept['resources_remaining'] = len(self.env.resources)
        return percept

    def _count_nearby_resources(self) -> int:
        """Count resources in cells adjacent to the agent.

        Returns:
            Number of adjacent resource cells.
        """
        count = 0
        r, c = self.position
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if (self.env.is_within_bounds(nr, nc) and
                    self.env.get_cell(nr, nc) == CellType.RESOURCE):
                count += 1
        return count

    def _count_unvisited_neighbors(self) -> int:
        """Count unvisited non-wall neighbors.

        Returns:
            Number of adjacent unvisited traversable cells.
        """
        count = 0
        r, c = self.position
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if (self.env.is_within_bounds(nr, nc) and
                    (nr, nc) not in self.visited_cells and
                    self.env.get_cell(nr, nc) != CellType.WALL):
                count += 1
        return count

    def choose_action(self, percept: dict = None) -> str:
        """Delegate action selection to the controller.

        If no controller is set, falls back to MemoryAgent behavior.

        Args:
            percept: Enhanced perception dict.

        Returns:
            Direction string.
        """
        if self.controller:
            if percept is None:
                percept = self.sense()
            return self.controller.decide(percept, self)
        return super().choose_action(percept)

    def _apply_noise(self, intended_action: str) -> str:
        """Apply stochastic noise to the intended action.

        With probability self.noise, the action deviates to a random
        perpendicular direction.

        Args:
            intended_action: The action the agent intended to take.

        Returns:
            The actual action after noise application.
        """
        if self.noise > 0 and random.random() < self.noise:
            perp = PERPENDICULAR_DIRECTIONS.get(intended_action, [])
            if perp:
                return random.choice(perp)
        return intended_action

    def act(self, action: str) -> str:
        """Execute action with noise and energy management.

        Applies stochastic noise, deducts energy, and handles resource
        energy restoration.

        Args:
            action: Intended direction string.

        Returns:
            Event string from the environment.
        """
        actual_action = self._apply_noise(action)
        event = super().act(actual_action)

        self.energy -= 1
        if event == 'resource_collected':
            from config import ENERGY_RESTORE_ON_RESOURCE, RESOURCE_REWARD
            self.energy = min(self.energy + ENERGY_RESTORE_ON_RESOURCE,
                              self.max_energy)
            self.total_reward += RESOURCE_REWARD

        if event == 'goal_reached':
            self.total_reward += 50

        self.energy_history.append(self.energy)
        return event

    def is_alive(self) -> bool:
        """Check if the agent has remaining energy.

        Returns:
            True if energy > 0.
        """
        return self.energy > 0

    def step(self) -> str:
        """Execute one perception-action cycle with energy check.

        Returns:
            Event string, or 'dead' if energy is depleted.
        """
        if not self.is_alive():
            return 'dead'

        old_visited_count = len(self.visited_cells)
        percept = self.sense()
        action = self.choose_action(percept)
        event = self.act(action)
        if len(self.visited_cells) == old_visited_count:
            self.revisit_count += 1
        return event

    def get_stats(self) -> dict:
        """Collect full performance statistics.

        Returns:
            Dict with all stats plus energy_remaining, total_reward,
            energy_efficiency, survival status.
        """
        stats = super().get_stats()
        stats.update({
            'energy_remaining': self.energy,
            'total_reward': self.total_reward,
            'survived': self.energy > 0 or self.goal_reached,
        })
        return stats
