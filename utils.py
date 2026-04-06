"""
utils.py - Shared utility functions for the navigation project.

Provides distance calculations, BFS pathfinding on grids, and
direction helper functions used across multiple modules.
"""

from collections import deque
from config import DIRECTIONS, OPPOSITE_DIRECTIONS, CellType


def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculate Manhattan distance between two (row, col) positions.

    Args:
        pos1: Tuple (row, col) for first position.
        pos2: Tuple (row, col) for second position.

    Returns:
        Integer Manhattan distance.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def opposite_direction(direction: str) -> str:
    """Return the opposite of a given direction.

    Args:
        direction: One of 'up', 'down', 'left', 'right'.

    Returns:
        The opposite direction string.
    """
    return OPPOSITE_DIRECTIONS[direction]


def bfs_path(grid: list, start: tuple, goal: tuple, rows: int, cols: int,
             passable: set = None) -> list:
    """Find the shortest path from start to goal on a grid using BFS.

    Args:
        grid: 2D list of CellType values representing the environment.
        start: (row, col) starting position.
        goal: (row, col) target position.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        passable: Set of CellType values the agent can traverse.
                  Defaults to {EMPTY, RESOURCE, GOAL, AGENT}.

    Returns:
        List of (row, col) positions from start to goal (inclusive),
        or an empty list if no path exists.
    """
    if passable is None:
        passable = {CellType.EMPTY, CellType.RESOURCE, CellType.GOAL, CellType.AGENT}

    if start == goal:
        return [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr][nc] in passable:
                    new_path = path + [(nr, nc)]
                    if (nr, nc) == goal:
                        return new_path
                    visited.add((nr, nc))
                    queue.append(((nr, nc), new_path))

    return []


def direction_from_positions(current: tuple, target: tuple) -> str:
    """Determine the direction to move from current to an adjacent target cell.

    Args:
        current: (row, col) current position.
        target: (row, col) adjacent target position.

    Returns:
        Direction string ('up', 'down', 'left', 'right'), or None
        if target is not adjacent.
    """
    dr = target[0] - current[0]
    dc = target[1] - current[1]
    for direction, (row_d, col_d) in DIRECTIONS.items():
        if dr == row_d and dc == col_d:
            return direction
    return None
