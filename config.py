"""
config.py - Configuration constants, cell types, and predefined grid layouts.

This module defines the core data types and constants used throughout the
Intelligent Agent Navigation project, including cell type enumerations,
grid layouts for 5x5 and 6x4 environments, direction mappings, and
default experiment parameters.
"""

from enum import Enum


class CellType(Enum):
    """Enumeration of possible cell contents in the grid environment.

    Each cell type has a single-character string value used for
    terminal rendering of the grid.
    """
    EMPTY = '.'
    AGENT = 'A'
    WALL = '#'
    OBSTACLE = 'O'
    RESOURCE = 'R'
    GOAL = 'G'


# Direction vectors: maps direction name to (row_delta, col_delta)
DIRECTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

# Reverse direction mapping
OPPOSITE_DIRECTIONS = {
    'up': 'down',
    'down': 'up',
    'left': 'right',
    'right': 'left',
}

# Perpendicular directions for stochastic action deviation
PERPENDICULAR_DIRECTIONS = {
    'up': ['left', 'right'],
    'down': ['left', 'right'],
    'left': ['up', 'down'],
    'right': ['up', 'down'],
}

# --- Predefined Grid Layouts ---
# Legend: '.' = empty, '#' = wall, 'R' = resource, 'G' = goal, 'A' = agent start
# Obstacles ('O') are placed dynamically in Phase 2+

GRID_5x5 = [
    ['A', '.', '#', '.', 'R'],
    ['.', '.', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '.', '#', 'R'],
    ['R', '.', '#', '.', 'G'],
]

GRID_6x4 = [
    ['A', '.', '.', 'R'],
    ['.', '#', '.', '.'],
    ['.', '.', '.', '#'],
    ['#', '.', 'R', '.'],
    ['.', '.', '#', '.'],
    ['R', '.', '.', 'G'],
]

# --- Experiment Parameters ---
DEFAULT_EPISODES = 50
MAX_STEPS_PHASE1 = 200
MAX_STEPS_PHASE2 = 300
MAX_STEPS_PHASE3 = 500
DEFAULT_ENERGY = 50
ENERGY_RESTORE_ON_RESOURCE = 10
RESOURCE_REWARD = 10
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
ENERGY_BUDGETS = [30, 50, 75, 100]
OBSTACLE_COUNTS = [0, 2, 4]
ENERGY_THRESHOLD_RATIO = 0.25  # go to goal when energy < max * this ratio
RANDOM_SEED = 42
