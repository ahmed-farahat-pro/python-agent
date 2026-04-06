"""
environment.py - Grid environment for the autonomous agent navigation project.

Implements the GridEnvironment class representing a discrete 2D world where
agents navigate, collect resources, avoid obstacles, and reach goals.
Supports three boundary modes: wall, bouncy, and wrap-around.
"""

import copy
import random
from config import CellType, DIRECTIONS, OPPOSITE_DIRECTIONS


class GridEnvironment:
    """Discrete grid environment for agent navigation.

    The environment is a 2D grid where each cell contains a CellType value.
    The agent interacts with the environment through movement actions, and
    the environment handles boundary conditions, obstacle interactions, and
    resource collection.

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        boundary_mode: How boundaries are handled ('wall', 'bouncy', 'wrap').
        grid: 2D list of CellType values.
        agent_pos: Current (row, col) position of the agent.
        goal_pos: (row, col) position of the goal cell.
        resources: Dict mapping (row, col) to resource presence.
        obstacles: Set of (row, col) positions with movable obstacles.
        total_resources: Total number of resources placed initially.
    """

    def __init__(self, rows: int, cols: int, layout: list = None,
                 boundary_mode: str = 'wall'):
        """Initialize the grid environment.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            layout: Optional 2D list of character strings matching CellType values.
                    If None, creates an empty grid.
            boundary_mode: Boundary handling mode - 'wall', 'bouncy', or 'wrap'.
        """
        self.rows = rows
        self.cols = cols
        self.boundary_mode = boundary_mode
        self.agent_pos = None
        self.goal_pos = None
        self.resources = {}
        self.obstacles = set()

        if layout is not None:
            self._build_from_layout(layout)
        else:
            self.grid = [[CellType.EMPTY for _ in range(cols)] for _ in range(rows)]

        self.total_resources = len(self.resources)
        self._initial_state = self._save_state()

    def _build_from_layout(self, layout: list):
        """Construct the grid from a character layout.

        Args:
            layout: 2D list of single-character strings.
        """
        char_to_cell = {ct.value: ct for ct in CellType}
        self.grid = []
        for r, row in enumerate(layout):
            grid_row = []
            for c, char in enumerate(row):
                cell = char_to_cell.get(char, CellType.EMPTY)
                if cell == CellType.AGENT:
                    self.agent_pos = (r, c)
                    grid_row.append(CellType.AGENT)
                elif cell == CellType.GOAL:
                    self.goal_pos = (r, c)
                    grid_row.append(CellType.GOAL)
                elif cell == CellType.RESOURCE:
                    self.resources[(r, c)] = True
                    grid_row.append(CellType.RESOURCE)
                elif cell == CellType.OBSTACLE:
                    self.obstacles.add((r, c))
                    grid_row.append(CellType.OBSTACLE)
                else:
                    grid_row.append(cell)
            self.grid.append(grid_row)

    def _save_state(self) -> dict:
        """Save the current environment state for reset.

        Returns:
            Dictionary containing deep copies of all mutable state.
        """
        return {
            'grid': copy.deepcopy(self.grid),
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'resources': dict(self.resources),
            'obstacles': set(self.obstacles),
            'total_resources': self.total_resources,
        }

    def reset(self):
        """Restore the environment to its initial state."""
        state = self._initial_state
        self.grid = copy.deepcopy(state['grid'])
        self.agent_pos = state['agent_pos']
        self.goal_pos = state['goal_pos']
        self.resources = dict(state['resources'])
        self.obstacles = set(state['obstacles'])
        self.total_resources = state['total_resources']

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Check if a position is within the grid boundaries.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if position is within bounds.
        """
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_cell(self, row: int, col: int) -> CellType:
        """Get the cell type at a given position.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            CellType at the specified position.
        """
        return self.grid[row][col]

    def set_cell(self, row: int, col: int, cell_type: CellType):
        """Set the cell type at a given position.

        Args:
            row: Row index.
            col: Column index.
            cell_type: The CellType to set.
        """
        self.grid[row][col] = cell_type

    def get_neighbors(self, row: int, col: int) -> list:
        """Get valid neighboring cells with their directions.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            List of (neighbor_row, neighbor_col, direction) tuples for
            in-bounds neighbors that are not walls.
        """
        neighbors = []
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = row + dr, col + dc
            if self.is_within_bounds(nr, nc) and self.grid[nr][nc] != CellType.WALL:
                neighbors.append((nr, nc, direction))
        return neighbors

    def get_valid_actions(self, row: int = None, col: int = None) -> list:
        """Get list of valid movement directions from a position.

        Args:
            row: Row index (defaults to agent position).
            col: Column index (defaults to agent position).

        Returns:
            List of direction strings the agent can move in.
        """
        if row is None:
            row, col = self.agent_pos
        actions = []
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = row + dr, col + dc
            # For wall mode, out-of-bounds is invalid
            if self.boundary_mode == 'wall':
                if not self.is_within_bounds(nr, nc):
                    continue
                if self.grid[nr][nc] == CellType.WALL:
                    continue
                actions.append(direction)
            elif self.boundary_mode == 'bouncy':
                # Bouncy: always valid (will bounce back if out of bounds)
                actions.append(direction)
            elif self.boundary_mode == 'wrap':
                # Wrap: always valid (wraps around)
                wr, wc = nr % self.rows, nc % self.cols
                if self.grid[wr][wc] != CellType.WALL:
                    actions.append(direction)
        return actions

    def move_agent(self, direction: str) -> tuple:
        """Attempt to move the agent in the specified direction.

        Handles boundary conditions (wall/bouncy/wrap), obstacle pushing,
        resource collection, and goal detection.

        Args:
            direction: Movement direction ('up', 'down', 'left', 'right').

        Returns:
            Tuple of (success: bool, new_pos: tuple, event: str).
            Events: 'moved', 'blocked', 'bounced', 'wrapped',
                    'resource_collected', 'obstacle_pushed', 'goal_reached'.
        """
        if self.agent_pos is None:
            return (False, None, 'no_agent')

        ar, ac = self.agent_pos
        dr, dc = DIRECTIONS[direction]
        nr, nc = ar + dr, ac + dc

        # --- Apply boundary handling ---
        if not self.is_within_bounds(nr, nc):
            if self.boundary_mode == 'wall':
                return (False, self.agent_pos, 'blocked')
            elif self.boundary_mode == 'bouncy':
                # Reverse direction
                opp = OPPOSITE_DIRECTIONS[direction]
                bdr, bdc = DIRECTIONS[opp]
                br, bc = ar + bdr, ac + bdc
                if self.is_within_bounds(br, bc) and self.grid[br][bc] not in (CellType.WALL, CellType.OBSTACLE):
                    self._move_agent_to(br, bc)
                    return (True, self.agent_pos, 'bounced')
                else:
                    return (False, self.agent_pos, 'blocked')
            elif self.boundary_mode == 'wrap':
                nr, nc = nr % self.rows, nc % self.cols
                # Fall through to normal movement logic with wrapped coordinates

        # --- Target cell handling ---
        target_cell = self.grid[nr][nc]

        if target_cell == CellType.WALL:
            if self.boundary_mode == 'bouncy':
                opp = OPPOSITE_DIRECTIONS[direction]
                bdr, bdc = DIRECTIONS[opp]
                br, bc = ar + bdr, ac + bdc
                if self.is_within_bounds(br, bc) and self.grid[br][bc] not in (CellType.WALL, CellType.OBSTACLE):
                    self._move_agent_to(br, bc)
                    return (True, self.agent_pos, 'bounced')
            return (False, self.agent_pos, 'blocked')

        if target_cell == CellType.OBSTACLE:
            # Try to push the obstacle
            pushed = self.try_push_obstacle((nr, nc), direction)
            if pushed:
                self._move_agent_to(nr, nc)
                return (True, self.agent_pos, 'obstacle_pushed')
            else:
                return (False, self.agent_pos, 'blocked')

        if target_cell == CellType.RESOURCE:
            self._move_agent_to(nr, nc)
            if (nr, nc) in self.resources:
                del self.resources[(nr, nc)]
            event = 'resource_collected'
            return (True, self.agent_pos, event)

        if target_cell == CellType.GOAL:
            self._move_agent_to(nr, nc)
            return (True, self.agent_pos, 'goal_reached')

        # Empty cell
        if target_cell == CellType.EMPTY or target_cell == CellType.AGENT:
            self._move_agent_to(nr, nc)
            event = 'wrapped' if self.boundary_mode == 'wrap' and not self.is_within_bounds(ar + dr, ac + dc) else 'moved'
            return (True, self.agent_pos, event)

        return (False, self.agent_pos, 'blocked')

    def _move_agent_to(self, new_row: int, new_col: int):
        """Move the agent to a new position, updating the grid.

        Args:
            new_row: Target row.
            new_col: Target column.
        """
        ar, ac = self.agent_pos
        # Restore old cell (empty, since agent was there)
        self.grid[ar][ac] = CellType.EMPTY
        # Place agent in new cell
        self.grid[new_row][new_col] = CellType.AGENT
        self.agent_pos = (new_row, new_col)

    def try_push_obstacle(self, obs_pos: tuple, direction: str) -> bool:
        """Attempt to push a movable obstacle in the given direction.

        The obstacle can be pushed only if the cell behind it (in the push
        direction) is empty and within bounds.

        Args:
            obs_pos: (row, col) position of the obstacle.
            direction: Direction of the push.

        Returns:
            True if the obstacle was successfully pushed.
        """
        dr, dc = DIRECTIONS[direction]
        behind_r = obs_pos[0] + dr
        behind_c = obs_pos[1] + dc

        if not self.is_within_bounds(behind_r, behind_c):
            return False

        behind_cell = self.grid[behind_r][behind_c]
        if behind_cell != CellType.EMPTY:
            return False

        # Push the obstacle
        self.grid[obs_pos[0]][obs_pos[1]] = CellType.EMPTY
        self.grid[behind_r][behind_c] = CellType.OBSTACLE
        self.obstacles.discard(obs_pos)
        self.obstacles.add((behind_r, behind_c))
        return True

    def place_resources_randomly(self, count: int):
        """Place resources in random empty cells.

        Args:
            count: Number of resources to place.
        """
        empty_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.grid[r][c] == CellType.EMPTY
        ]
        random.shuffle(empty_cells)
        for i in range(min(count, len(empty_cells))):
            r, c = empty_cells[i]
            self.grid[r][c] = CellType.RESOURCE
            self.resources[(r, c)] = True
        self.total_resources = len(self.resources)
        self._initial_state = self._save_state()

    def place_obstacles_randomly(self, count: int):
        """Place movable obstacles in random empty cells.

        Args:
            count: Number of obstacles to place.
        """
        empty_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.grid[r][c] == CellType.EMPTY
        ]
        random.shuffle(empty_cells)
        for i in range(min(count, len(empty_cells))):
            r, c = empty_cells[i]
            self.grid[r][c] = CellType.OBSTACLE
            self.obstacles.add((r, c))
        self._initial_state = self._save_state()

    def count_non_wall_cells(self) -> int:
        """Count the number of non-wall cells in the grid.

        Returns:
            Integer count of traversable cells.
        """
        count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != CellType.WALL:
                    count += 1
        return count

    def get_resource_positions(self) -> list:
        """Get current positions of all remaining resources.

        Returns:
            List of (row, col) tuples.
        """
        return list(self.resources.keys())

    def render(self) -> str:
        """Render the grid as a formatted text string.

        Returns:
            String representation of the grid with row/column indices.
        """
        header = '    ' + ' '.join(str(c) for c in range(self.cols))
        separator = '    ' + '-' * (self.cols * 2 - 1)
        lines = [header, separator]
        for r in range(self.rows):
            row_str = f'{r} | '
            row_str += ' '.join(self.grid[r][c].value for c in range(self.cols))
            lines.append(row_str)
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.render()
