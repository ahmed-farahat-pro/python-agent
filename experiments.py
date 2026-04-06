"""
experiments.py - Experiment runner and metrics collection.

Provides the ExperimentRunner class that executes all seven experiments
across the three project phases. Collects per-episode metrics and
computes summary statistics (mean, standard deviation) for analysis.
"""

import random
import copy
from config import (
    GRID_5x5, GRID_6x4, DEFAULT_EPISODES, MAX_STEPS_PHASE1,
    MAX_STEPS_PHASE2, MAX_STEPS_PHASE3, DEFAULT_ENERGY, NOISE_LEVELS,
    ENERGY_BUDGETS, OBSTACLE_COUNTS, RANDOM_SEED,
)
from environment import GridEnvironment
from agent import ReactiveAgent, MemoryAgent, HierarchicalAgent
from mst import MSTAgent
from controllers import (
    ExplorerController, ReactiveController,
    DeliberativeController, HierarchicalController,
)


def _compute_mean(values: list) -> float:
    """Compute arithmetic mean of a list of numbers."""
    return sum(values) / len(values) if values else 0.0


def _compute_std(values: list) -> float:
    """Compute standard deviation of a list of numbers."""
    if len(values) < 2:
        return 0.0
    mean = _compute_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _summarize(results: list, key: str) -> dict:
    """Compute mean and std for a specific metric across episodes.

    Args:
        results: List of stat dicts from agent.get_stats().
        key: Metric key to summarize.

    Returns:
        Dict with 'mean' and 'std' values.
    """
    values = [r[key] for r in results if key in r]
    return {'mean': _compute_mean(values), 'std': _compute_std(values)}


class ExperimentRunner:
    """Runs all experiments and collects results.

    Each experiment method returns a structured results dict containing
    per-condition summaries with mean and standard deviation for each metric.

    Attributes:
        num_episodes: Number of episodes per experimental condition.
        seed: Base random seed for reproducibility.
    """

    def __init__(self, num_episodes: int = DEFAULT_EPISODES, seed: int = RANDOM_SEED):
        """Initialize the experiment runner.

        Args:
            num_episodes: Episodes to run per condition.
            seed: Base random seed.
        """
        self.num_episodes = num_episodes
        self.seed = seed

    def _run_episodes(self, layout, grid_rows, grid_cols, agent_class,
                      boundary_mode='wall', max_steps=200,
                      agent_kwargs=None, obstacle_count=0) -> list:
        """Run multiple episodes and collect statistics.

        Args:
            layout: Grid layout (2D list of chars).
            grid_rows: Number of rows.
            grid_cols: Number of columns.
            agent_class: Agent class to instantiate.
            boundary_mode: Boundary handling mode.
            max_steps: Maximum steps per episode.
            agent_kwargs: Additional keyword args for agent constructor.
            obstacle_count: Number of random obstacles to place.

        Returns:
            List of stat dicts, one per episode.
        """
        all_stats = []
        if agent_kwargs is None:
            agent_kwargs = {}

        for ep in range(self.num_episodes):
            random.seed(self.seed + ep)
            env = GridEnvironment(grid_rows, grid_cols, layout=copy.deepcopy(layout),
                                  boundary_mode=boundary_mode)
            if obstacle_count > 0:
                env.place_obstacles_randomly(obstacle_count)

            agent = agent_class(env, **agent_kwargs)

            for step_num in range(max_steps):
                event = agent.step()
                if event == 'goal_reached':
                    break
                if hasattr(agent, 'is_alive') and not agent.is_alive():
                    break

            all_stats.append(agent.get_stats())

        return all_stats

    # =================================================================
    # Phase 1 Experiments
    # =================================================================

    def run_e1_boundary_modes(self) -> dict:
        """E1: Compare boundary modes (wall, bouncy, wrap) on 5x5 grid.

        Research Question: How do different boundary conditions affect
        agent reachability and navigation efficiency?

        Returns:
            Dict mapping boundary_mode to summary stats.
        """
        results = {}
        for mode in ['wall', 'bouncy', 'wrap']:
            stats = self._run_episodes(
                GRID_5x5, 5, 5, ReactiveAgent,
                boundary_mode=mode, max_steps=MAX_STEPS_PHASE1)
            results[mode] = {
                'coverage': _summarize(stats, 'coverage'),
                'steps_taken': _summarize(stats, 'steps_taken'),
                'goal_reached_rate': _compute_mean(
                    [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                'resources_collected': _summarize(stats, 'resources_collected'),
                'raw': stats,
            }
        return results

    def run_e2_grid_sizes(self) -> dict:
        """E2: Compare grid sizes (5x5 vs 6x4) with wall boundary.

        Returns:
            Dict mapping grid_name to summary stats.
        """
        configs = [
            ('5x5', GRID_5x5, 5, 5),
            ('6x4', GRID_6x4, 6, 4),
        ]
        results = {}
        for name, layout, rows, cols in configs:
            stats = self._run_episodes(
                layout, rows, cols, ReactiveAgent,
                boundary_mode='wall', max_steps=MAX_STEPS_PHASE1)
            results[name] = {
                'coverage': _summarize(stats, 'coverage'),
                'steps_taken': _summarize(stats, 'steps_taken'),
                'goal_reached_rate': _compute_mean(
                    [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                'raw': stats,
            }
        return results

    def run_e1_coverage_over_time(self) -> dict:
        """E1 supplement: Track coverage over time for each boundary mode.

        Returns:
            Dict mapping boundary_mode to list of average coverage at each step.
        """
        results = {}
        max_steps = MAX_STEPS_PHASE1
        for mode in ['wall', 'bouncy', 'wrap']:
            coverage_over_time = [[] for _ in range(max_steps)]
            for ep in range(self.num_episodes):
                random.seed(self.seed + ep)
                env = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5),
                                      boundary_mode=mode)
                agent = ReactiveAgent(env)
                non_wall = env.count_non_wall_cells()
                for step_num in range(max_steps):
                    agent.step()
                    cov = len(agent.visited_cells) / non_wall if non_wall > 0 else 0
                    coverage_over_time[step_num].append(cov)
                    if agent.goal_reached:
                        # Fill remaining steps with final coverage
                        for s in range(step_num + 1, max_steps):
                            coverage_over_time[s].append(cov)
                        break
            # Average across episodes
            results[mode] = [_compute_mean(step_data) if step_data else 0
                             for step_data in coverage_over_time]
        return results

    # =================================================================
    # Phase 2 Experiments
    # =================================================================

    def run_e3_memory_comparison(self) -> dict:
        """E3: Compare Reactive vs Memory vs MST agents.

        Research Question: How does agent memory improve navigation
        efficiency and decision quality?

        Returns:
            Dict mapping agent_type to summary stats.
        """
        agent_configs = [
            ('Reactive', ReactiveAgent),
            ('Memory', MemoryAgent),
            ('MST', MSTAgent),
        ]
        results = {}
        for name, agent_class in agent_configs:
            stats = self._run_episodes(
                GRID_5x5, 5, 5, agent_class,
                boundary_mode='wall', max_steps=MAX_STEPS_PHASE2)
            results[name] = {
                'steps_taken': _summarize(stats, 'steps_taken'),
                'coverage': _summarize(stats, 'coverage'),
                'resources_collected': _summarize(stats, 'resources_collected'),
                'goal_reached_rate': _compute_mean(
                    [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                'revisit_ratio': _summarize(stats, 'revisit_ratio')
                    if 'revisit_ratio' in stats[0] else {'mean': 0, 'std': 0},
                'raw': stats,
            }
        return results

    def run_e4_obstacle_impact(self) -> dict:
        """E4: Vary obstacle count with MemoryAgent.

        Returns:
            Dict mapping obstacle_count to summary stats.
        """
        results = {}
        for obs_count in OBSTACLE_COUNTS:
            stats = self._run_episodes(
                GRID_5x5, 5, 5, MemoryAgent,
                boundary_mode='wall', max_steps=MAX_STEPS_PHASE2,
                obstacle_count=obs_count)
            results[obs_count] = {
                'steps_taken': _summarize(stats, 'steps_taken'),
                'coverage': _summarize(stats, 'coverage'),
                'resources_collected': _summarize(stats, 'resources_collected'),
                'goal_reached_rate': _compute_mean(
                    [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                'obstacles_pushed': _summarize(stats, 'obstacles_pushed'),
                'raw': stats,
            }
        return results

    # =================================================================
    # Phase 3 Experiments
    # =================================================================

    def _run_phase3_episodes(self, controller, energy=DEFAULT_ENERGY,
                              noise=0.0, max_steps=MAX_STEPS_PHASE3) -> list:
        """Helper to run Phase 3 episodes with a given controller.

        Args:
            controller: Controller instance.
            energy: Starting energy budget.
            noise: Action noise probability.
            max_steps: Maximum steps per episode.

        Returns:
            List of stat dicts.
        """
        all_stats = []
        for ep in range(self.num_episodes):
            random.seed(self.seed + ep)
            env = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5),
                                  boundary_mode='wall')
            # Reset stateful controllers between episodes
            if hasattr(controller, 'recent_positions'):
                controller.recent_positions = []
            agent = HierarchicalAgent(
                env, energy=energy, controller=controller, noise=noise)

            for step_num in range(max_steps):
                event = agent.step()
                if event == 'goal_reached' or event == 'dead':
                    break
                if not agent.is_alive():
                    break

            all_stats.append(agent.get_stats())
        return all_stats

    def run_e5_controller_comparison(self) -> dict:
        """E5: Compare four controller types.

        Research Question: What is the impact of hierarchical decision-making
        on agent performance under uncertainty?

        Returns:
            Dict mapping controller_name to summary stats.
        """
        controllers = [
            ('Explorer', ExplorerController()),
            ('Reactive', ReactiveController()),
            ('Deliberative', DeliberativeController()),
            ('Hierarchical', HierarchicalController()),
        ]
        results = {}
        for name, ctrl in controllers:
            stats = self._run_phase3_episodes(ctrl, noise=0.1)
            results[name] = {
                'steps_taken': _summarize(stats, 'steps_taken'),
                'coverage': _summarize(stats, 'coverage'),
                'resources_collected': _summarize(stats, 'resources_collected'),
                'goal_reached_rate': _compute_mean(
                    [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                'energy_remaining': _summarize(stats, 'energy_remaining'),
                'total_reward': _summarize(stats, 'total_reward'),
                'survived_rate': _compute_mean(
                    [1.0 if s.get('survived', False) else 0.0 for s in stats]),
                'raw': stats,
            }
        return results

    def run_e6_noise_sensitivity(self) -> dict:
        """E6: Vary noise levels across controller types.

        Returns:
            Dict mapping (controller_name, noise_level) to summary stats.
        """
        controllers = [
            ('Explorer', ExplorerController()),
            ('Reactive', ReactiveController()),
            ('Deliberative', DeliberativeController()),
            ('Hierarchical', HierarchicalController()),
        ]
        results = {}
        for ctrl_name, ctrl in controllers:
            for noise in NOISE_LEVELS:
                stats = self._run_phase3_episodes(ctrl, noise=noise)
                results[(ctrl_name, noise)] = {
                    'goal_reached_rate': _compute_mean(
                        [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                    'steps_taken': _summarize(stats, 'steps_taken'),
                    'resources_collected': _summarize(stats, 'resources_collected'),
                    'survived_rate': _compute_mean(
                        [1.0 if s.get('survived', False) else 0.0 for s in stats]),
                }
        return results

    def run_e7_energy_budget(self) -> dict:
        """E7: Vary starting energy budgets with HierarchicalController.

        Returns:
            Dict mapping energy_budget to summary stats.
        """
        results = {}
        ctrl = HierarchicalController()
        for energy in ENERGY_BUDGETS:
            stats = self._run_phase3_episodes(ctrl, energy=energy, noise=0.1)
            results[energy] = {
                'goal_reached_rate': _compute_mean(
                    [1.0 if s['goal_reached'] else 0.0 for s in stats]),
                'resources_collected': _summarize(stats, 'resources_collected'),
                'steps_taken': _summarize(stats, 'steps_taken'),
                'survived_rate': _compute_mean(
                    [1.0 if s.get('survived', False) else 0.0 for s in stats]),
            }
        return results

    def run_e5_energy_over_time(self) -> dict:
        """E5 supplement: Track average energy over time per controller.

        Returns:
            Dict mapping controller_name to list of average energy at each step.
        """
        controllers = [
            ('Explorer', ExplorerController()),
            ('Reactive', ReactiveController()),
            ('Deliberative', DeliberativeController()),
            ('Hierarchical', HierarchicalController()),
        ]
        results = {}
        max_steps = 100  # Track first 100 steps for clarity

        for ctrl_name, ctrl in controllers:
            energy_traces = []
            for ep in range(self.num_episodes):
                random.seed(self.seed + ep)
                env = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5),
                                      boundary_mode='wall')
                agent = HierarchicalAgent(
                    env, energy=DEFAULT_ENERGY, controller=ctrl, noise=0.1)

                episode_energy = [agent.energy]
                for step_num in range(max_steps):
                    event = agent.step()
                    episode_energy.append(agent.energy)
                    if event == 'goal_reached' or event == 'dead':
                        break
                    if not agent.is_alive():
                        break

                # Pad to max_steps if episode ended early
                while len(episode_energy) < max_steps + 1:
                    episode_energy.append(episode_energy[-1])
                energy_traces.append(episode_energy[:max_steps + 1])

            # Average across episodes at each time step
            avg_energy = []
            for t in range(max_steps + 1):
                values = [trace[t] for trace in energy_traces]
                avg_energy.append(_compute_mean(values))
            results[ctrl_name] = avg_energy

        return results
