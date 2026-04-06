"""
main.py - Entry point for the Intelligent Agent Navigation project.

Runs all three experimental phases and generates results:
  Phase 1: Reactive Navigation with boundary mode comparison
  Phase 2: Memory-based agents and MST planning
  Phase 3: Hierarchical controllers with energy and stochasticity

Usage:
    python3 main.py

Output:
    Terminal: Grid visualizations, experiment summaries
    results/: PNG plots for each experiment
"""

import os
import random
import copy
from config import GRID_5x5, GRID_6x4, RANDOM_SEED, DEFAULT_ENERGY
from environment import GridEnvironment
from agent import ReactiveAgent, MemoryAgent, HierarchicalAgent
from mst import MSTAgent
from controllers import HierarchicalController
from experiments import ExperimentRunner
from visualizer import (
    print_grid, ensure_results_dir,
    plot_coverage_by_boundary_mode, plot_coverage_over_time,
    plot_grid_size_comparison,
    plot_agent_comparison, plot_revisit_ratio, plot_obstacle_impact,
    plot_controller_comparison, plot_noise_sensitivity,
    plot_energy_over_time, plot_energy_budget,
    print_summary_table,
)


def demo_grid(title: str, layout: list, rows: int, cols: int,
              boundary_mode: str = 'wall'):
    """Display a grid environment demo in the terminal.

    Args:
        title: Description to print.
        layout: Grid layout.
        rows: Number of rows.
        cols: Number of columns.
        boundary_mode: Boundary mode to use.
    """
    print(f'\n--- {title} ---')
    env = GridEnvironment(rows, cols, layout=copy.deepcopy(layout),
                          boundary_mode=boundary_mode)
    print(env.render())
    print()


def run_phase_1(runner: ExperimentRunner):
    """Phase 1: Reactive Navigation experiments.

    Demonstrates grid environments and runs boundary mode comparison
    experiments to answer: How do different boundary conditions affect
    agent reachability and navigation efficiency?

    Args:
        runner: ExperimentRunner instance.
    """
    print('\n' + '=' * 70)
    print('  PHASE 1: Reactive Navigation')
    print('=' * 70)
    print('\nResearch Question: How do different boundary conditions')
    print('affect agent reachability and navigation efficiency?\n')

    # Demo: show both grid layouts
    demo_grid('5x5 Grid Layout', GRID_5x5, 5, 5)
    demo_grid('6x4 Grid Layout', GRID_6x4, 6, 4)

    # Demo: show agent moving a few steps
    print('--- Reactive Agent Demo (5x5, wall mode, 10 steps) ---')
    random.seed(RANDOM_SEED)
    env = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5), boundary_mode='wall')
    agent = ReactiveAgent(env)
    print_grid(env, step=0, agent=agent)
    for s in range(1, 11):
        event = agent.step()
        if s % 5 == 0 or event in ('resource_collected', 'goal_reached'):
            print(f'  Step {s}: action -> {event}')
    print_grid(env, step=10, agent=agent)

    # E1: Boundary mode comparison
    print('Running E1: Boundary Mode Comparison...')
    e1_results = runner.run_e1_boundary_modes()
    print_summary_table('Phase 1 - Boundary Modes', e1_results,
                        ['coverage', 'goal_reached_rate', 'resources_collected'])
    plot_coverage_by_boundary_mode(e1_results)

    # E1 supplement: Coverage over time
    print('Running E1 supplement: Coverage Over Time...')
    e1_cov = runner.run_e1_coverage_over_time()
    plot_coverage_over_time(e1_cov)

    # E2: Grid size comparison
    print('Running E2: Grid Size Comparison...')
    e2_results = runner.run_e2_grid_sizes()
    print_summary_table('Phase 1 - Grid Sizes', e2_results,
                        ['coverage', 'goal_reached_rate'])
    plot_grid_size_comparison(e2_results)

    print('Phase 1 complete.\n')


def run_phase_2(runner: ExperimentRunner):
    """Phase 2: Object Interaction and Planning experiments.

    Compares reactive, memory-based, and MST agents to answer:
    How does agent memory improve navigation efficiency and decision quality?

    Args:
        runner: ExperimentRunner instance.
    """
    print('\n' + '=' * 70)
    print('  PHASE 2: Object Interaction and Planning')
    print('=' * 70)
    print('\nResearch Question: How does agent memory improve navigation')
    print('efficiency and decision quality?\n')

    # Demo: Memory agent with obstacle pushing
    print('--- Memory Agent Demo (5x5, 10 steps) ---')
    random.seed(RANDOM_SEED)
    env = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5), boundary_mode='wall')
    env.place_obstacles_randomly(2)
    agent = MemoryAgent(env)
    print_grid(env, step=0, agent=agent)
    for s in range(1, 11):
        event = agent.step()
        if event in ('obstacle_pushed', 'resource_collected', 'goal_reached'):
            print(f'  Step {s}: {event}')
    print_grid(env, step=10, agent=agent)
    stats = agent.get_stats()
    print(f'  Visited: {stats["visited_count"]} cells | '
          f'Revisits: {stats["revisit_count"]} | '
          f'Candidates remaining: {stats["candidate_edges_remaining"]}')

    # Demo: MST agent route planning
    print('\n--- MST Agent Demo (5x5, 15 steps) ---')
    random.seed(RANDOM_SEED)
    env2 = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5), boundary_mode='wall')
    mst_agent = MSTAgent(env2)
    print(f'  MST visit order: {mst_agent.visit_order}')
    print_grid(env2, step=0, agent=mst_agent)
    for s in range(1, 16):
        event = mst_agent.step()
        if event in ('resource_collected', 'goal_reached'):
            print(f'  Step {s}: {event} at {mst_agent.position}')
    print_grid(env2, step=15, agent=mst_agent)

    # E3: Memory comparison
    print('Running E3: Agent Memory Comparison...')
    e3_results = runner.run_e3_memory_comparison()
    print_summary_table('Phase 2 - Agent Comparison', e3_results,
                        ['coverage', 'goal_reached_rate', 'resources_collected', 'revisit_ratio'])
    plot_agent_comparison(e3_results)
    plot_revisit_ratio(e3_results)

    # E4: Obstacle impact
    print('Running E4: Obstacle Impact...')
    e4_results = runner.run_e4_obstacle_impact()
    print_summary_table('Phase 2 - Obstacle Impact', e4_results,
                        ['coverage', 'goal_reached_rate', 'obstacles_pushed'])
    plot_obstacle_impact(e4_results)

    print('Phase 2 complete.\n')


def run_phase_3(runner: ExperimentRunner):
    """Phase 3: Perception, Hierarchy, and Learning experiments.

    Compares four controller strategies under varying noise and energy
    conditions to answer: What is the impact of hierarchical decision-making
    on agent performance under uncertainty?

    Args:
        runner: ExperimentRunner instance.
    """
    print('\n' + '=' * 70)
    print('  PHASE 3: Perception, Hierarchy, and Learning')
    print('=' * 70)
    print('\nResearch Question: What is the impact of hierarchical')
    print('decision-making on agent performance under uncertainty?\n')

    # Demo: Hierarchical agent
    print('--- Hierarchical Agent Demo (5x5, energy=50, noise=0.1) ---')
    random.seed(RANDOM_SEED)
    env = GridEnvironment(5, 5, layout=copy.deepcopy(GRID_5x5), boundary_mode='wall')
    ctrl = HierarchicalController()
    agent = HierarchicalAgent(env, energy=DEFAULT_ENERGY, controller=ctrl, noise=0.1)
    print_grid(env, step=0, agent=agent)
    for s in range(1, 21):
        event = agent.step()
        if event in ('resource_collected', 'goal_reached', 'dead'):
            print(f'  Step {s}: {event} | Energy: {agent.energy} | '
                  f'Strategy: {ctrl.current_strategy}')
            if event in ('goal_reached', 'dead'):
                break
    print_grid(env, step=min(s, 20), agent=agent)
    stats = agent.get_stats()
    print(f'  Total reward: {stats["total_reward"]} | '
          f'Energy remaining: {stats["energy_remaining"]} | '
          f'Resources: {stats["resources_collected"]}')

    # E5: Controller comparison
    print('\nRunning E5: Controller Comparison...')
    e5_results = runner.run_e5_controller_comparison()
    print_summary_table('Phase 3 - Controller Comparison', e5_results,
                        ['goal_reached_rate', 'resources_collected', 'total_reward'])
    plot_controller_comparison(e5_results)

    # E5 supplement: Energy over time
    print('Running E5 supplement: Energy Over Time...')
    e5_energy = runner.run_e5_energy_over_time()
    plot_energy_over_time(e5_energy)

    # E6: Noise sensitivity
    print('Running E6: Noise Sensitivity...')
    e6_results = runner.run_e6_noise_sensitivity()
    plot_noise_sensitivity(e6_results)

    # Print noise sensitivity table
    print(f'\n{"=" * 70}')
    print('  Phase 3 - Noise Sensitivity (Goal Reached Rate %)')
    print(f'{"=" * 70}')
    header = f'{"Controller":<18}'
    noise_levels = sorted(set(n for _, n in e6_results.keys()))
    for n in noise_levels:
        header += f'{"noise=" + str(n):<14}'
    print(header)
    print('-' * 70)
    controllers_seen = []
    for (ctrl, _) in e6_results.keys():
        if ctrl not in controllers_seen:
            controllers_seen.append(ctrl)
    for ctrl in controllers_seen:
        row = f'{ctrl:<18}'
        for n in noise_levels:
            rate = e6_results[(ctrl, n)]['goal_reached_rate'] * 100
            row += f'{rate:>6.1f}%       '
        print(row)
    print(f'{"=" * 70}\n')

    # E7: Energy budget
    print('Running E7: Energy Budget Impact...')
    e7_results = runner.run_e7_energy_budget()
    print_summary_table('Phase 3 - Energy Budget', e7_results,
                        ['goal_reached_rate', 'resources_collected', 'survived_rate'])
    plot_energy_budget(e7_results)

    print('Phase 3 complete.\n')


def main():
    """Main entry point: runs all three phases and generates all results."""
    print('=' * 70)
    print('  Intelligent Agent Navigation and Decision-Making')
    print('  in Discrete Environments')
    print('=' * 70)
    print(f'\nRandom seed: {RANDOM_SEED}')
    print('Results will be saved to the results/ directory.\n')

    random.seed(RANDOM_SEED)
    ensure_results_dir()

    runner = ExperimentRunner()

    run_phase_1(runner)
    run_phase_2(runner)
    run_phase_3(runner)

    # Final summary
    print('=' * 70)
    print('  ALL EXPERIMENTS COMPLETE')
    print('=' * 70)
    print('\nGenerated plots:')
    results_dir = 'results'
    if os.path.exists(results_dir):
        for f in sorted(os.listdir(results_dir)):
            if f.endswith('.png'):
                print(f'  - {f}')
    print(f'\nAll results saved to {results_dir}/')
    print('Experiment run complete.')


if __name__ == '__main__':
    main()
