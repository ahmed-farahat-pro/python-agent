"""
visualizer.py - Terminal rendering and matplotlib plot generation.

Provides text-based grid display for terminal output and matplotlib
plotting functions for generating experimental result visualizations.
All plots are saved as PNG files in the results/ directory.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


# Use a clean academic style
plt.rcParams.update({
    'figure.figsize': (8, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

RESULTS_DIR = 'results'
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
MARKERS = ['o', 's', '^', 'D', 'v']


def ensure_results_dir():
    """Create the results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def print_grid(env, step: int = None, agent=None):
    """Print the grid environment to the terminal.

    Args:
        env: GridEnvironment instance.
        step: Optional current step number for the header.
        agent: Optional agent reference for status display.
    """
    header_parts = []
    if step is not None:
        header_parts.append(f'Step {step}')
    if agent is not None:
        if hasattr(agent, 'energy'):
            header_parts.append(f'Energy: {agent.energy}')
        header_parts.append(f'Resources: {agent.resources_collected}')
        header_parts.append(f'Coverage: {len(agent.visited_cells)}/{env.count_non_wall_cells()}')

    if header_parts:
        print(' | '.join(header_parts))
    print(env.render())
    print()


# =================================================================
# Phase 1 Plots
# =================================================================

def plot_coverage_by_boundary_mode(results: dict, filename: str = 'p1_coverage_by_boundary.png'):
    """Bar chart: Coverage % by boundary mode with error bars.

    Args:
        results: Dict from run_e1_boundary_modes().
        filename: Output filename.
    """
    ensure_results_dir()
    modes = list(results.keys())
    means = [results[m]['coverage']['mean'] * 100 for m in modes]
    stds = [results[m]['coverage']['std'] * 100 for m in modes]

    fig, ax = plt.subplots()
    bars = ax.bar(modes, means, yerr=stds, capsize=5,
                  color=COLORS[:len(modes)], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Boundary Mode')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Phase 1: Grid Coverage by Boundary Mode (5x5 Grid)')
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


def plot_coverage_over_time(results: dict, filename: str = 'p1_coverage_over_time.png'):
    """Line plot: Coverage over time by boundary mode.

    Args:
        results: Dict from run_e1_coverage_over_time().
        filename: Output filename.
    """
    ensure_results_dir()
    fig, ax = plt.subplots()

    for i, (mode, coverage_data) in enumerate(results.items()):
        steps = range(len(coverage_data))
        ax.plot(steps, [c * 100 for c in coverage_data],
                label=mode.capitalize(), color=COLORS[i],
                marker=MARKERS[i], markevery=max(1, len(coverage_data) // 10),
                linewidth=1.5, markersize=4)

    ax.set_xlabel('Step Number')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Phase 1: Coverage Over Time by Boundary Mode')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


def plot_grid_size_comparison(results: dict, filename: str = 'p1_grid_size_comparison.png'):
    """Bar chart: Coverage and goal rate by grid size.

    Args:
        results: Dict from run_e2_grid_sizes().
        filename: Output filename.
    """
    ensure_results_dir()
    grids = list(results.keys())
    coverage_means = [results[g]['coverage']['mean'] * 100 for g in grids]
    goal_rates = [results[g]['goal_reached_rate'] * 100 for g in grids]

    x = range(len(grids))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar([i - width / 2 for i in x], coverage_means, width,
                   label='Coverage %', color=COLORS[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar([i + width / 2 for i in x], goal_rates, width,
                   label='Goal Reached %', color=COLORS[1], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Phase 1: Performance by Grid Size')
    ax.set_xticks(list(x))
    ax.set_xticklabels(grids)
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


# =================================================================
# Phase 2 Plots
# =================================================================

def plot_agent_comparison(results: dict, filename: str = 'p2_agent_comparison.png'):
    """Grouped bar chart: Steps and resources by agent type.

    Args:
        results: Dict from run_e3_memory_comparison().
        filename: Output filename.
    """
    ensure_results_dir()
    agents = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Resources collected
    res_means = [results[a]['resources_collected']['mean'] for a in agents]
    res_stds = [results[a]['resources_collected']['std'] for a in agents]
    bars1 = ax1.bar(agents, res_means, yerr=res_stds, capsize=5,
                    color=COLORS[:len(agents)], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Agent Type')
    ax1.set_ylabel('Resources Collected')
    ax1.set_title('Resources Collected by Agent Type')

    for bar, mean in zip(bars1, res_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    # Right: Goal reached rate
    goal_rates = [results[a]['goal_reached_rate'] * 100 for a in agents]
    bars2 = ax2.bar(agents, goal_rates,
                    color=COLORS[:len(agents)], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Agent Type')
    ax2.set_ylabel('Goal Reached Rate (%)')
    ax2.set_title('Goal Reached Rate by Agent Type')
    ax2.set_ylim(0, 110)

    for bar, rate in zip(bars2, goal_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


def plot_revisit_ratio(results: dict, filename: str = 'p2_revisit_ratio.png'):
    """Bar chart: Revisit ratio by agent type.

    Args:
        results: Dict from run_e3_memory_comparison().
        filename: Output filename.
    """
    ensure_results_dir()
    agents = list(results.keys())
    ratios = [results[a]['revisit_ratio']['mean'] * 100 for a in agents]
    stds = [results[a]['revisit_ratio']['std'] * 100 for a in agents]

    fig, ax = plt.subplots()
    bars = ax.bar(agents, ratios, yerr=stds, capsize=5,
                  color=COLORS[:len(agents)], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Revisit Ratio (%)')
    ax.set_title('Phase 2: Revisit Ratio by Agent Type')

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


def plot_obstacle_impact(results: dict, filename: str = 'p2_obstacle_impact.png'):
    """Bar chart: Performance by obstacle count.

    Args:
        results: Dict from run_e4_obstacle_impact().
        filename: Output filename.
    """
    ensure_results_dir()
    counts = sorted(results.keys())
    labels = [str(c) for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Steps taken
    steps_means = [results[c]['steps_taken']['mean'] for c in counts]
    steps_stds = [results[c]['steps_taken']['std'] for c in counts]
    ax1.bar(labels, steps_means, yerr=steps_stds, capsize=5,
            color=COLORS[:len(counts)], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Obstacles')
    ax1.set_ylabel('Steps Taken')
    ax1.set_title('Steps Taken by Obstacle Count')

    # Right: Goal reached rate
    goal_rates = [results[c]['goal_reached_rate'] * 100 for c in counts]
    ax2.bar(labels, goal_rates,
            color=COLORS[:len(counts)], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Obstacles')
    ax2.set_ylabel('Goal Reached Rate (%)')
    ax2.set_title('Goal Reached Rate by Obstacle Count')
    ax2.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


# =================================================================
# Phase 3 Plots
# =================================================================

def plot_controller_comparison(results: dict, filename: str = 'p3_controller_comparison.png'):
    """Grouped bar chart: Performance by controller type.

    Args:
        results: Dict from run_e5_controller_comparison().
        filename: Output filename.
    """
    ensure_results_dir()
    controllers = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Success rate
    rates = [results[c]['goal_reached_rate'] * 100 for c in controllers]
    axes[0].bar(controllers, rates, color=COLORS[:len(controllers)],
                edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Goal Reached Rate (%)')
    axes[0].set_title('Success Rate')
    axes[0].set_ylim(0, 110)

    # Resources collected
    res_means = [results[c]['resources_collected']['mean'] for c in controllers]
    res_stds = [results[c]['resources_collected']['std'] for c in controllers]
    axes[1].bar(controllers, res_means, yerr=res_stds, capsize=5,
                color=COLORS[:len(controllers)], edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Resources Collected')
    axes[1].set_title('Resources Collected')

    # Total reward
    rew_means = [results[c]['total_reward']['mean'] for c in controllers]
    rew_stds = [results[c]['total_reward']['std'] for c in controllers]
    axes[2].bar(controllers, rew_means, yerr=rew_stds, capsize=5,
                color=COLORS[:len(controllers)], edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('Total Reward')
    axes[2].set_title('Total Reward')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Phase 3: Controller Performance Comparison (noise=0.1)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')


def plot_noise_sensitivity(results: dict, filename: str = 'p3_noise_sensitivity.png'):
    """Grouped bar chart: Success rate by controller and noise level.

    Args:
        results: Dict from run_e6_noise_sensitivity().
        filename: Output filename.
    """
    ensure_results_dir()
    # Extract unique controllers and noise levels
    controllers = []
    noise_levels = []
    for (ctrl, noise) in results.keys():
        if ctrl not in controllers:
            controllers.append(ctrl)
        if noise not in noise_levels:
            noise_levels.append(noise)
    noise_levels.sort()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(noise_levels))
    width = 0.18
    offsets = [(-1.5, -0.5, 0.5, 1.5)[i] * width for i in range(len(controllers))]

    for i, ctrl in enumerate(controllers):
        rates = [results[(ctrl, n)]['goal_reached_rate'] * 100 for n in noise_levels]
        offset = offsets[i]
        ax.bar([xi + offset for xi in x], rates, width,
               label=ctrl, color=COLORS[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Goal Reached Rate (%)')
    ax.set_title('Phase 3: Success Rate by Controller and Noise Level')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(n) for n in noise_levels])
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


def plot_energy_over_time(results: dict, filename: str = 'p3_energy_over_time.png'):
    """Line plot: Average energy over time by controller type.

    Args:
        results: Dict from run_e5_energy_over_time().
        filename: Output filename.
    """
    ensure_results_dir()
    fig, ax = plt.subplots()

    for i, (ctrl_name, energy_data) in enumerate(results.items()):
        steps = range(len(energy_data))
        ax.plot(steps, energy_data, label=ctrl_name, color=COLORS[i],
                marker=MARKERS[i], markevery=max(1, len(energy_data) // 10),
                linewidth=1.5, markersize=4)

    ax.set_xlabel('Step Number')
    ax.set_ylabel('Average Energy')
    ax.set_title('Phase 3: Energy Over Time by Controller Type')
    ax.legend()
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Energy = 0')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f'  Saved: {filename}')


def plot_energy_budget(results: dict, filename: str = 'p3_energy_budget.png'):
    """Bar chart: Success rate by starting energy budget.

    Args:
        results: Dict from run_e7_energy_budget().
        filename: Output filename.
    """
    ensure_results_dir()
    budgets = sorted(results.keys())
    labels = [str(b) for b in budgets]
    rates = [results[b]['goal_reached_rate'] * 100 for b in budgets]
    resources = [results[b]['resources_collected']['mean'] for b in budgets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(labels, rates, color=COLORS[:len(budgets)], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Starting Energy')
    ax1.set_ylabel('Goal Reached Rate (%)')
    ax1.set_title('Success Rate by Energy Budget')
    ax1.set_ylim(0, 110)

    ax2.bar(labels, resources, color=COLORS[:len(budgets)], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Starting Energy')
    ax2.set_ylabel('Avg Resources Collected')
    ax2.set_title('Resources Collected by Energy Budget')

    plt.suptitle('Phase 3: Energy Budget Impact (Hierarchical Controller)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')


# =================================================================
# Summary Table
# =================================================================

def print_summary_table(phase: str, results: dict, metrics: list):
    """Print a formatted summary table to the terminal.

    Args:
        phase: Phase name for the header.
        results: Dict mapping condition name to metric dicts.
        metrics: List of metric names to display.
    """
    print(f'\n{"=" * 70}')
    print(f'  {phase} - Summary Results')
    print(f'{"=" * 70}')

    # Header
    header = f'{"Condition":<20}'
    for metric in metrics:
        header += f'{metric:<18}'
    print(header)
    print('-' * 70)

    # Rows
    for condition, data in results.items():
        row = f'{str(condition):<20}'
        for metric in metrics:
            if metric in data:
                val = data[metric]
                if isinstance(val, dict) and 'mean' in val:
                    row += f'{val["mean"]:>7.2f} +/- {val["std"]:>5.2f}  '
                elif isinstance(val, float):
                    row += f'{val:>7.2f}            '
                else:
                    row += f'{str(val):<18}'
            else:
                row += f'{"N/A":<18}'
        print(row)

    print(f'{"=" * 70}\n')
