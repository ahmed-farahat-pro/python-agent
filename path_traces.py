"""
path_traces.py - ASCII grid snapshots and step-by-step move logs for reports.

Used to generate results.html without depending on matplotlib.
"""

import copy
import html
import os
import random
from typing import Optional

ACTION_TO_COMPASS = {
    'up': 'NORTH',
    'down': 'SOUTH',
    'left': 'WEST',
    'right': 'EAST',
}


def render_plain_grid(env) -> str:
    """Space-separated ASCII grid (``A`` marks the agent)."""
    lines = []
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            if env.agent_pos is not None and (r, c) == env.agent_pos:
                row.append('A')
            else:
                row.append(env.grid[r][c].value)
        lines.append(' '.join(row))
    return '\n'.join(lines)


def format_step_moves(agent) -> str:
    """``Step N: Move <COMPASS>`` for each entry in action_history."""
    lines = []
    for i, (action, _event) in enumerate(agent.action_history, start=1):
        compass = ACTION_TO_COMPASS.get(action, action.upper())
        lines.append(f'Step {i}: Move {compass}')
    return '\n'.join(lines)


def _run_path_trace_deliberative(seed: int, grid_layout, rows: int, cols: int):
    from environment import GridEnvironment
    from agent import HierarchicalAgent
    from controllers import DeliberativeController

    random.seed(seed)
    env = GridEnvironment(rows, cols, layout=copy.deepcopy(grid_layout),
                          boundary_mode='wall')
    agent = HierarchicalAgent(
        env, energy=100, controller=DeliberativeController(), noise=0.0)
    initial = render_plain_grid(env)
    while not agent.goal_reached and agent.steps_taken < 500:
        ev = agent.step()
        if ev == 'dead':
            break
    final = render_plain_grid(env)
    moves = format_step_moves(agent)
    summary = (
        f'DeliberativeController, deterministic (noise=0), seed={seed}. '
        f'Steps: {agent.steps_taken}, goal_reached={agent.goal_reached}.'
    )
    return initial, final, moves, summary


def _run_path_trace_reactive(seed: int, grid_layout, rows: int, cols: int,
                             max_steps: int = 15):
    from environment import GridEnvironment
    from agent import ReactiveAgent

    random.seed(seed)
    env = GridEnvironment(rows, cols, layout=copy.deepcopy(grid_layout),
                          boundary_mode='wall')
    agent = ReactiveAgent(env)
    initial = render_plain_grid(env)
    for _ in range(max_steps):
        agent.step()
    final = render_plain_grid(env)
    moves = format_step_moves(agent)
    summary = (
        f'ReactiveAgent, first {max_steps} steps, seed={seed}. '
        f'goal_reached={agent.goal_reached}.'
    )
    return initial, final, moves, summary


# Plot gallery: ties each README methodology/results section to PNG files in results/
RESULT_PLOT_SECTIONS = [
    {
        'id': 'phase1-plots',
        'heading': 'Phase 1 plots (README §2.1 Environment — boundaries &amp; grid sizes)',
        'intro': (
            'Wall, bouncy, and wrap boundary modes on 5x5; 5x5 vs 6x4 comparison '
            '(ReactiveAgent). Same content as README §3.1 narrative.'
        ),
        'plots': [
            (
                'Coverage by boundary mode (E1)',
                'Mean coverage with error bars across 50 episodes per boundary mode.',
                'p1_coverage_by_boundary.png',
            ),
            (
                'Coverage over time (E1 supplement)',
                'Average coverage at each step for wall, bouncy, and wrap.',
                'p1_coverage_over_time.png',
            ),
            (
                'Performance by grid size (E2)',
                '5x5 vs 6x4: coverage and goal-reached rate.',
                'p1_grid_size_comparison.png',
            ),
        ],
    },
    {
        'id': 'phase2-plots',
        'heading': 'Phase 2 plots (README §2.2 — Reactive, Memory, MST &amp; obstacles)',
        'intro': (
            'Agent comparison (E3), revisit ratio (E3), obstacle impact with MemoryAgent (E4). '
            'Aligns with README §3.2.'
        ),
        'plots': [
            (
                'Resources &amp; goal rate by agent (E3)',
                'Reactive vs Memory vs MST: resources collected and goal-reached rate.',
                'p2_agent_comparison.png',
            ),
            (
                'Revisit ratio by agent (E3)',
                'Share of steps revisiting an already-visited cell.',
                'p2_revisit_ratio.png',
            ),
            (
                'Obstacle impact (E4)',
                'MemoryAgent with 0 / 2 / 4 movable obstacles: steps and goal rate.',
                'p2_obstacle_impact.png',
            ),
        ],
    },
    {
        'id': 'phase3-plots',
        'heading': 'Phase 3 plots (README §2.2 — controllers, noise, energy)',
        'intro': (
            'Four controllers (E5), energy over time (supplement), noise sensitivity (E6), '
            'energy budget (E7). Aligns with README §3.3.'
        ),
        'plots': [
            (
                'Controller comparison (E5, noise=0.1)',
                'Goal rate, resources collected, total reward.',
                'p3_controller_comparison.png',
            ),
            (
                'Noise sensitivity (E6)',
                'Goal-reached rate vs noise level for each controller.',
                'p3_noise_sensitivity.png',
            ),
            (
                'Energy over time (E5 supplement)',
                'Average energy remaining per step by controller type.',
                'p3_energy_over_time.png',
            ),
            (
                'Energy budget impact (E7)',
                'Hierarchical controller with varying starting energy.',
                'p3_energy_budget.png',
            ),
        ],
    },
]


def _plot_gallery_html() -> str:
    """Build HTML for embedded result figures (paths relative to repo root)."""
    parts = []
    for sec in RESULT_PLOT_SECTIONS:
        parts.append(
            f'<h2 id="{sec["id"]}">{sec["heading"]}</h2>'
            f'<p class="meta">{sec["intro"]}</p>'
        )
        for title, caption, fname in sec['plots']:
            src = html.escape(f'results/{fname}')
            parts.append(
                '<div class="plot-card">'
                f'<h3>{title}</h3>'
                f'<img src="{src}" alt="{html.escape(title)}" loading="lazy">'
                f'<p class="plot-cap">{caption} File: <code>{html.escape(fname)}</code></p>'
                '</div>'
            )
    return '\n'.join(parts)


def write_path_results_page(
        filename: str = 'results.html',
        seed: Optional[int] = None):
    """Write results.html: path traces plus all experiment plots (project root)."""
    from config import GRID_5x5, RANDOM_SEED as DEFAULT_SEED

    if seed is None:
        seed = DEFAULT_SEED

    project_root = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(project_root, filename)

    d_init, d_fin, d_moves, d_sum = _run_path_trace_deliberative(
        DEFAULT_SEED, GRID_5x5, 5, 5)
    r_init, r_fin, r_moves, r_sum = _run_path_trace_reactive(
        seed, GRID_5x5, 5, 5, max_steps=15)

    def pre_block(label: str, text: str) -> str:
        esc = html.escape(text)
        return (
            f'<div class="trace-block">'
            f'<h3>{html.escape(label)}</h3>'
            f'<pre class="trace-pre">{esc}</pre></div>'
        )

    legend = (
        'Legend:  . = empty   # = wall   A = agent   G = goal   '
        'R = resource   O = movable obstacle'
    )

    gallery = _plot_gallery_html()

    path_section = f"""
    <h2 id="path-traces">Path traces (README §2.2 / §3.0)</h2>
    <p class="meta">ASCII grids and <strong>Step N: Move NORTH|SOUTH|EAST|WEST</strong> logs. Reactive baseline vs Deliberative (BFS) controller. Ties to agent types described in the README.</p>
    <div class="legend">{html.escape(legend)}</div>

    <h3 class="h3-sub">1. Goal-directed path (DeliberativeController)</h3>
    <p class="meta">{html.escape(d_sum)}</p>
    {pre_block('Initial grid', d_init)}
    {pre_block('Step-by-step movement', d_moves)}
    {pre_block('Final grid', d_fin)}

    <h3 class="h3-sub">2. Reactive exploration (first 15 steps)</h3>
    <p class="meta">{html.escape(r_sum)}</p>
    {pre_block('Initial grid', r_init)}
    {pre_block('Step-by-step movement', r_moves)}
    {pre_block('Final grid', r_fin)}
"""

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experimental results — plots &amp; path traces</title>
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #1a1d27;
            --border: #2e3348;
            --text: #e2e4ed;
            --muted: #8b8fa8;
            --accent: #6c8cff;
        }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            max-width: 920px;
            margin: 0 auto;
            padding: 2rem 1.25rem 4rem;
        }}
        .subnav {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 1rem;
            margin: 1.25rem 0 2rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .subnav a {{
            color: var(--accent);
            text-decoration: none;
            font-size: 0.9rem;
        }}
        .subnav a:hover {{ text-decoration: underline; }}
        h1 {{ font-size: 1.75rem; margin-bottom: 0.5rem; }}
        .lede {{ color: var(--muted); margin-bottom: 1.5rem; }}
        .legend {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 2rem;
            font-size: 0.95rem;
            color: var(--muted);
        }}
        h2 {{
            font-size: 1.2rem;
            margin: 2.5rem 0 1rem;
            color: var(--accent);
        }}
        .h3-sub {{
            font-size: 1.05rem;
            margin: 1.75rem 0 0.5rem;
            color: var(--text);
        }}
        .meta {{
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }}
        .trace-block {{ margin-bottom: 1.5rem; }}
        .trace-block h3 {{
            font-size: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        .trace-pre {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            overflow-x: auto;
            font-size: 0.9rem;
            line-height: 1.7;
            font-family: ui-monospace, 'Cascadia Code', monospace;
            white-space: pre;
        }}
        .plot-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }}
        .plot-card h3 {{ font-size: 1rem; margin: 0 0 0.75rem; font-weight: 600; }}
        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }}
        .plot-cap {{
            margin: 0.75rem 0 0;
            font-size: 0.88rem;
            color: var(--muted);
        }}
        .plot-card code {{
            background: var(--bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        .nav-foot {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            font-size: 0.9rem;
            color: var(--muted);
        }}
        .nav-foot a {{ color: var(--accent); }}
    </style>
</head>
<body>
    <h1>Experimental results</h1>
    <p class="lede">All figures under <code>results/</code>, path traces, and README cross-references in one page. Regenerated when you run <code>python3 main.py</code> (see <code>path_traces.py</code>).</p>
    <nav class="subnav" aria-label="On this page">
        <a href="#path-traces">Path traces</a>
        <a href="#phase1-plots">Phase 1 plots</a>
        <a href="#phase2-plots">Phase 2 plots</a>
        <a href="#phase3-plots">Phase 3 plots</a>
        <a href="index.html">index.html report</a>
    </nav>

    {path_section}

    <hr style="border: none; border-top: 1px solid var(--border); margin: 3rem 0;">

    {gallery}

    <div class="nav-foot">
        <p>Narrative write-up: <a href="README.md">README.md</a> (view on GitHub). Extended figures with discussion: <a href="index.html">index.html</a>.</p>
    </div>
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(body)
    print(f'  Saved: {filename}')
