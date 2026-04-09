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


def write_path_results_page(
        filename: str = 'results.html',
        seed: Optional[int] = None):
    """Write results.html with sample path traces (project root)."""
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

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Path traces — Agent navigation results</title>
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
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1.25rem 4rem;
        }}
        h1 {{ font-size: 1.75rem; margin-bottom: 0.5rem; }}
        .lede {{ color: var(--muted); margin-bottom: 2rem; }}
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
    <h1>Agent path traces</h1>
    <p class="lede">ASCII grid snapshots and step-by-step movement output
        (same style as the project discussion examples). Regenerated when you
        run <code style="background:var(--surface);padding:2px 8px;border-radius:4px;">python3 main.py</code>.</p>
    <div class="legend">{html.escape(legend)}</div>

    <h2>1. Goal-directed path (DeliberativeController)</h2>
    <p class="meta">{html.escape(d_sum)}</p>
    {pre_block('Initial grid', d_init)}
    {pre_block('Step-by-step movement', d_moves)}
    {pre_block('Final grid', d_fin)}

    <h2>2. Reactive exploration sample (first 15 steps)</h2>
    <p class="meta">{html.escape(r_sum)}</p>
    {pre_block('Initial grid', r_init)}
    {pre_block('Step-by-step movement', r_moves)}
    {pre_block('Final grid', r_fin)}

    <div class="nav-foot">
        <p>Plots and figures: see <a href="index.html">index.html</a> (full report) and the <code>results/</code> directory for PNG exports.</p>
    </div>
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(body)
    print(f'  Saved: {filename}')
