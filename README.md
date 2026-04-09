Intelligent Agent Navigation and Decision Making in Discrete Environments

1. Introduction

Autonomous systems that operate in structured, constrained spaces are among the most active areas of applied artificial intelligence today. From warehouse logistics robots sorting thousands of parcels each hour to search-and-rescue drones mapping collapsed buildings after natural disasters, the ability of a software agent to perceive its surroundings, take informed decisions, and act in the face of uncertainty is no longer a theoretical exercise but a pressing engineering challenge. This project takes those real-world demands and distills them into a controlled laboratory setting: a grid-based indoor environment where a simulated service robot has to navigate corridors, collect scattered resources such as food packages or data points, avoid both fixed walls and movable obstacles, reach a designated goal location, and do all of this while managing a finite energy budget.

The investigation is organized around three progressively harder phases. Phase 1 establishes a reactive baseline, asking how different boundary conditions, namely rigid walls, bouncy reflections, and toroidal wrap-around, shape the agent’s ability to explore the map. Phase 2 introduces object-oriented memory structures and graph-oriented planning through a Minimum Spanning Tree traversal, asking whether remembering where the agent has already been can meaningfully reduce wasted effort. Phase 3 layers on a hierarchical two-level controller, stochastic action noise, and a depleting energy reserve, asking whether structured decision-making outperforms simpler strategies when the world becomes unpredictable. Together, the three phases trace a path from pure reflex to deliberate reasoning, emulating the evolution seen in real robotic architectures.

2. Methodology

2.1 Environment Design

The environment is modeled as a discrete two-dimensional grid available in two sizes, five-by-five and six-by-four, each containing a mixture of empty floor cells, impassable wall cells, collectible resource cells, a single goal cell, and, in later phases, movable obstacle cells. Three boundary modes govern what happens when the agent attempts to step off the edge of the map. In wall mode, the move is simply rejected, and the agent stays put. In bouncy mode, the agent’s direction reverses upon hitting the boundary, pushing it one cell in the opposite direction. In wrap-around mode, the grid is treated as a torus so that walking off the right edge places the agent on the left edge of the same row. These modes were chosen because they correspond to common assumptions in robotics: hard physical walls, elastic bumpers, and cyclic conveyor layouts, respectively.

**Corresponding outputs.** Quantitative results for boundary modes, coverage over time, and 5x5 vs 6x4 grids appear in `results/p1_coverage_by_boundary.png`, `results/p1_coverage_over_time.png`, and `results/p1_grid_size_comparison.png`. The same figures are embedded in `results.html` under “Phase 1 plots (README §2.1)”. For a qualitative view of movement on the fixed 5x5 layout (wall boundaries), see the reactive step log in `results.html` under “Path traces”.

2.2 Agent Architecture

All agents inherit from a common reactive base class that senses adjacent cells, picks a random valid direction, executes the move, and records statistics such as steps taken, cells visited, and resources collected. The memory agent extends this base by maintaining a set of visited positions and a frontier of candidate edges, which are unvisited cells adjacent to already-visited ones. At each step, the memory agent prefers to move into an unvisited neighbor; when none is available, it uses breadth-first search to find the shortest path back to the nearest frontier cell, thereby systematically covering the grid without the aimless wandering of a purely random walker. The MST agent goes further: before moving, it computes a Minimum Spanning Tree over all resource positions and the goal using Prim’s algorithm with Manhattan-distance weights, then performs a depth-first traversal of the tree to determine an efficient visitation order, navigating between tree nodes via grid-level BFS pathfinding.

For Phase 3, the hierarchical agent adds an energy attribute that decreases by 1 per step and is partially restored whenever a resource is collected. A noise parameter introduces stochastic deviation: with a configurable probability, the agent executes a random perpendicular action instead of the intended one, emulating real-world actuator imprecision. Decision-making is delegated to a pluggable controller object. Four controllers were implemented and compared. The explorer controller picks a direction uniformly at random from all four compass points, including invalid ones, so it frequently wastes moves bumping into walls. The reactive controller first filters out walls, then randomly selects a valid option. The deliberative controller uses BFS to find the shortest path to the goal and follows it, with a cycle-detection mechanism that injects a random move when the agent oscillates between two positions. The hierarchical controller operates at two levels: a high-level strategy selector chooses between survival mode (when energy is low), directing the agent straight to the goal; gathering mode (when resources are nearby); and exploration mode (otherwise), while a low-level navigator uses BFS and greedy direction selection to move toward the chosen sub-goal.

**Corresponding outputs.**  
• **ReactiveAgent** — Phase 1 PNGs above; short random-walk trace in `results.html` (“Path traces”). Compared against Memory and MST in `results/p2_agent_comparison.png` and `results/p2_revisit_ratio.png` (also in `results.html`, Phase 2 plots).  
• **MemoryAgent** — `results/p2_agent_comparison.png`, `results/p2_revisit_ratio.png`; obstacle-count study in `results/p2_obstacle_impact.png` (Phase 2 plots in `results.html`).  
• **MSTAgent** — same Phase 2 figures as Memory.  
• **Explorer / Reactive / Deliberative / Hierarchical controllers** (Phase 3) — `results/p3_controller_comparison.png`, `results/p3_noise_sensitivity.png`, `results/p3_energy_over_time.png`, `results/p3_energy_budget.png` (Phase 3 plots in `results.html`).  
• **DeliberativeController** (deterministic shortest path to goal) — worked example with grids and “Step N: Move …” lines in `results.html` (Deliberative path under Path traces).

2.3 Experimental Setup

Each experimental condition was repeated 50 times with deterministic seeding to ensure reproducibility. A base random seed of forty-two was offset by the episode number so that every agent type faced the same sequence of random settings within a given experiment. Phase 1 episodes ran for up to 200 steps, Phase 2 for 300, and Phase 3 for 500. The metrics recorded per episode included the number of steps taken, whether the goal was reached, the count of resources collected, the fraction of non-wall cells visited, which is referred to as coverage, the revisit ratio defined as the number of steps landing on an already-visited cell divided by total steps, and, for Phase 3, the remaining energy and cumulative reward. Summary statistics report the arithmetic mean and sample standard deviation across episodes.

**Corresponding outputs.** Every run of `python3 main.py` refreshes the ten PNG files under `results/` and the compiled page `results.html` (path traces plus all plots). The extended narrative with the same figures is in `index.html`. Mapping: E1/E2 → Phase 1 PNGs; E3/E4 → Phase 2 PNGs; E5–E7 and supplements → Phase 3 PNGs (see §2.1–2.2 for file names).

3. Results

3.0 Agent path trace (presentation format)

In addition to aggregate metrics and plots, results include an ASCII grid view of the environment and a chronological move log for discussion, in the form ``Step N: Move NORTH`` (or SOUTH, EAST, WEST). Running ``python3 main.py`` refreshes ``results.html`` in the project root. That page includes, in order: (1) **Path traces** — a goal-directed path using the DeliberativeController (no action noise) and a short reactive random walk; (2) **Phase 1–3 plot galleries** — the same PNGs as in ``results/``, each labeled with the experiment ID (E1–E7). The symbol legend matches the grid layouts in code: period for empty floor, hash for walls, A for the agent, G for the goal when the agent is not standing on it, R for resources, and O for movable obstacles.

Example (5x5 layout from ``config.GRID_5x5``, DeliberativeController, seed 42, eight steps to the goal): initial grid:

    A . # . R
    . . . # .
    . # . . .
    . . . # R
    R . # . G

Step-by-step movement:

    Step 1: Move SOUTH
    Step 2: Move EAST
    Step 3: Move EAST
    Step 4: Move SOUTH
    Step 5: Move EAST
    Step 6: Move EAST
    Step 7: Move SOUTH
    Step 8: Move SOUTH

After the episode the agent occupies the goal cell; the rendered grid shows ``A`` in that cell because the environment represents the agent overlay on the former goal location.

**Figures:** ``results/p1_coverage_by_boundary.png``, ``results/p1_coverage_over_time.png``, ``results/p1_grid_size_comparison.png``. In ``results.html``, open the “Phase 1 plots” section.

3.1 Phase 1: Boundary Mode Effects

On the five-by-five grid, the wall-bounded agent obtained a mean coverage of 78.7 percent with a goal-reached rate of 72 percent. Bouncy boundaries produced slightly lower coverage at 75.5 percent and a 62 percent goal rate because the reflection sometimes pushed the agent away from productive directions. Wrap-around yielded only 45.6 percent coverage but, counterintuitively, a 100 percent goal-reached rate: the toroidal topology created shortcuts that let the random walker stumble onto the goal quickly, even though it explored fewer unique cells. Comparing grid sizes, the six-by-four layout produced a higher goal rate of 92 percent versus 72 percent for five-by-five, likely because the rectangular shape offers fewer dead-end corridors.

**Figures:** same as §3.0 Phase 1 list above.

3.2 Phase 2: Memory and Planning

The reactive agent revisited cells on 80.6 percent of its steps, confirming that random movement is overwhelmingly redundant. The memory agent cut that figure to 15.6 percent by actively steering toward unvisited frontier cells. The MST agent was similarly efficient at 16.7 percent revisits but reached the goal in every single episode, compared to the reactive agent’s 84 percent, and collected a consistent 2 out of 3 resources on its planned route. When movable obstacles were introduced, performance dropped sharply: with two obstacles, the memory agent’s goal rate fell to 18 percent, and with four obstacles to just 8 percent, indicating that obstacle pushing, while functional, often blocks critical corridors.

**Figures:** ``results/p2_agent_comparison.png``, ``results/p2_revisit_ratio.png``, ``results/p2_obstacle_impact.png``. In ``results.html``, see “Phase 2 plots”.

3.3 Phase 3: Hierarchical Decision-Making

The controller comparison at 10 percent action noise revealed a sharp hierarchy. The explorer controller reached the goal in only 8 percent of episodes and accumulated a mean reward of 10. The reactive controller improved to 20 percent and 19.4 reward. The deliberative controller, with BFS pathfinding, achieved a perfect 100 percent goal rate and 60 reward, but collected only one resource on average because it headed straight for the goal without detours. The hierarchical controller matched the 100 percent goal rate while also collecting all three resources for a mean reward of 80, demonstrating that the two-level strategy of gathering resources before heading to the goal pays off. Noise sensitivity analysis showed that explorer and reactive controllers degraded as noise increased, while deliberative and hierarchical controllers remained at 100 percent across all tested noise levels from 0 to 30%. The energy-over-time plot provided further insight: explorer and reactive agents’ energy curves plunged toward zero within fifty steps, whereas the hierarchical agent’s energy dipped briefly during resource gathering then stabilized around forty-one thanks to the energy restored by collecting resources.

**Figures:** ``results/p3_controller_comparison.png``, ``results/p3_noise_sensitivity.png``, ``results/p3_energy_over_time.png``, ``results/p3_energy_budget.png``. In ``results.html``, see “Phase 3 plots”. The noise-rate table in ``index.html`` matches E6.

4. Discussion

The three research questions posed at the outset can now be answered concretely. Boundary conditions meaningfully alter navigation dynamics. Wall mode is the most predictable and yields the highest coverage for a random agent because reflections and wraps introduce movement patterns that may bypass useful areas of the grid. Wrap-around creates topological short-cuts that boost goal attainment at the cost of careful exploration, a trade-off that a real system designer would need to weigh against mission requirements.

Agent memory dramatically improves efficiency. Reducing the revisit ratio from over 80 percent to under 17 percent means the memory agent spends five times fewer steps retracing its own path. The MST agent includes a global planning layer that guarantees an efficient visitation order, but its advantage over the simpler memory agent is modest on small grids; the benefit would likely grow on larger maps with more resources.

Hierarchical decision-making is the single largest contributor to effective performance in uncertain conditions. The deliberative controller proves that knowing the shortest path is valuable, but without resource awareness, it leaves reward on the table. The hierarchical controller’s ability to switch strategies based on its internal state, prioritizing survival when energy is low and gathering when resources are nearby, resembles the subsumption architecture proposed by Brooks and remains effective even when 30 percent of its actions deviate randomly.

A main limitation of this study is the grid size. The five-by-five and six-by-four grids are small enough that even a random agent will eventually stumble onto the goal. Scaling to larger environments would likely amplify the performance gaps observed here and might expose weaknesses in the BFS pathfinding, which is efficient on tiny grids but would need heuristic acceleration such as A-star on maps with hundreds of cells.

5. Conclusion

This project implemented and evaluated a progression of self-governing agent strategies, from blind reactive movement to memory-guided exploration and MST-based planning, culminating in hierarchical energy-aware decision-making, all operating within discrete grid worlds that abstract the challenges faced by real indoor service robots. The experimental results confirm that each added layer of advancement delivers measurable gains: memory cuts wasted movement by a factor of five, planning guarantees goal attainment, and layered control maximizes total reward whilst preserving tolerance to action noise. Further research might extend the framework with reinforcement learning to let the agent automatically adapt its strategy thresholds, multi-agent coordination for collaborative resource gathering, and larger, procedurally generated maps to stress-test scalability.
