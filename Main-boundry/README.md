# Chevron-Voronoi: Voronoi-Based Point Placement Optimization

## Overview
The Chevron-Voronoi library provides a modular framework for optimizing the placement of additional points within a convex polygon based on Voronoi feasibility criteria. It is designed to:

- **Generate** random convex polygons with reproducible interior “fixed” seed points.
- **Define** spatial loss functions encoding constraints like minimum cell area, pairwise distances, and boundary margins.
- **Solve** for optimal added-point configurations using population-based and stochastic optimizers (QPSO, GA, DE, SPSA).
- **Search** automatically for the maximum number of additional points that can be placed feasibly.
- **Visualize** results with clipped Voronoi diagrams and quantitative summaries.

This project is ideal for applications such as oil-well placement, sensor distribution, or any scenario where spatial coverage and separation constraints are critical.

---

## Installation

Requires Python 3.8+ and the following packages:

```bash
pip install numpy shapely scipy matplotlib deap
```

---

## Module Reference

### 1. `random_polygon.py`
- **`generate_random_convex_polygon_with_fixed(rand_seed: int = 30, convexhull_param: int = 30, num_fixed: int = 5) -> (Polygon, List[(str, (float, float))])`**  
  - **Purpose:** Creates a reproducible random convex hull and samples `num_fixed` interior points labeled `F_1…F_n`.  
  - **Key steps:**  
    1. Seed RNGs (`rand_seed`).  
    2. Sample `convexhull_param` points and compute their convex hull.  
    3. Uniformly sample interior points, rejecting those outside the polygon.

### 2. `losses.py`
Defines feasibility loss functions:

- **`loss_type_1(fixed_points, added_points, polygon, L2_min) -> float`**  
  - Ensures all points lie strictly inside `polygon` and every pair (added–added or added–fixed) is ≥ `L2_min`.  
  - Returns `-min_cell_area` if feasible, else `+inf`.

- **`loss_type_2(fixed_points, added_points, polygon, L2_min, d_min) -> float`**  
  - Builds on `loss_type_1` and additionally enforces that each added point is ≥ `d_min` from the polygon boundary.

### 3. `solvers.py`
Implements various optimizers for point placement:

- **Internal Helpers**  
  - `_seed_rng(seed)`: Seed Python and NumPy RNGs.  
  - `_project_to_polygon(vec, n, polygon)`: Snap out-of-polygon samples back onto the polygon.  
  - `_sample_points_within(polygon, n, d_min)`: Uniformly sample `n` points inside with a boundary margin.

- **Solvers**  
  - `qpso_pairwise_solver(problem, n, params)`: Quantum-behaved PSO with pairwise coordinate updates and early exit on meeting `A_min`.  
  - `ga_deap_solver(problem, n, params)`: Genetic Algorithm via DEAP (tournament selection, two-point crossover, Gaussian mutation).  
  - `de_scipy_solver(problem, n, params)`: SciPy Differential Evolution with a callback for early termination.  

Each solver returns a list of `(name, (x, y))` for the `n` added points.

### 4. `is_feasible.py`
- **`is_feasible(n, problem, config) -> (bool, List[(str, (float, float))])`**  
  - **Inputs:**  
    - `n`: number of points to add.  
    - `problem`: dict with `fixed`, `polygon`, `A_min`, `loss` (callable), and optionally `d_min`.  
    - `config`: dict with `name` (solver ID) and `params`.  
  - **Behavior:**  
    1. Baseline check for `n=0`.  
    2. Dispatch to the chosen solver (`qpso_pairwise`, `ga_deap`, or `de_scipy`).  
    3. Evaluate the solver’s output via the loss function.  
    4. Return feasibility status and the candidate points.

### 5. `find_max_additional.py`
- **`find_max_additional(problem, config, plot_each=False) -> (int, List[(str, (float, float))])`**  
  - **Purpose:** Automatically determine the maximum feasible `n` by:  
    1. **Exponential search**: double `n` until infeasible.  
    2. **Binary search**: refine between the last feasible and first infeasible `n`.  
  - **Optionally** plots intermediate feasible configurations via `plot_voronoi`.

### 6. `plot_voronoi.py`
- **`plot_voronoi(fixed, added, polygon, solver_name) -> (List[Polygon], List[(str, float)])`**  
  - Computes and clips the Voronoi diagram to `polygon`.  
  - Shades cells, plots fixed (red) and added (blue) points.  
  - Calculates and prints:  
    - Minimum cell area  
    - Minimum pairwise distance (added–added & added–fixed)  
    - Minimum boundary distance (added → polygon exterior)  
  - Displays a Matplotlib figure and returns the raw cell geometries and area data.

---
