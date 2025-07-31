# Chevron-Voronoi Point Placement Tutorial

## Overview

This project provides tools for optimizing spatial placement of points within a polygon using Voronoi‐based feasibility criteria. You can:
- Generate random convex polygons with interior seed points.
- Compute and plot clipped Voronoi diagrams.
- Solve for the maximum number of additional points meeting area and spacing constraints using various solvers:
  - QPSO (Quantum Particle Swarm Optimization)
  - GA (Genetic Algorithm)
  - DE (Differential Evolution)
  - SPSA (Simultaneous Perturbation Stochastic Approximation)

## Functions Reference

Below is a descriptive list of all key functions, their inputs, outputs, and detailed roles.

### random_polygon.py
- **generate_random_convex_polygon_with_fixed(rand_seed: int, convexhull_param: int, num_fixed: int) -> (Polygon, List[Tuple[float, float]])**  
  - **Inputs**  
    - `rand_seed`: Random seed for reproducibility.  
    - `convexhull_param`: Parameter controlling convex hull generation (e.g., number of random points).  
    - `num_fixed`: Number of fixed interior points to generate.  
  - **Outputs**  
    - `polygon`: A Shapely `Polygon` object representing the convex hull.  
    - `fixed`: A list of `(x, y)` tuples for the fixed seed points inside the polygon.  
  - **Role & Dependencies**  
    This function creates a random set of points and computes their convex hull to form the bounding polygon. It also samples a specified number of interior points to serve as the “fixed” seeds. It depends on `numpy` for random sampling and `shapely` for geometric operations.

### losses.py
- **loss_type_1(points: np.ndarray, polygon: Polygon, L2_min: float) -> float**  
  - **Inputs**  
    - `points`: Array of shape `(n, 2)` for the candidate points.  
    - `polygon`: A Shapely `Polygon` to clip Voronoi cells.  
    - `L2_min`: Minimum allowed pairwise Euclidean distance.  
  - **Outputs**  
    - `loss`: Returns `float('inf')` if any cell area falls below the minimum area threshold or if any point‐to‐point distance is less than `L2_min`; otherwise returns the negative of the smallest cell area.  
  - **Role & Dependencies**  
    This function evaluates feasibility by computing the Voronoi tessellation of the combined fixed and candidate points (using `shapely.ops.voronoi_diagram`), then checks area and spacing constraints. It uses `numpy` for distance calculations and `shapely` for geometry.

### plot_voronoi.py
- **plot_voronoi(fixed: List[Tuple[float, float]], added: List[Tuple[float, float]], polygon: Polygon, solver_name: str) -> None**  
  - **Inputs**  
    - `fixed`: List of fixed seed point coordinates.  
    - `added`: List of newly added seed point coordinates.  
    - `polygon`: Shapely `Polygon` to clip the diagram.  
    - `solver_name`: Name of the solver ("qpso", "ga", "de", "spsa").  
  - **Outputs**  
    - None (displays a Matplotlib figure of the clipped Voronoi diagram).  
  - **Role & Dependencies**  
    Visualizes the final or intermediate placement of seeds by generating a clipped Voronoi diagram. Relies on `matplotlib` for plotting, `shapely` for geometry, and optionally on `scipy.spatial.Voronoi` for tessellation.

### solvers.py
- **qpso_solver(n: int, problem: dict, seed: int, popsize: int, maxiter: int) -> np.ndarray**  
- **qpso_pairwise_solver(n: int, problem: dict, seed: int, popsize: int, maxiter: int) -> np.ndarray**  
- **ga_solver(n: int, problem: dict, popsize: int, maxiter: int, seed: int) -> np.ndarray**  
- **de_solver(n: int, problem: dict, popsize: int, maxiter: int, seed: int) -> np.ndarray**  
- **spsa_solver(n: int, problem: dict, a: float, c: float, maxiter: int, seed: int) -> np.ndarray**  
  - **Inputs** (common)  
    - `n`: Number of additional points to place.  
    - `problem`: Dictionary containing `fixed`, `polygon`, `A_min`, `l2_min`, and `loss` function.  
    - `popsize`, `maxiter`, `seed`: Algorithm control parameters.  
    - `a`, `c`: Gain parameters for the SPSA solver.  
  - **Outputs**  
    - Array of shape `(n, 2)` with optimized coordinates for the `n` additional points.  
  - **Role & Dependencies**  
    Implements various heuristic and population‐based solvers. Dependencies include `numpy`, `random`, `scipy.optimize` (for DE), `deap` (for GA), and custom loss functions. Each solver searches for a configuration minimizing the loss while respecting feasibility constraints.

### is_feasible.py
- **is_feasible(n: int, problem: dict, config: dict) -> Tuple[bool, List[Tuple[float, float]]]**  
  - **Inputs**  
    - `n`: Candidate number of additional points.  
    - `problem`: Problem dictionary as above.  
    - `config`: Solver configuration dictionary with `name` and `params`.  
  - **Outputs**  
    - `feasible`: `True` if a feasible configuration is found.  
    - `points`: List of `(x, y)` tuples if feasible, otherwise empty.  
  - **Role & Dependencies**  
    Acts as a wrapper to call the chosen solver and evaluate feasibility using `losses.py`. Depends on the solvers module and the problem definition.

### find_max_additional.py
- **find_max_additional(problem: dict, config: dict, plot_each: bool = False) -> Tuple[int, List[Tuple[float, float]]]**  
  - **Inputs**  
    - `problem`: Problem dictionary as above.  
    - `config`: Solver configuration.  
    - `plot_each`: Flag to plot Voronoi diagrams for each trial `n`.  
  - **Outputs**  
    - `max_n`: Maximum feasible number of additional points.  
    - `best_points`: List of `(x, y)` tuples for the best feasible configuration.  
  - **Role & Dependencies**  
    Coordinates an exponential and binary search over `n`, calling `is_feasible` at each step. Optionally visualizes progress. Depends on `is_feasible.py` and plotting utilities.

## Example Workflow

1. **Generate Inputs**  
   Use `random_polygon.generate_random_convex_polygon_with_fixed` to create your base polygon and fixed seed points.

2. **Define the Problem**  
   Construct a dictionary specifying `fixed`, `polygon`, `A_min`, `l2_min`, and your chosen `loss` function from `losses.py`.

3. **Configure the Solver**  
   Choose a solver by name (e.g., `"qpso"`) and set parameters like population size, iterations, and random seed.

4. **Run the Search**  
   Call `find_max_additional` with your problem and config. This will perform an efficient search over possible numbers of additional points, using `is_feasible` and the solvers, and will return the maximum feasible count along with the optimal coordinates.

5. **Visualize Results**  
   The final set of fixed + added points can be plotted via `plot_voronoi` to inspect coverage and spacing within your polygon.  

Explore `Example.ipynb` for a hands‐on demonstration of this workflow, including intermediate plots and solver comparisons.