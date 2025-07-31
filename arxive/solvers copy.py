"""
Module: solvers

Optimization solvers for placing points within a polygon
subject to Voronoi‐based feasibility (minimum cell area, distance).
Each solver returns only the newly added (name, (x, y)) points.
"""

import random
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

__all__ = [
    "qpso_solver",
    "qpso_pairwise_solver",
    "ga_solver",
    "de_solver",
    "spsa_solver"
]

import random
from shapely.geometry import Point, Polygon
from typing import List, Tuple

def sample_points_within(
    polygon: Polygon,
    n: int,
    d_min: float,
    *,
    seed: int = None
) -> List[Tuple[float, float]]:
    """
    Uniformly sample `n` random points inside `polygon` such that each point
    lies at least `d_min` away from the polygon boundary.

    Args:
        polygon:   A Shapely Polygon defining the sampling region.
        n:         Number of points to sample (must be non‐negative).
        d_min:     Minimum allowed distance from each sampled point to the polygon boundary (>= 0).
        seed:      Optional random seed for reproducibility.

    Returns:
        A list of `n` (x, y) tuples representing the sampled points.

    Raises:
        TypeError:  If argument types are incorrect.
        ValueError: If `n` is negative, `d_min` is negative, or sampling fails after many tries.
    """
    # --- Type & value checks ---
    if not isinstance(polygon, Polygon):
        raise TypeError(f"polygon must be a shapely.geometry.Polygon, got {type(polygon).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError("n must be non‐negative")
    if not isinstance(d_min, (int, float)):
        raise TypeError(f"d_min must be a number, got {type(d_min).__name__}")
    if d_min < 0:
        raise ValueError("d_min must be non‐negative")

    _seed_rng(seed)

    minx, miny, maxx, maxy = polygon.bounds
    # Shrink sampling box by d_min in all directions
    low_x, high_x = minx + d_min, maxx - d_min
    low_y, high_y = miny + d_min, maxy - d_min
    if low_x >= high_x or low_y >= high_y:
        raise ValueError("d_min is too large for the polygon’s dimensions")

    samples: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(10000, n * 1000)

    while len(samples) < n:
        if attempts >= max_attempts:
            raise ValueError(f"Failed to sample {n} points after {attempts} attempts.")
        attempts += 1

        xi = random.uniform(low_x, high_x)
        yi = random.uniform(low_y, high_y)
        pt = Point(xi, yi)

        # Check containment and extra boundary margin (redundant given bounding‐box shrink, but safe)
        if polygon.contains(pt) and polygon.exterior.distance(pt) >= d_min:
            samples.append((xi, yi))

    return samples



def _seed_rng(seed):
    """Seed Python and NumPy RNGs for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _generate_names(prefix, n):
    """Generate sequential names PREFIX_1, PREFIX_2, …, PREFIX_n."""
    return [f"{prefix}_{i+1}" for i in range(n)]


def _project_to_polygon(vec, n, polygon):
    """
    Clamp or snap each (x, y) pair in flat vector `vec` back into `polygon`.
    Uses nearest_points, falling back to bounding‐box clamp on error.
    """
    minx, miny, maxx, maxy = polygon.bounds
    proj = []
    for i in range(n):
        x, y = vec[2*i], vec[2*i+1]
        pt = Point(x, y)
        if not polygon.contains(pt):
            p = nearest_points(polygon, pt)[0]
            if not polygon.contains(p):
                epsilon = 1e-8
                cent = polygon.centroid
                p = Point((1-epsilon)*p.x + epsilon*cent.x,
                          (1-epsilon)*p.y + epsilon*cent.y)
            x, y = p.x, p.y
        proj.extend([x, y])
    return np.array(proj)


def qpso_solver(problem, n, params):
    """
    Quantum‐behaved PSO (QPSO) for spatial point placement.
    Each particle is a 2n‐dim vector of candidate coordinates.
    """
    fixed    = problem["fixed"]
    polygon  = problem["polygon"]
    loss_fn  = problem["loss"]
    A_min    = problem["A_min"]
    d_min    = problem["d_min"]
    popsize  = params["popsize"]
    maxiter  = params["maxiter"]
    alpha    = params.get("alpha", 0.75)
    _seed_rng(params.get("seed"))

    names = _generate_names("A", n)

    def evaluate(vec):
        # build only added list
        added = [(names[i], (vec[2*i], vec[2*i+1])) for i in range(n)]
        return loss_fn(added)

    # Initialize swarm inside polygon using our sampler
    swarm = []
    for i in range(popsize):
        # sample_points_within returns a list of n (x,y) tuples
        pts = sample_points_within(
            polygon=polygon,
            n=n,
            d_min=d_min,            # no extra boundary margin here
            seed=params.get("seed", None)  # optional reproducibility
        )
        # flatten [(x,y),...] → [x,y,x,y,...]
        particle = np.array([coord for pt in pts for coord in pt])
        swarm.append(particle)


    # Personal bests and global best
    pbest    = [p.copy() for p in swarm]
    pbest_f  = [evaluate(p)   for p in swarm]
    gbest_i  = int(np.argmin(pbest_f))
    gbest    = pbest[gbest_i].copy()
    gbest_f  = pbest_f[gbest_i]

    # Early exit if already feasible
    if -gbest_f >= A_min:
        return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    # Main QPSO iterations
    for _ in range(maxiter):
        mbest = sum(pbest) / len(pbest)
        for i, x_i in enumerate(swarm):
            new_x = np.zeros_like(x_i)
            for d in range(len(x_i)):
                u   = random.random()
                phi = random.random()
                Pi  = phi*pbest[i][d] + (1-phi)*gbest[d]
                sign = 1 if random.random() > 0.5 else -1
                new_x[d] = Pi + sign*alpha*abs(mbest[d]-x_i[d])*np.log(1/u)

            new_x = _project_to_polygon(new_x, n, polygon)
            f_new = evaluate(new_x)
            swarm[i] = new_x

            if f_new < pbest_f[i]:
                pbest[i], pbest_f[i] = new_x.copy(), f_new
            if f_new < gbest_f:
                gbest, gbest_f = new_x.copy(), f_new
                if -gbest_f >= A_min:
                    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]


def qpso_pairwise_solver(problem, n, params):
    """
    QPSO variant updating each (x,y) pair jointly rather than per-coordinate.
    """
    fixed   = problem["fixed"]
    polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min   = problem["A_min"]
    d_min    = problem["d_min"]

    popsize = params["popsize"]
    maxiter = params["maxiter"]
    alpha   = params.get("alpha", 0.75)
    _seed_rng(params.get("seed"))

    names = _generate_names("A", n)

    def evaluate(vec):
        added = [(names[i], (vec[2*i], vec[2*i+1])) for i in range(n)]
        return loss_fn(added)

    # Initialize swarm inside polygon using our sampler
    swarm = []
    for i in range(popsize):
        # sample_points_within returns a list of n (x,y) tuples
        pts = sample_points_within(
            polygon=polygon,
            n=n,
            d_min=d_min,            # no extra boundary margin here
            seed=params.get("seed", None)  # optional reproducibility
        )
        # flatten [(x,y),...] → [x,y,x,y,...]
        particle = np.array([coord for pt in pts for coord in pt])
        swarm.append(particle)


    pbest   = [p.copy() for p in swarm]
    pbest_f = [evaluate(p)   for p in swarm]
    gbest_i = int(np.argmin(pbest_f))
    gbest   = pbest[gbest_i].copy()
    gbest_f = pbest_f[gbest_i]

    if -gbest_f >= A_min:
        return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    for _ in range(maxiter):
        mbest = sum(pbest) / len(pbest)
        for i, x_i in enumerate(swarm):
            new_vec = np.zeros_like(x_i)
            for j in range(n):
                u, phi = random.random(), random.random()
                Pi_x = phi*pbest[i][2*j]   + (1-phi)*gbest[2*j]
                Pi_y = phi*pbest[i][2*j+1] + (1-phi)*gbest[2*j+1]
                sign = 1 if random.random() > 0.5 else -1
                new_vec[2*j]   = Pi_x   + sign*alpha*abs(mbest[2*j]   - x_i[2*j])   * np.log(1/u)
                new_vec[2*j+1] = Pi_y   + sign*alpha*abs(mbest[2*j+1] - x_i[2*j+1]) * np.log(1/u)

            new_vec = _project_to_polygon(new_vec, n, polygon)
            f_new   = evaluate(new_vec)
            swarm[i] = new_vec

            if f_new < pbest_f[i]:
                pbest[i], pbest_f[i] = new_vec.copy(), f_new
            if f_new < gbest_f:
                gbest, gbest_f = new_vec.copy(), f_new
                if -gbest_f >= A_min:
                    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]


def ga_solver(problem, n, params):
    """
    Genetic Algorithm solver using DEAP:
      - Tournament selection
      - Two-point crossover
      - Gaussian mutation
    """
    from deap import base, creator, tools

    fixed   = problem["fixed"]
    polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min   = problem["A_min"]
    d_min    = problem["d_min"]

    popsize = params["popsize"]
    ngen    = params["ngen"]
    sigma   = params.get("sigma", 0.1)
    indpb   = params.get("indpb", 0.1)
    cxpb    = params.get("cxpb", 0.5)
    mutpb   = params.get("mutpb", 0.2)
    _seed_rng(params.get("seed"))

    # Only create these types if they don't already exist
    if not hasattr(creator, "FitnessMinGA"):
        creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualGA"):
        creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

    toolbox = base.Toolbox()
    names = _generate_names("A", n)

    def init_ind():
        # Sample `n` points uniformly inside `polygon` with no extra boundary margin
        pts = sample_points_within(
            polygon=polygon,
            n=n,
            d_min=d_min,
            seed=params.get("seed", None)
        )
        # Flatten [(x, y), …] → [x, y, x, y, …]
        coords = [coord for pt in pts for coord in pt]
        return creator.IndividualGA(coords)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(ind):
        added = [(names[i], (ind[2*i], ind[2*i+1])) for i in range(n)]
        return (loss_fn(added),)

    toolbox.register("evaluate", evaluate)

    # Initialize and evaluate
    pop = toolbox.population(n=popsize)
    for ind in pop:
        ind.fitness.values = evaluate(ind)

    # Early termination check
    best = tools.selBest(pop, 1)[0]
    if -best.fitness.values[0] >= A_min:
        return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]

    # Evolution loop
    for _ in range(ngen):
        offs = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        # Crossover & mutation
        for c1, c2 in zip(offs[::2], offs[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offs:
            if random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values

        # Evaluate and replace
        invalid = [ind for ind in offs if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = evaluate(ind)
        pop[:] = offs

        best = tools.selBest(pop, 1)[0]
        if -best.fitness.values[0] >= A_min:
            return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]

    # Final best
    best = tools.selBest(pop, 1)[0]
    return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]



def de_solver(problem, n, params):
    """
    Differential Evolution solver (SciPy).
    """
    from scipy.optimize import differential_evolution as de

    fixed   = problem["fixed"]
    polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min   = problem["A_min"]
    d_min    = problem["d_min"]

    popsize = params["popsize"]
    maxiter = params["maxiter"]
    _seed_rng(params.get("seed"))

    names = _generate_names("A", n)
    minx, miny, maxx, maxy = polygon.bounds
    bounds = [(minx + d_min, maxx - d_min), (miny + d_min, maxy - d_min)] * n

    def evaluate(x):
        added = [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]
        return loss_fn(added)

    def callback(xk, convergence=None):
        return -evaluate(xk) >= A_min

    result = de(
        func=evaluate,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=params.get("seed"),
        updating="deferred",
        callback=callback,
        polish=False
    )

    bx = result.x
    return [(names[i], (bx[2*i], bx[2*i+1])) for i in range(n)]

























def spsa_solver(problem, n, params):
    """
    SPSA solver for zero‐order optimization.
    """
    import numpy as np

    fixed   = problem["fixed"]
    polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min   = problem["A_min"]

    a       = params.get("a", 0.1)
    c       = params.get("c", 0.1)
    alpha   = params.get("alpha", 0.602)
    gamma   = params.get("gamma", 0.101)
    maxiter = params.get("maxiter", 100)
    restarts= params.get("restarts", 5)
    _seed_rng(params.get("seed"))

    names = _generate_names("A", n)
    dim   = 2 * n
    minx, miny, maxx, maxy = polygon.bounds

    def project(x):
        return _project_to_polygon(x, n, polygon)

    best_x = None
    best_f = float("inf")

    for r in range(restarts):
        # --- random init inside polygon ---
        x = []
        while len(x) < dim:
            xi = random.uniform(minx, maxx)
            yi = random.uniform(miny, maxy)
            if polygon.contains(Point(xi, yi)):
                x.extend([xi, yi])

        # Evaluate initial
        f_x = loss_fn([(names[i], (x[2*i], x[2*i+1])) for i in range(n)])

        # Initialize best_x/best_f on first restart
        if best_x is None or f_x < best_f:
            best_x, best_f = x.copy(), f_x

        # Early exit if feasible
        if -f_x >= A_min:
            return [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]

        # SPSA iterations
        for k in range(1, maxiter+1):
            a_k = a / (k**alpha)
            c_k = c / (k**gamma)
            delta = np.random.choice([-1,1], size=dim)

            x_plus  = project(np.array(x) + c_k*delta)
            f_plus  = loss_fn([(names[i], (x_plus[2*i], x_plus[2*i+1])) for i in range(n)])
            v       = (f_plus - f_x)*(x_plus - x)
            norm    = np.linalg.norm(v)
            ghat    = v / norm if norm>0 and np.isfinite(norm) else np.zeros_like(v)

            x       = project(np.array(x) - a_k*ghat)
            f_x     = loss_fn([(names[i], (x[2*i], x[2*i+1])) for i in range(n)])

            # Update best if improved
            if f_x < best_f:
                best_x, best_f = x.copy(), f_x

            # Early exit if feasible
            if -f_x >= A_min:
                return [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]

    # After all restarts, return the best found
    return [(names[i], (best_x[2*i], best_x[2*i+1])) for i in range(n)]





from scipy.optimize import minimize

__all__.append("nelder_mead_solver")


def nelder_mead_solver(problem, n, params):
    """
    Zero‐order Nelder–Mead solver for spatial point placement.
    Uses SciPy's gradient‐free 'Nelder‐Mead' method, projecting
    iterates back into the polygon after each step.
    """
    fixed    = problem["fixed"]
    polygon  = problem["polygon"]
    loss_fn  = problem["loss"]
    A_min    = problem["A_min"]

    maxiter  = params.get("maxiter", 200)
    seed     = params.get("seed")
    _seed_rng(seed)

    names = _generate_names("A", n)

    # Objective: given flat vector x of length 2n,
    # project into polygon and evaluate loss on added only.
    def obj(x):
        x_proj = _project_to_polygon(x, n, polygon)
        added  = [(names[i], (x_proj[2*i], x_proj[2*i+1])) for i in range(n)]
        return loss_fn(added)

    # Initial guess: uniform random inside polygon
    dim = 2 * n
    x0  = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(x0) < dim:
        xi = random.uniform(minx, maxx)
        yi = random.uniform(miny, maxy)
        if polygon.contains(Point(xi, yi)):
            x0.extend([xi, yi])
    x0 = np.array(x0)

    # Run Nelder–Mead
    res = minimize(
        fun=obj,
        x0=x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False}
    )

    # Project final solution into polygon
    best_x = _project_to_polygon(res.x, n, polygon)
    added  = [(names[i], (best_x[2*i], best_x[2*i+1])) for i in range(n)]

    return added
