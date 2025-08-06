"""
Module: solvers

Population-based optimizers for placing new seed points inside a convex polygon
under Voronoi-based feasibility (minimum cell area, optional minimum pairwise distance).
Each solver returns only the newly added (name, (x, y)) points.
"""

import random
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

__all__ = [
    "qpso_pairwise_solver",
    "ga_deap_solver",
    "de_scipy_solver",
    "_sample_points_within",
    "_project_to_polygon",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers (not exported)
# ─────────────────────────────────────────────────────────────────────────────
def _seed_rng(seed: Optional[int]) -> None:
    """Seed Python & NumPy RNGs for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _generate_names(prefix: str, n: int) -> List[str]:
    """Generate identifiers PREFIX_1 … PREFIX_n."""
    return [f"{prefix}_{i+1}" for i in range(n)]


def _project_to_polygon(vec: np.ndarray, n: int, polygon: Polygon) -> np.ndarray:
    """
    Snap any out-of-polygon (x, y) pairs in `vec` back into `polygon`.
    Uses nearest_points to project, ensures feasibility.
    """
    proj: List[float] = []
    for i in range(n):
        x, y = vec[2*i], vec[2*i+1]
        pt = Point(x, y)
        if not polygon.contains(pt):
            pt = nearest_points(polygon, pt)[0]
        proj.extend([pt.x, pt.y])
    return np.array(proj)


def _sample_points_within(
    polygon: Polygon,
    n: int,
    d_min: float
) -> List[Tuple[float, float]]:
    """
    Uniformly sample `n` points inside `polygon` with a boundary margin `d_min`.
    Retries until successful or raises after many attempts.
    """
    if not isinstance(polygon, Polygon):
        raise TypeError("`polygon` must be a Shapely Polygon")
    if n < 0 or d_min < 0:
        raise ValueError("`n` and `d_min` must be non-negative")

    minx, miny, maxx, maxy = polygon.bounds
    low_x, high_x = minx + d_min, maxx - d_min
    low_y, high_y = miny + d_min, maxy - d_min
    if low_x >= high_x or low_y >= high_y:
        raise ValueError("`d_min` too large for polygon dimensions")

    pts: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(10000, n * 1000)
    while len(pts) < n:
        if attempts > max_attempts:
            raise RuntimeError(f"Sampling {n} points failed after {attempts} tries")
        attempts += 1
        x = random.uniform(low_x, high_x)
        y = random.uniform(low_y, high_y)
        p = Point(x, y)
        if polygon.contains(p):
            pts.append((x, y))
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Solver: QPSO – pairwise coordinate updates
# ─────────────────────────────────────────────────────────────────────────────
def qpso_pairwise_solver(
    problem: Dict[str, Any],
    n: int,
    params: Dict[str, Any]
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Quantum-behaved PSO updating each (x, y) jointly.
    Initializes swarm via uniform sampling with optional minimum distance.

    popsize: relative multiplier for swarm size; actual swarm size = popsize * (2*n)
    """
    _seed_rng(params.get("seed"))

    polygon: Polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min: float = problem["A_min"]
    d_min: float = problem.get("d_min", 0.0)

    # interpret popsize as relative multiplier
    rel_popsize: int = params["popsize"]
    dim = 2 * n
    popsize: int = int(rel_popsize * dim)

    maxiter: int = params["maxiter"]
    alpha: float = params.get("alpha", 0.75)

    names = _generate_names("A", n)

    def evaluate(vec: np.ndarray) -> float:
        pts = [(names[i], (vec[2*i], vec[2*i+1])) for i in range(n)]
        return loss_fn(pts)

    # build initial swarm
    swarm = [
        np.array([c for pt in _sample_points_within(polygon, n, d_min) for c in pt])
        for _ in range(popsize)
    ]

    pbest = [p.copy() for p in swarm]
    pbest_f = [evaluate(p) for p in pbest]
    idx = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[idx].copy(), pbest_f[idx]

    if -gbest_f >= A_min:
        return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    for _ in range(maxiter):
        mbest = sum(pbest) / len(pbest)
        for i, x_i in enumerate(swarm):
            new_vec = np.zeros_like(x_i)
            for j in range(n):
                u, phi = random.random(), random.random()
                px, py = pbest[i][2*j:2*j+2]
                gx, gy = gbest[2*j:2*j+2]
                for d, (p_d, g_d) in enumerate(((px, gx), (py, gy))):
                    Pi = phi * p_d + (1 - phi) * g_d
                    sign = 1 if random.random() > 0.5 else -1
                    new_vec[2*j+d] = Pi + sign * alpha * abs(mbest[2*j+d] - x_i[2*j+d]) * np.log(1/u)

            new_vec = _project_to_polygon(new_vec, n, polygon)
            f_new = evaluate(new_vec)
            swarm[i] = new_vec

            if f_new < pbest_f[i]:
                pbest[i], pbest_f[i] = new_vec.copy(), f_new
            if f_new < gbest_f:
                gbest, gbest_f = new_vec.copy(), f_new
                if -gbest_f >= A_min:
                    return [(names[k], (gbest[2*k], gbest[2*k+1])) for k in range(n)]

    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Solver: Genetic Algorithm via DEAP
# ─────────────────────────────────────────────────────────────────────────────
def ga_deap_solver(
    problem: Dict[str, Any],
    n: int,
    params: Dict[str, Any]
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    GA with tournament selection, two-point crossover, Gaussian mutation.
    Individuals are initialized via uniform sampling inside polygon.

    popsize: relative multiplier for population size; actual population size = popsize * (2*n)
    """
    from deap import base, creator, tools

    _seed_rng(params.get("seed"))

    polygon: Polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min: float = problem["A_min"]
    d_min: float = problem.get("d_min", 0.0)

    # interpret popsize as relative multiplier
    rel_popsize: int = params["popsize"]
    dim = 2 * n
    popsize: int = int(rel_popsize * dim)

    ngen: int = params["ngen"]
    sigma: float = params.get("sigma", 0.1)
    indpb: float = params.get("indpb", 0.1)
    cxpb: float = params.get("cxpb", 0.5)
    mutpb: float = params.get("mutpb", 0.2)

    # register DEAP types once
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    names = _generate_names("A", n)

    def init_ind() -> Any:
        pts = _sample_points_within(polygon, n, d_min)
        flat = [c for pt in pts for c in pt]
        return creator.Individual(flat)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register(
        "evaluate",
        lambda ind: (loss_fn([(names[i], (ind[2*i], ind[2*i+1])) for i in range(n)]),)
    )

    pop = toolbox.population(n=popsize)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    best = tools.selBest(pop, 1)[0]
    if -best.fitness.values[0] >= A_min:
        return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]

    for _ in range(ngen):
        offs = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offs[::2], offs[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offs:
            if random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offs if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offs

        best = tools.selBest(pop, 1)[0]
        if -best.fitness.values[0] >= A_min:
            return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]

    best = tools.selBest(pop, 1)[0]
    return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Solver: Differential Evolution via SciPy
# ─────────────────────────────────────────────────────────────────────────────
def de_scipy_solver(
    problem: Dict[str, Any],
    n: int,
    params: Dict[str, Any]
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    SciPy’s differential_evolution with an early-exit callback on A_min.
    """
    from scipy.optimize import differential_evolution as de

    _seed_rng(params.get("seed"))

    polygon: Polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min: float = problem["A_min"]
    d_min: float = problem.get("d_min", 0.0)
    popsize: int = params["popsize"]    # here popsize is the relative multiplier
    maxiter: int = params["maxiter"]

    names = _generate_names("A", n)
    minx, miny, maxx, maxy = polygon.bounds
    bounds = [(minx + d_min, maxx - d_min), (miny + d_min, maxy - d_min)] * n

    def evaluate(x: np.ndarray) -> float:
        pts = [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]
        return loss_fn(pts)

    def callback(xk: np.ndarray, convergence: Optional[float] = None) -> bool:
        return -evaluate(xk) >= A_min

    res = de(
        func=evaluate,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=params.get("seed"),
        updating="deferred",
        callback=callback,
        polish=False
    )
    bx = res.x
    return [(names[i], (bx[2*i], bx[2*i+1])) for i in range(n)]
