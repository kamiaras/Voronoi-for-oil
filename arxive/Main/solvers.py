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
                p = Point((1-epsilon) * p.x + epsilon * cent.x, (1-epsilon) * p.y + epsilon * cent.y)
            x, y = p.x, p.y
            if not polygon.contains(p):
                print("aaaaaa")
        proj.extend([x, y])
    return np.array(proj)


def qpso_solver(problem, n, params):
    """
    Quantum‐behaved PSO (QPSO) for spatial point placement.

    Each particle is a 2n‐dim vector of candidate coordinates.
    Updates contract toward the mean of personal bests and global best.
    """
    fixed    = problem["fixed"]
    polygon  = problem["polygon"]
    loss_fn  = problem["loss"]
    A_min    = problem["A_min"]

    popsize  = params["popsize"]
    maxiter  = params["maxiter"]
    alpha    = params.get("alpha", 0.75)
    _seed_rng(params.get("seed"))

    # Prepare names and evaluation function
    names = _generate_names("A", n)
    def evaluate(vec):
        pts = fixed + [(names[i], (vec[2*i], vec[2*i+1])) for i in range(n)]
        return loss_fn(pts)

    # Initialize swarm inside polygon
    minx, miny, maxx, maxy = polygon.bounds
    swarm = []
    for _ in range(popsize):
        particle = []
        for _ in range(n):
            while True:
                x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
                if polygon.contains(Point(x, y)):
                    particle.extend([x, y])
                    break
        swarm.append(np.array(particle))

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
            # Update each coordinate
            for d in range(len(x_i)):
                u   = random.random()
                phi = random.random()
                Pi  = phi*pbest[i][d] + (1-phi)*gbest[d]
                sign = 1 if random.random()>0.5 else -1
                new_x[d] = Pi + sign*alpha*abs(mbest[d]-x_i[d])*np.log(1/u)

            # Project back into polygon, per (x,y) pair
            new_x = _project_to_polygon(new_x, n, polygon)
            f_new = evaluate(new_x)
            swarm[i] = new_x

            # Update personal & global bests
            if f_new < pbest_f[i]:
                pbest[i], pbest_f[i] = new_x.copy(), f_new
            if f_new < gbest_f:
                gbest, gbest_f = new_x.copy(), f_new
                if -gbest_f >= A_min:
                    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    # Return best found
    return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]


def qpso_pairwise_solver(problem, n, params):
    """
    QPSO variant updating each (x,y) pair jointly rather than per-coordinate.
    """
    fixed   = problem["fixed"]
    polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min   = problem["A_min"]

    popsize = params["popsize"]
    maxiter = params["maxiter"]
    alpha   = params.get("alpha", 0.75)
    _seed_rng(params.get("seed"))

    names = _generate_names("A", n)
    def evaluate(vec):
        pts = fixed + [(names[i], (vec[2*i], vec[2*i+1])) for i in range(n)]
        return loss_fn(pts)

    # Initialize swarm in 2n-D
    minx, miny, maxx, maxy = polygon.bounds
    swarm = []
    for _ in range(popsize):
        particle = []
        for _ in range(n):
            while True:
                x, y = random.uniform(minx, maxx), random.uniform(miny, maxy)
                if polygon.contains(Point(x, y)):
                    particle.extend([x, y])
                    break
        swarm.append(np.array(particle))

    pbest   = [p.copy() for p in swarm]
    pbest_f = [evaluate(p)   for p in swarm]
    gbest_i = int(np.argmin(pbest_f))
    gbest   = pbest[gbest_i].copy()
    gbest_f = pbest_f[gbest_i]

    if -gbest_f >= A_min:
        return [(names[i], (gbest[2*i], gbest[2*i+1])) for i in range(n)]

    # Pairwise updates per iteration
    for _ in range(maxiter):
        mbest = sum(pbest) / len(pbest)
        for i, x_i in enumerate(swarm):
            new_vec = np.zeros_like(x_i)
            for j in range(n):
                u, phi = random.random(), random.random()
                Pi_x = phi*pbest[i][2*j]   + (1-phi)*gbest[2*j]
                Pi_y = phi*pbest[i][2*j+1] + (1-phi)*gbest[2*j+1]
                sign = 1 if random.random()>0.5 else -1
                new_vec[2*j]   = Pi_x   + sign*alpha*abs(mbest[2*j]   - x_i[2*j])   * np.log(1/u)
                new_vec[2*j+1] = Pi_y + sign*alpha*abs(mbest[2*j+1] - x_i[2*j+1]) * np.log(1/u)

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

    popsize = params["popsize"]
    ngen    = params["ngen"]
    sigma   = params.get("sigma", 0.1)
    indpb   = params.get("indpb", 0.1)
    cxpb    = params.get("cxpb", 0.5)
    mutpb   = params.get("mutpb", 0.2)
    _seed_rng(params.get("seed"))

    # Set up DEAP types once
    try:
        creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
        creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)
    except Exception:
        pass

    toolbox = base.Toolbox()
    names = _generate_names("A", n)

    # Individual initializer: random inside polygon
    bounds = polygon.bounds
    def init_ind():
        coords = []
        while len(coords) < 2*n:
            x = random.uniform(bounds[0], bounds[2])
            y = random.uniform(bounds[1], bounds[3])
            if polygon.contains(Point(x, y)):
                coords.extend([x, y])
        return creator.IndividualGA(coords)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(ind):
        pts = fixed + [(names[i], (ind[2*i], ind[2*i+1])) for i in range(n)]
        return (loss_fn(pts),)

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
    Differential Evolution solver (SciPy):
    Early exit when min cell area ≥ A_min.
    """
    from scipy.optimize import differential_evolution as de

    fixed   = problem["fixed"]
    polygon = problem["polygon"]
    loss_fn = problem["loss"]
    A_min   = problem["A_min"]

    popsize = params["popsize"]
    maxiter = params["maxiter"]
    _seed_rng(params.get("seed"))

    names = _generate_names("A", n)
    # Build bounds [(minx, maxx),(miny,maxy)] * n
    minx, miny, maxx, maxy = polygon.bounds
    bounds = [(minx, maxx), (miny, maxy)] * n

    def evaluate(x):
        pts = fixed + [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]
        return loss_fn(pts)

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
    SPSA solver for zero‐order optimization:
    One‐sided directional approximation and multiple restarts.
    Early exit when min cell area ≥ A_min.
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

    # random init inside polygon
    x = []
    while len(x) < dim:
        xi = random.uniform(minx, maxx)
        yi = random.uniform(miny, maxy)
        if polygon.contains(Point(xi, yi)):
            x.extend([xi, yi])
    best = x.copy()
    best_f = loss_fn(fixed + [(names[i], (x[2*i], x[2*i+1])) for i in range(n)])
    
    for _ in range(restarts):
        # random init inside polygon
        x = []
        while len(x) < dim:
            xi = random.uniform(minx, maxx)
            yi = random.uniform(miny, maxy)
            if polygon.contains(Point(xi, yi)):
                x.extend([xi, yi])

        f_x = loss_fn(fixed + [(names[i], (x[2*i], x[2*i+1])) for i in range(n)])
        if -f_x >= A_min:
            return [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]
        if f_x < best_f:
            best, best_f = x.copy(), f_x

        for k in range(1, maxiter+1):
            a_k = a / (k**alpha)
            c_k = c / (k**gamma)
            delta = np.random.choice([-1,1], size=dim)

            x_plus  = project(x + c_k*delta)
            f_plus  = loss_fn(fixed + [(names[i], (x_plus[2*i], x_plus[2*i+1])) for i in range(n)])
            v       = (f_plus - f_x)*(x_plus - x)
            norm    = np.linalg.norm(v)
            if norm <= 0 or not np.isfinite(norm):
                ghat = np.zeros_like(v)
            else:
                ghat = v / norm
            x       = project(x - a_k*ghat)
            f_x     = loss_fn(fixed + [(names[i], (x[2*i], x[2*i+1])) for i in range(n)])

            if -f_x >= A_min:
                return [(names[i], (x[2*i], x[2*i+1])) for i in range(n)]
            if f_x < best_f:
                best, best_f = x.copy(), f_x
    return [(names[i], (best[2*i], best[2*i+1])) for i in range(n)]
