import solvers

__all__ = ["is_feasible"]


def is_feasible(n, problem: dict, config: dict):
    """
    Check whether adding `n` new points can meet the Voronoi feasibility criteria.

    Feasibility is defined by:
      - Every Voronoi cell has area ≥ A_min.
      - (Optionally) Other constraints encoded by the loss function.

    Parameters:
    -----------
    n : int
        Number of new points to add.
    problem : dict
        - 'fixed':   list of (name,(x,y)) for existing seed points
        - 'polygon': Shapely Polygon defining the domain
        - 'A_min':   minimum cell area threshold
        - 'loss':    callable(points_list) → float
                     returns –(min_cell_area) or +inf if any constraint fails
    config : dict
        - 'name':   solver identifier string (e.g. 'qpso', 'ga', 'de', 'spsa')
        - 'params': dict of solver‐specific parameters

    Returns:
    --------
    ok : bool
        True if a configuration of size `n` meets A_min (and any other encoded constraints).
    new_pts : list
        When ok is True and n > 0, a list of newly added (name,(x,y)) points.
        Otherwise, an empty list.
    """
    # Unpack fixed seeds and common thresholds
    fixed   = problem["fixed"]
    A_min   = problem["A_min"]
    loss_fn = problem["loss"]

    # 1) Baseline check (no new points)
    if n == 0:
        score = loss_fn(fixed)
        # +inf signals infeasible; score = –min_area otherwise
        return (score != float("inf") and -score >= A_min), []

    # 2) Generate candidate points with the chosen solver
    solver = config.get("name", "")
    params = config.get("params", {})

    if solver == "qpso":
        new_pts = solvers.qpso_solver(problem, n, params)
    elif solver == "qpso_pairwise":
        new_pts = solvers.qpso_pairwise_solver(problem, n, params)
    elif solver == "ga":
        new_pts = solvers.ga_solver(problem, n, params)
    elif solver == "de":
        new_pts = solvers.de_solver(problem, n, params)
    elif solver == "spsa":
        new_pts = solvers.spsa_solver(problem, n, params)
    else:
        raise NotImplementedError(f"Solver '{solver}' is not supported")

    # 3) Evaluate the combined configuration
    score = loss_fn(fixed + new_pts)
    if score == float("inf"):
        # Constraint violation → infeasible
        return False, []

    # Positive feasibility if min_area ≥ A_min
    min_area = -score
    return (min_area >= A_min), new_pts
