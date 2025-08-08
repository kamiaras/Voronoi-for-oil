from is_feasible import is_feasible
from plot_voronoi import plot_voronoi

__all__ = ["find_max_additional"]


def find_max_additional(problem: dict, config: dict, plot_each: bool = False):
    """
    Determine the largest number of additional points that satisfy
    Voronoi‐based feasibility (area & distance constraints).

    Uses:
      1) Exponential search to bracket an infeasible N.
      2) Binary search within that bracket to find the exact maximum N.

    Args:
        problem: dict with keys:
            - 'fixed':    list of (name,(x,y)) fixed seed points
            - 'polygon':  Shapely Polygon (convex hull)
            - 'A_min':    minimum allowed Voronoi cell area
            - 'l2_min':   minimum allowed pairwise distance
            - 'loss':     callable pts_list -> float (–min_area or inf)
        config: dict with keys:
            - 'name':   solver identifier (e.g. 'qpso', 'ga', etc.)
            - 'params': solver‐specific parameters
        plot_each: if True, plot each feasible configuration

    Returns:
        max_count:   the largest feasible number of added points
        best_added:  list of (name,(x,y)) for that configuration
    """
    fixed = problem["fixed"]
    polygon = problem["polygon"]
    solver_name = config.get("name", "")

    # 1) Baseline feasibility: no new points
    ok, _ = is_feasible(0, problem, config)
    if not ok:
        print("Even the baseline (no extra points) is infeasible.")
        return 0, []

    low = 0
    best_added = []

    # 2) Exponential search: double N until infeasible
    n = 1
    while True:
        ok, added = is_feasible(n, problem, config)
        if not ok:
            print(f"n={n} is infeasible.")
            break

        low = n
        best_added = added
        if plot_each:
            print(f"n={n} is feasible; plotting.")
            plot_voronoi(fixed, added, polygon, solver_name)

        n *= 2

    print(f"Binary search between feasible n={low} and infeasible n={n}")

    # 3) Binary search between low and n
    lo, hi = low, n
    while lo < hi - 1:
        mid = (lo + hi) // 2
        ok, added = is_feasible(mid, problem, config)
        if ok:
            lo = mid
            best_added = added
            if plot_each:
                print(f"n={mid} is feasible; plotting.")
                plot_voronoi(fixed, added, polygon, solver_name)
        else:
            print(f"n={mid} is infeasible.")
            hi = mid

    # 4) Result
    print(f"Result → Max extra points: {lo}")
    return lo, best_added
