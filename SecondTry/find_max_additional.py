# find_max_additional.py

"""
Module to find the maximum number of additional points that satisfy
given feasibility constraints, via exponential + binary search.
"""

# Replace this import with wherever your is_feasible function lives:
from your_feasibility_module import is_feasible  

def find_max_additional(problem: dict, config: dict):
    """
    Finds the maximum number of additional points that satisfy feasibility constraints.

    Args:
        problem (dict): Must contain keys:
            - 'fixed': list of existing seed points
            - 'polygon': the convex hull polygon
            - 'A_min': minimum area threshold (float)
            - 'l2_min': minimum L2‐distance threshold (float)
        config (dict): Must contain keys:
            - 'name': solver name (str)
            - 'params': dict of solver-specific parameters

    Returns:
        (max_count, best_added):
            max_count (int): largest N for which is_feasible(N,problem,config) is True
            best_added (list): the corresponding list of added points
    """
    # check N=0
    ok, _ = is_feasible(0, problem, config)
    if not ok:
        return 0, []

    # exponential search for an infeasible upper bound
    low, n = 0, 1
    best_added = []
    while True:
        ok, added = is_feasible(n, problem, config)
        if not ok:
            break
        low, best_added = n, added
        n *= 2

    print(f"Binary search between n={low} (feasible) and n={n} (infeasible)")

    # binary search in (low, n)
    lo, hi = low, n
    while lo < hi - 1:
        mid = (lo + hi) // 2
        ok, added = is_feasible(mid, problem, config)
        if ok:
            lo, best_added = mid, added
        else:
            hi = mid

    print(f"Result → Max extra points: {lo}")
    return lo, best_added