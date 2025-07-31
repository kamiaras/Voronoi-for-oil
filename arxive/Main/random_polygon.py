import random
import numpy as np
from shapely.geometry import MultiPoint, Point

def generate_random_convex_polygon_with_fixed(
    rand_seed: int = 30,
    convexhull_param: int = 30,
    num_fixed: int = 5
):
    """
    Generate a random convex polygon and a set of named interior seed points.

    Args:
        rand_seed:      Seed for Python and NumPy RNGs (ensures reproducibility).
        convexhull_param: Number of random points used to form the convex hull.
        num_fixed:      Number of interior seed points to sample.

    Returns:
        polygon:        A Shapely Polygon representing the convex hull.
        fixed:          A list of (name, (x, y)) for interior seeds labeled F_1, F_2, ….
    """
    # 1) Seed the RNGs for repeatable output
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    # 2) Generate random points and compute their convex hull
    raw_pts = np.random.rand(convexhull_param, 2)
    polygon = MultiPoint(raw_pts).convex_hull

    # 3) Sample interior points:
    #    • Draw uniformly from the polygon’s bounding box
    #    • Keep only those contained within the polygon
    minx, miny, maxx, maxy = polygon.bounds
    fixed = []
    label_index = 1

    while len(fixed) < num_fixed:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if not polygon.contains(Point(x, y)):
            continue

        # Name points sequentially: F_1, F_2, …
        name = f"F_{label_index}"
        fixed.append((name, (x, y)))
        label_index += 1

    return polygon, fixed

__all__ = ["generate_random_convex_polygon_with_fixed"]
