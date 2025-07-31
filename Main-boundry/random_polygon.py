import random
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from typing import Tuple, List

def generate_random_convex_polygon_with_fixed(
    rand_seed: int = 30,
    convexhull_param: int = 30,
    num_fixed: int = 5
) -> Tuple[Polygon, List[Tuple[str, Tuple[float, float]]]]:
    """
    Generate a random convex polygon and a set of named interior seed points.

    Args:
        rand_seed:         Seed for Python and NumPy RNGs (ensures reproducibility).
        convexhull_param:  Number of random points used to form the convex hull.
        num_fixed:         Number of interior seed points to sample.

    Returns:
        polygon:           A Shapely Polygon representing the convex hull.
        fixed:             A list of (name, (x, y)) for interior seeds labeled F_1, F_2, ….

    Raises:
        TypeError:         If any argument is of the wrong type.
        ValueError:        If any argument is not positive.
        RuntimeError:      If the convex hull is not a Polygon.
    """
    # --- Type checks ---
    if not isinstance(rand_seed, int):
        raise TypeError(f"rand_seed must be int, got {type(rand_seed).__name__}")
    if not isinstance(convexhull_param, int):
        raise TypeError(f"convexhull_param must be int, got {type(convexhull_param).__name__}")
    if not isinstance(num_fixed, int):
        raise TypeError(f"num_fixed must be int, got {type(num_fixed).__name__}")

    # --- Value checks ---
    if rand_seed < 0:
        raise ValueError("rand_seed must be non-negative")
    if convexhull_param <= 0:
        raise ValueError("convexhull_param must be positive")
    if num_fixed < 0:
        raise ValueError("num_fixed must be non-negative")

    # Seed RNGs
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    # Generate convex hull
    raw_pts = np.random.rand(convexhull_param, 2)
    geom_hull = MultiPoint(raw_pts).convex_hull

    # Ensure it's a Polygon, not e.g. a Point or LineString
    if not isinstance(geom_hull, Polygon):
        raise RuntimeError(f"Expected convex hull to be a Polygon, got {type(geom_hull).__name__}")
    polygon: Polygon = geom_hull  # type-guarded cast

    # Sample interior points
    minx, miny, maxx, maxy = polygon.bounds
    fixed: List[Tuple[str, Tuple[float, float]]] = []
    label_index = 1

    while len(fixed) < num_fixed:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if not polygon.contains(Point(x, y)):
            continue
        name = f"F_{label_index}"
        fixed.append((name, (x, y)))
        label_index += 1

    return polygon, fixed

__all__ = ["generate_random_convex_polygon_with_fixed"]
