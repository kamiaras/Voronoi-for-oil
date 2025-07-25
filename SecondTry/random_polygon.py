import random
import string
import numpy as np
from shapely.geometry import MultiPoint, Point

def generate_random_convex_polygon_with_fixed(
    rand_seed: int = 30,
    convexhullparam: int = 30,
    numpoints: int = 5
):
    """
    Generates a random convex polygon and a set of named fixed interior seed points.

    Parameters:
    rand_seed (int): Seed for random number generators (default: 30).
    convexhullparam (int): Number of random points to generate before computing the convex hull (default: 30).
    numpoints (int): Number of fixed interior seed points to sample (default: 5).

    Returns:
    polygon (shapely.geometry.Polygon): The convex hull polygon.
    fixed (list of tuple): List of (name, (x, y)) for fixed seed points inside the polygon.
    """
    # Seed the random number generators
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    # Create random convex polygon
    pts = np.random.rand(convexhullparam, 2)
    polygon = MultiPoint([tuple(p) for p in pts]).convex_hull

    # Sample fixed interior points with unique 3-character names
    fixed = []
    used_names = set()
    minx, miny, maxx, maxy = polygon.bounds
    while len(fixed) < numpoints:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if not polygon.contains(Point(x, y)):
            continue
        # Generate a unique 3-character name
        while True:
            name = ''.join(random.choices(string.ascii_uppercase, k=3))
            if name not in used_names:
                used_names.add(name)
                break
        fixed.append((name, (x, y)))

    return polygon, fixed

__all__ = ["generate_random_convex_polygon_with_fixed"]


