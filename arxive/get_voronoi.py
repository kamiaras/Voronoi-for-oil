# get_voronoi.py

"""
Module: get_voronoi

Compute and clip Voronoi cells for named points within a given polygon,
returning each point’s name, its cell, and the cell’s area, plus
the list of pairwise point-to-point distances.
"""

from shapely.geometry import MultiPoint
import shapely.geometry as geom
import itertools
import math

__all__ = ["get_voronoi_cells_and_areas"]


def get_voronoi_cells_and_areas(points, polygon):
    """
    Given a list of named points and a Shapely polygon, compute:
      1) The Voronoi diagram clipped to `polygon`, returning a list of
         (name, cell_polygon, area)
      2) The list of distances between every pair of points.

    Parameters:
    -----------
    points : list of (name, (x, y))
        The site points, each with a unique string `name`.
    polygon : shapely.geometry.Polygon
        The bounding polygon to which each Voronoi cell will be clipped.

    Returns:
    --------
    cells_and_areas : list of tuples
        Each element is (name, shapely.geometry.Polygon cell, float area).
    distances : list of tuples
        Each element is (name1, name2, float distance).
    """
    # Unzip names and coordinates
    names, coords = zip(*points)

    # Compute all pairwise distances
    distances = []
    for (n1, (x1, y1)), (n2, (x2, y2)) in itertools.combinations(points, 2):
        d = math.hypot(x2 - x1, y2 - y1)
        distances.append((n1, n2, d))

    multipoint = MultiPoint(coords)

    # Try Shapely 2.0+ voronoi_diagram
    try:
        from shapely.ops import voronoi_diagram
        vor = voronoi_diagram(multipoint, envelope=polygon)
        cells_and_areas = []
        for name, region in zip(names, vor.geoms):
            cell = region.intersection(polygon)
            cells_and_areas.append((name, cell, cell.area))
        return cells_and_areas, distances

    except ImportError:
        # SciPy fallback
        from scipy.spatial import Voronoi
        vor = Voronoi(list(coords))

        # Build a bounding box to approximate infinite regions
        minx, miny, maxx, maxy = polygon.bounds
        bbox = geom.Polygon([
            (minx, miny), (minx, maxy),
            (maxx, maxy), (maxx, miny)
        ])

        cells_and_areas = []
        for idx, name in enumerate(names):
            region_index = vor.point_region[idx]
            vertex_indices = vor.regions[region_index]

            if not vertex_indices or -1 in vertex_indices:
                # Infinite region: use bounding box
                cell_poly = bbox
            else:
                pts = [tuple(vor.vertices[i]) for i in vertex_indices]
                cell_poly = geom.Polygon(pts)

            cell = cell_poly.intersection(polygon)
            cells_and_areas.append((name, cell, cell.area))

        return cells_and_areas, distances


if __name__ == "__main__":
    # Example usage
    from shapely.geometry import Polygon

    # Define a square polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Named sample points
    pts = [
        ("A01", (0.2, 0.3)),
        ("B02", (0.8, 0.5)),
        ("C03", (0.4, 0.8)),
    ]

    cells, dists = get_voronoi_cells_and_areas(pts, poly)

    print("Voronoi cells & areas:")
    for name, cell, area in cells:
        print(f"  {name}: area = {area:.4f}")

    print("\nPairwise distances:")
    for n1, n2, dist in dists:
        print(f"  {n1} ↔ {n2}: {dist:.4f}")
