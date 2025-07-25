# get_voronoi.py

"""
Module: get_voronoi

Compute and clip Voronoi cells for named points within a given polygon,
returning each point’s name, its cell, and the cell’s area.
"""

from shapely.geometry import MultiPoint
import shapely.geometry as geom

__all__ = ["get_voronoi_cells_and_areas"]


def get_voronoi_cells_and_areas(points, polygon):
    """
    Given a list of named points and a Shapely polygon, compute the Voronoi
    diagram, clip each cell to `polygon`, and return a list of
    (name, cell_polygon, area).

    Parameters:
    -----------
    points : list of (name, (x, y))
        The site points, each with a unique string `name`.
    polygon : shapely.geometry.Polygon
        The bounding polygon to which each Voronoi cell will be clipped.

    Returns:
    --------
    results : list of tuples
        Each element is (name, shapely.geometry.Polygon cell, float area).
    """
    # Unzip names and coordinates
    names, coords = zip(*points)
    multipoint = MultiPoint(coords)

    # Try Shapely 2.0+ voronoi_diagram
    try:
        from shapely.ops import voronoi_diagram
        vor = voronoi_diagram(multipoint, envelope=polygon)
        results = []
        for name, region in zip(names, vor.geoms):
            cell = region.intersection(polygon)
            results.append((name, cell, cell.area))
        return results

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

        results = []
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
            results.append((name, cell, cell.area))

        return results


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

    cells = get_voronoi_cells_and_areas(pts, poly)
    for name, cell, area in cells:
        print(f"{name}: area = {area:.4f}")
