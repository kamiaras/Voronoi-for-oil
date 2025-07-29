"""
losses.py

Defines spatial loss functions for point‐placement optimization.
Each loss returns +inf if any feasibility constraint fails,
otherwise returns the negative of the minimum Voronoi cell area
(to be maximized by minimization routines).
"""

import math
from shapely.geometry import Point, MultiPoint
import shapely.geometry as geom

__all__ = ["loss_type_1"]


def loss_type_1(points, polygon, L2_min):
    """
    Loss Type 1

    Feasibility checks:
      1) All points must lie strictly inside the polygon.
      2) Every pair of points must be at least L2_min apart.

    Objective:
      Maximize the smallest Voronoi cell area among all seeds.
      Implemented as returning -min_area (so smaller loss = better).

    Parameters:
    -----------
    points : list of (name, (x, y))
        Seed points with identifiers.
    polygon : shapely.geometry.Polygon
        Convex hull or other region to clip Voronoi cells.
    L2_min : float
        Minimum allowed Euclidean distance between any two seeds.

    Returns:
    --------
    float
        +inf if any feasibility check fails;
        otherwise the negative of the smallest clipped Voronoi cell area.
    """
    # --- 1) Extract raw coordinates ---
    coords = [coord for _, coord in points]

    # --- 2) Check that all points lie inside the polygon ---
    for x, y in coords:
        if not polygon.contains(Point(x, y)):
            return float("inf")

    # --- 3) Enforce minimum pairwise distance constraint ---
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            if math.hypot(x1 - x2, y1 - y2) < L2_min:
                return float("inf")

    # --- 4) Compute Voronoi cell areas ---
    # Try Shapely 2.0's voronoi_diagram for speed; fallback to SciPy.
    try:
        from shapely.ops import voronoi_diagram
        multipoint = MultiPoint(coords)
        vor = voronoi_diagram(multipoint, envelope=polygon)
        areas = [region.intersection(polygon).area for region in vor.geoms]

    except ImportError:
        from scipy.spatial import Voronoi
        vor = Voronoi(coords)

        # Build bounding box polygon for infinite regions
        minx, miny, maxx, maxy = polygon.bounds
        bbox = geom.Polygon([
            (minx, miny), (minx, maxy),
            (maxx, maxy), (maxx, miny)
        ])

        areas = []
        for idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            # If region is unbounded or empty, use bounding box
            if not region or -1 in region:
                cell = bbox
            else:
                vertices = [tuple(vor.vertices[i]) for i in region]
                cell = geom.Polygon(vertices)
            # Clip to the original polygon
            areas.append(cell.intersection(polygon).area)

    # --- 5) Return the negative of the smallest area for minimization ---
    return -min(areas)
