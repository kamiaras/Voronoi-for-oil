"""
losses.py

Defines spatial loss functions for point‐placement optimization.
Each loss returns +inf if any feasibility constraint fails,
otherwise returns the negative of the minimum Voronoi cell area
(to be maximized by minimization routines).
"""

import math
from shapely.geometry import Point, MultiPoint, Polygon
from shapely import geometry as geom

__all__ = ["loss_type_1", "loss_type_2"]


def loss_type_1(
    fixed_points: list[tuple[str, tuple[float, float]]],
    added_points: list[tuple[str, tuple[float, float]]],
    polygon: Polygon,
    L2_min: float
) -> float:
    """
    Loss Type 1

    Feasibility checks:
      1) All points (fixed + added) must lie strictly inside the polygon.
      2) Every added point must be at least L2_min away from:
         - Every other added point.
         - Every fixed point.
        (Fixed–fixed distances are not checked.)

    Objective:
      Maximize the smallest Voronoi cell area among all seeds.
      Implemented as returning -min_area (so smaller loss = better).
    """
    # 1) Containment
    all_points = fixed_points + added_points
    for _, (x, y) in all_points:
        if not polygon.contains(Point(x, y)):
            return float("inf")

    # 2) Pairwise distances
    added_coords = [coord for _, coord in added_points]
    for i, (x1, y1) in enumerate(added_coords):
        # added-added
        for x2, y2 in added_coords[i+1:]:
            if math.hypot(x1 - x2, y1 - y2) < L2_min:
                return float("inf")
        # added-fixed
        for _, (xf, yf) in fixed_points:
            if math.hypot(x1 - xf, y1 - yf) < L2_min:
                return float("inf")

    # 3) Voronoi area
    coords = [coord for _, coord in all_points]
    try:
        from shapely.ops import voronoi_diagram
        vor = voronoi_diagram(MultiPoint(coords), envelope=polygon)
        areas = [region.intersection(polygon).area for region in vor.geoms]
    except ImportError:
        from scipy.spatial import Voronoi
        vor = Voronoi(coords)
        minx, miny, maxx, maxy = polygon.bounds
        bbox = geom.Polygon([(minx, miny), (minx, maxy),
                              (maxx, maxy), (maxx, miny)])
        areas = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not region or -1 in region:
                cell = bbox
            else:
                verts = [tuple(vor.vertices[i]) for i in region]
                cell = geom.Polygon(verts)
            areas.append(cell.intersection(polygon).area)

    return -min(areas)


def loss_type_2(
    fixed_points: list[tuple[str, tuple[float, float]]],
    added_points: list[tuple[str, tuple[float, float]]],
    polygon: Polygon,
    L2_min: float,
    d_min: float
) -> float:
    """
    Loss Type 2

    All checks from loss_type_1, plus:
      3) Each added point must be at least d_min away from the polygon boundary.

    Returns:
      +inf if any check fails; otherwise -min_cell_area.
    """
    # 1) Containment
    all_points = fixed_points + added_points
    for _, (x, y) in all_points:
        if not polygon.contains(Point(x, y)):
            return float("inf")

    # 2) Pairwise distances (added-added and added-fixed)
    added_coords = [coord for _, coord in added_points]
    for i, (x1, y1) in enumerate(added_coords):
        for x2, y2 in added_coords[i+1:]:
            if math.hypot(x1 - x2, y1 - y2) < L2_min:
                return float("inf")
        for _, (xf, yf) in fixed_points:
            if math.hypot(x1 - xf, y1 - yf) < L2_min:
                return float("inf")

    # 3) Boundary distance
    # Use distance to the polygon's exterior
    for _, (x, y) in added_points:
        if polygon.exterior.distance(Point(x, y)) < d_min:
            return float("inf")

    # 4) Voronoi area (same as loss_type_1)
    coords = [coord for _, coord in all_points]
    try:
        from shapely.ops import voronoi_diagram
        vor = voronoi_diagram(MultiPoint(coords), envelope=polygon)
        areas = [region.intersection(polygon).area for region in vor.geoms]
    except ImportError:
        from scipy.spatial import Voronoi
        vor = Voronoi(coords)
        minx, miny, maxx, maxy = polygon.bounds
        bbox = geom.Polygon([(minx, miny), (minx, maxy),
                              (maxx, maxy), (maxx, miny)])
        areas = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not region or -1 in region:
                cell = bbox
            else:
                verts = [tuple(vor.vertices[i]) for i in region]
                cell = geom.Polygon(verts)
            areas.append(cell.intersection(polygon).area)

    return -min(areas)
