import numpy as np
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def get_voronoi_cells_and_areas(points, polygon):
    """
    Given a list of points and a convex Shapely polygon, compute the Voronoi
    diagram (using Shapely >=2.0 if available, otherwise SciPy fallback),
    clip each cell to `polygon`, and return (cells, areas).
    """
    multipoint = MultiPoint(points)
    try:
        # Shapely 2.0+
        from shapely.ops import voronoi_diagram
        vor = voronoi_diagram(multipoint, envelope=polygon)
        cells, areas = [], []
        for region in vor.geoms:
            clipped = region.intersection(polygon)
            cells.append(clipped)
            areas.append(clipped.area)
        return cells, areas
    except ImportError:
        # SciPy fallback
        pts = np.array(points)
        vor = Voronoi(pts)
        cells, areas = [], []
        for pt_i, reg_i in enumerate(vor.point_region):
            region = vor.regions[reg_i]
            if region is None or -1 in region or len(region) == 0:
                continue
            coords = [vor.vertices[i] for i in region]
            poly = Polygon(coords)
            clipped = poly.intersection(polygon)
            if not clipped.is_empty:
                cells.append(clipped)
                areas.append(clipped.area)
        return cells, areas

def plot_voronoi(fixed, added, polygon):
    """
    Plot the Voronoi diagram of (fixed + added) points clipped to polygon.
    - boundary in black
    - cells shaded
    - fixed seeds in red, added seeds in blue
    Prints areas (3 significant digits) before showing.
    """
    all_pts = fixed + added
    cells, areas = get_voronoi_cells_and_areas(all_pts, polygon)

    fig, ax = plt.subplots()
    # polygon outline
    x_poly, y_poly = polygon.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1.5)

    # fill cells
    for cell in cells:
        if cell.is_empty:
            continue
        if cell.geom_type == 'Polygon':
            x, y = cell.exterior.xy
            ax.fill(x, y, alpha=0.4)
        else:
            for sub in cell.geoms:
                if sub.geom_type == 'Polygon':
                    x, y = sub.exterior.xy
                    ax.fill(x, y, alpha=0.4)

    # scatter seeds
    fx = [p[0] for p in fixed]; fy = [p[1] for p in fixed]
    ax.scatter(fx, fy, color='red', label='fixed')
    if added:
        ax.scatter([p[0] for p in added], [p[1] for p in added],
                   color='blue', label='added')
    ax.legend()
    ax.set_aspect('equal')

    print("Voronoi cell areas:", [f"{a:.3g}" for a in areas])
    plt.show()

    return cells, areas
