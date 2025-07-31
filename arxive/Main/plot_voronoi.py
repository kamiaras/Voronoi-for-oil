import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint, Polygon
from scipy.spatial import Voronoi
try:
    from shapely.ops import voronoi_diagram
except ImportError:
    voronoi_diagram = None

def plot_voronoi(fixed, added, polygon, solver):
    """
    Render the clipped Voronoi diagram for two sets of seeds.

    Parameters:
    -----------
    fixed    : list of (name,(x,y))
               Pre‐existing seed points (plotted in red).
    added    : list of (name,(x,y))
               Newly placed seed points (plotted in blue).
    polygon  : shapely.geometry.Polygon
               Convex hull to clip Voronoi cells.
    solver   : str
               Identifier of the solver used (shown in the title).

    Returns:
    --------
    cells     : list of shapely.geometry.Polygon
                The Voronoi cells clipped to `polygon`.
    area_list : list of (name, area)
                Each seed’s corresponding cell area.
    """
    # --- 1) Extract names & raw coordinates ---
    names, coords = zip(*(fixed + added)) if fixed or added else ([], [])
    coords = list(coords)

    # --- 2) Compute Voronoi cells & areas ---
    cells = []
    area_list = []
    if voronoi_diagram:
        # Shapely 2.0+ optimized path
        vor = voronoi_diagram(MultiPoint(coords), envelope=polygon)
        for name, region in zip(names, vor.geoms):
            cell = region.intersection(polygon)
            cells.append(cell)
            area_list.append((name, cell.area))
    else:
        # SciPy fallback
        vor = Voronoi(np.array(coords))
        minx, miny, maxx, maxy = polygon.bounds
        bbox = Polygon([(minx,miny),(minx,maxy),(maxx,maxy),(maxx,miny)])
        for idx, name in enumerate(names):
            region = vor.regions[vor.point_region[idx]]
            if not region or -1 in region:
                cell = bbox
            else:
                cell = Polygon([vor.vertices[i] for i in region])
            clipped = cell.intersection(polygon)
            cells.append(clipped)
            area_list.append((name, clipped.area))

    # --- 3) Compute summary statistics ---
    n_added  = len(added)
    min_area = min(a for _, a in area_list) if area_list else 0.0

    # Compute minimum pairwise distance among ALL seeds
    min_dist = np.inf
    for i, (x1,y1) in enumerate(coords):
        for x2,y2 in coords[i+1:]:
            d = np.hypot(x1-x2, y1-y2)
            if d < min_dist:
                min_dist = d
    if len(coords) < 2:
        min_dist = 0.0

    # --- 4) Plotting ---
    fig, ax = plt.subplots()
    # Draw polygon boundary
    x_poly, y_poly = polygon.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1.5)

    # Shade each cell
    for cell in cells:
        if cell.is_empty:
            continue
        if cell.geom_type == 'Polygon':
            x, y = cell.exterior.xy
            ax.fill(x, y, alpha=0.4)
        else:
            for part in cell.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, alpha=0.4)

    # Plot fixed seeds in red, with labels
    for name, (x, y) in fixed:
        ax.scatter(x, y, color='red')
        ax.text(x, y, name, fontsize=8, ha='right', va='bottom')
    # Plot added seeds in blue, with labels
    for name, (x, y) in added:
        ax.scatter(x, y, color='blue')
        ax.text(x, y, name, fontsize=8, ha='right', va='bottom')

    # Build legend manually
    ax.scatter([], [], color='red', label='fixed')
    ax.scatter([], [], color='blue', label='added')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # --- 5) Title & console output ---
    ax.set_title(
        f"Solver: {solver} | n={n_added} | "
        f"Min area={min_area:.3g} | Min dist={min_dist:.3g}"
    )

    print("Voronoi cell areas:")
    for name, area in area_list:
        print(f"  {name}: {area:.3g}")

    plt.show()
    return cells, area_list
