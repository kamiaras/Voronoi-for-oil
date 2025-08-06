import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import voronoi_diagram

def plot_voronoi(fixed, added, polygon, solver):
    """
    Render the clipped Voronoi diagram for two sets of seeds using Shapely 2.0+ only.

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
    # 1) Gather all seeds
    all_seeds = fixed + added
    names, coords = zip(*all_seeds) if all_seeds else ([], [])
    coords = list(coords)

    # 2) Compute Voronoi cells via Shapely
    cells = []
    area_list = []
    vor = voronoi_diagram(MultiPoint(coords), envelope=polygon)
    for name, region in zip(names, vor.geoms):
        cell = region.intersection(polygon)
        cells.append(cell)
        area_list.append((name, cell.area))

    # 3) Summary statistics
    n_added = len(added)
    min_area = min(area for _, area in area_list) if area_list else 0.0

    # Minimum pairwise distance among:
    #  - added–added
    #  - fixed–added
    # (exclude fixed–fixed)
    min_dist = 0.0
    if added:
        min_dist = np.inf
        # added–added
        for i, (_, (x1, y1)) in enumerate(added):
            for _, (x2, y2) in added[i+1:]:
                d = np.hypot(x1 - x2, y1 - y2)
                if d < min_dist:
                    min_dist = d
        # fixed–added
        for _, (x1, y1) in fixed:
            for _, (x2, y2) in added:
                d = np.hypot(x1 - x2, y1 - y2)
                if d < min_dist:
                    min_dist = d

    # Minimum distance from added points to polygon boundary
    min_bdy_dist = 0.0
    if added:
        bdy_dists = [
            polygon.exterior.distance(Point(x, y))
            for _, (x, y) in added
        ]
        min_bdy_dist = min(bdy_dists)

    # 4) Plotting
    fig, ax = plt.subplots()
    # Draw polygon outline
    x_poly, y_poly = polygon.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1.5)

    # Shade each Voronoi cell
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

    # Plot fixed seeds in red
    for name, (x, y) in fixed:
        ax.scatter(x, y, color='red')
        ax.text(x, y, name, fontsize=8, ha='right', va='bottom')
    # Plot added seeds in blue
    for name, (x, y) in added:
        ax.scatter(x, y, color='blue')
        ax.text(x, y, name, fontsize=8, ha='right', va='bottom')

    # Legend and equal aspect
    ax.scatter([], [], color='red', label='fixed')
    ax.scatter([], [], color='blue', label='added')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # 5) Title & console output
    # Prepare the conditional strings
    min_dist_str = '---' if not added else f"{min_dist:.3g}"
    min_bdy_str = '---' if not added else f"{min_bdy_dist:.3g}"

    title = (
        f"Solver: {solver} | n={n_added} | "
        f"Min area={min_area:.3g} | Min dist={min_dist_str} | "
        f"Min bdy dist={min_bdy_str}"
    )
    ax.set_title(title)


    print("Voronoi cell areas:")
    for name, area in area_list:
        print(f"  {name}: {area:.3g}")

    plt.show()
    return cells, area_list
