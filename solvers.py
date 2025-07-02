import random
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

def differential_evolution_custom(loss, bounds, polygon, A_min, params):
    """
    Your custom DE from before.
    """
    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(None)

    dim = len(bounds)
    minx, maxx = bounds[0]
    miny, maxy = bounds[1]
    maxiter = params["maxiter"]
    popsize = params["popsize"]
    F = params["F"]
    CR = params["CR"]

    # initialize population inside polygon
    pop = []
    for _ in range(popsize):
        ind = []
        while len(ind) < dim:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if polygon.contains(Point(x, y)):
                ind.extend([x, y])
        pop.append(np.array(ind))

    fitness = [loss(ind) for ind in pop]
    best_idx = int(np.argmin(fitness))
    best, best_fit = pop[best_idx].copy(), fitness[best_idx]

    if -best_fit >= A_min:
        return best, best_fit

    for _ in range(maxiter):
        for i in range(popsize):
            idxs = [j for j in range(popsize) if j != i]
            a, b, c = random.sample(idxs, 3)
            v = pop[a] + F * (pop[b] - pop[c])

            # project back inside polygon
            projected = []
            for p in range(dim // 2):
                x, y = v[2*p], v[2*p+1]
                pt = Point(x, y)
                if not polygon.contains(pt):
                    proj = nearest_points(polygon, pt)[0]
                    x, y = proj.x, proj.y
                projected.extend([x, y])
            v = np.array(projected)

            # pairwise crossover on coordinate pairs
            u = pop[i].copy()
            n_pairs = dim // 2
            j_rand = random.randrange(n_pairs)
            for p in range(n_pairs):
                if random.random() < CR or p == j_rand:
                    u[2*p:2*p+2] = v[2*p:2*p+2]

            fu = loss(u)
            if fu < fitness[i]:
                pop[i], fitness[i] = u, fu
                if fu < best_fit:
                    best_fit, best = fu, u.copy()
                    if -best_fit >= A_min:
                        return best, best_fit

    return best, best_fit

def qpso_solver(loss, bounds, polygon, A_min, params):
    """
    Original QPSO (per-dimension updates).
    """
    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    else:
        random.seed(None)

    dim = len(bounds)
    swarm_size = params["popsize"]
    maxiter = params["maxiter"]
    alpha = params.get("alpha", 0.75)

    # initialize swarm
    swarm = []
    for _ in range(swarm_size):
        pos = []
        while len(pos) < dim:
            x = random.uniform(bounds[0][0], bounds[0][1])
            y = random.uniform(bounds[1][0], bounds[1][1])
            if polygon.contains(Point(x, y)):
                pos.extend([x, y])
        swarm.append(np.array(pos))

    pbest = [s.copy() for s in swarm]
    pbest_f = [loss(s) for s in swarm]
    gbest_idx = int(np.argmin(pbest_f))
    gbest = pbest[gbest_idx].copy()
    gbest_f = pbest_f[gbest_idx]

    if -gbest_f >= A_min:
        return gbest, gbest_f

    for _ in range(maxiter):
        mbest = sum(pbest) / len(pbest)
        for i, x_i in enumerate(swarm):
            new_x = np.zeros(dim)
            # per-dimension update
            for d in range(dim):
                u = random.random()
                phi = random.random()
                Pi = phi * pbest[i][d] + (1 - phi) * gbest[d]
                sign = 1 if random.random() > 0.5 else -1
                new_x[d] = Pi + sign * alpha * abs(mbest[d] - x_i[d]) * np.log(1/u)

            # project back into polygon in pairs
            proj = []
            for p in range(dim // 2):
                xd, yd = new_x[2*p], new_x[2*p+1]
                pt = Point(xd, yd)
                if not polygon.contains(pt):
                    proj_pt = nearest_points(polygon, pt)[0]
                    xd, yd = proj_pt.x, proj_pt.y
                proj.extend([xd, yd])
            new_x = np.array(proj)

            f_new = loss(new_x)
            swarm[i] = new_x
            if f_new < pbest_f[i]:
                pbest[i], pbest_f[i] = new_x.copy(), f_new
            if f_new < gbest_f:
                gbest, gbest_f = new_x.copy(), f_new
                if -gbest_f >= A_min:
                    return gbest, gbest_f

    return gbest, gbest_f

def qpso_pairwise_solver(loss, bounds, polygon, A_min, params):
    """
    QPSO variant that updates coordinate pairs jointly.
    """
    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    else:
        random.seed(None)

    dim = len(bounds)
    swarm_size = params["popsize"]
    maxiter = params["maxiter"]
    alpha = params.get("alpha", 0.75)

    # initialize swarm
    swarm = []
    for _ in range(swarm_size):
        pos = []
        while len(pos) < dim:
            x = random.uniform(bounds[0][0], bounds[0][1])
            y = random.uniform(bounds[1][0], bounds[1][1])
            if polygon.contains(Point(x, y)):
                pos.extend([x, y])
        swarm.append(np.array(pos))

    pbest = [s.copy() for s in swarm]
    pbest_f = [loss(s) for s in swarm]
    gbest_idx = int(np.argmin(pbest_f))
    gbest = pbest[gbest_idx].copy()
    gbest_f = pbest_f[gbest_idx]

    if -gbest_f >= A_min:
        return gbest, gbest_f

    for _ in range(maxiter):
        mbest = sum(pbest) / len(pbest)
        for i, x_i in enumerate(swarm):
            new_x = np.zeros(dim)
            # update in coordinate-pair blocks
            for p in range(dim // 2):
                u = random.random()
                phi = random.random()
                Pi_x = phi * pbest[i][2*p]   + (1-phi) * gbest[2*p]
                Pi_y = phi * pbest[i][2*p+1] + (1-phi) * gbest[2*p+1]
                sign = 1 if random.random() > 0.5 else -1
                new_x[2*p]   = Pi_x   + sign * alpha * abs(mbest[2*p]   - x_i[2*p])   * np.log(1/u)
                new_x[2*p+1] = Pi_y + sign * alpha * abs(mbest[2*p+1] - x_i[2*p+1]) * np.log(1/u)

            # project back
            proj = []
            for p in range(dim // 2):
                xd, yd = new_x[2*p], new_x[2*p+1]
                pt = Point(xd, yd)
                if not polygon.contains(pt):
                    proj_pt = nearest_points(polygon, pt)[0]
                    xd, yd = proj_pt.x, proj_pt.y
                proj.extend([xd, yd])
            new_x = np.array(proj)

            f_new = loss(new_x)
            swarm[i] = new_x
            if f_new < pbest_f[i]:
                pbest[i], pbest_f[i] = new_x.copy(), f_new
            if f_new < gbest_f:
                gbest, gbest_f = new_x.copy(), f_new
                if -gbest_f >= A_min:
                    return gbest, gbest_f

    return gbest, gbest_f

def ga_solver(loss, bounds, polygon, A_min, params):
    """
    Genetic Algorithm via DEAP.
    """
    from deap import base, creator, tools

    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    else:
        random.seed(None)

    dim = len(bounds)
    # set up DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    def init_ind():
        ind = []
        while len(ind) < dim:
            x = random.uniform(bounds[0][0], bounds[0][1])
            y = random.uniform(bounds[1][0], bounds[1][1])
            if polygon.contains(Point(x, y)):
                ind.extend([x, y])
        return creator.Individual(ind)

    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", loss)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0, sigma=params.get("sigma", 0.1),
                     indpb=params.get("indpb", 0.1))
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=params["popsize"])
    # evaluate initial
    invalid = pop
    for ind in invalid:
        ind.fitness.values = (loss(ind),)
    best = tools.selBest(pop, 1)[0]
    if -best.fitness.values[0] >= A_min:
        return list(best), best.fitness.values[0]

    ngen = params["ngen"]
    cxpb, mutpb = params.get("cxpb", 0.5), params.get("mutpb", 0.2)

    for _ in range(ngen):
        offs = toolbox.select(pop, len(pop))
        offs = list(map(toolbox.clone, offs))
        # crossover
        for c1, c2 in zip(offs[::2], offs[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        # mutation
        for m in offs:
            if random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values
        # evaluate
        invalid = [ind for ind in offs if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = (loss(ind),)
        pop[:] = offs

        best = tools.selBest(pop, 1)[0]
        if -best.fitness.values[0] >= A_min:
            return list(best), best.fitness.values[0]

    best = tools.selBest(pop, 1)[0]
    return list(best), best.fitness.values[0]





def spsa_solver(loss, bounds, polygon, A_min, params):
    """
    SPSA-based zero-order solver with random initialization and multiple restarts,
    using one-sided directional approximation and normalization:
        ghat = normalize( (f_plus - f_x) * (x_plus - x) )
    """
    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed(None)

    # SPSA hyperparameters
    a = params.get("a", 0.1)
    c = params.get("c", 0.1)
    alpha = params.get("alpha", 0.602)
    gamma = params.get("gamma", 0.101)
    maxiter = params.get("maxiter", 100)
    restarts = params.get("restarts", 5)

    dim = len(bounds)
    # bounds assumed pairwise: [(minx,maxx),(miny,maxy)] * n
    minx, maxx = bounds[0]
    miny, maxy = bounds[1]
    n = dim // 2

    best_global = None
    best_global_f = np.inf

    def project(x_vec):
        proj = []
        for i in range(n):
            xi, yi = x_vec[2*i], x_vec[2*i+1]
            pt = Point(xi, yi)
            if not polygon.contains(pt):
                p = nearest_points(polygon, pt)[0]
                xi, yi = p.x, p.y
            proj.extend([xi, yi])
        return np.array(proj)

    for _ in range(restarts):
        # random initialization inside polygon
        added = []
        while len(added) < n:
            x0 = random.uniform(minx, maxx)
            y0 = random.uniform(miny, maxy)
            if polygon.contains(Point(x0, y0)):
                added.append((x0, y0))
        x = np.array([coord for pt in added for coord in pt])
        x = project(x)

        # initial loss
        f_x = loss(x)
        best, best_f = x.copy(), f_x

        if best_f < best_global_f:
            best_global, best_global_f = best.copy(), best_f
        if -best_global_f >= A_min:
            return best_global, best_global_f

        for k in range(1, maxiter + 1):
            ak = a / (k ** alpha)
            ck = c / (k ** gamma)

            # generate perturbation ±1
            delta = np.random.choice([-1, 1], size=dim)

            # forward evaluation
            x_plus = project(x + ck * delta)
            f_plus = loss(x_plus)

            # one-sided directional approximation
            diff = x_plus - x
            v = (f_plus - f_x) * diff
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                ghat = v / norm_v
            else:
                ghat = np.zeros(dim)

            # update
            x = project(x - ak * ghat)
            f_x = loss(x)

            if f_x < best_f:
                best, best_f = x.copy(), f_x
                if best_f < best_global_f:
                    best_global, best_global_f = best.copy(), best_f
                if -best_global_f >= A_min:
                    return best_global, best_global_f

    return best_global, best_global_f





























import voronoi_utils as vu

def lloyd_cvt_solver(loss, bounds, polygon, A_min, params):
    """
    Lloyd–CVT solver with multiple random initializations.
    Uses averaging with centroids: new_added = 0.5*(centroid + old point).
    """
    import random, numpy as np
    from shapely.geometry import Point
    import voronoi_utils as vu

    seed = params.get("seed", None)
    restarts = params.get("restarts", 1)
    n = len(bounds) // 2

    best_x_flat = None
    best_loss = float('inf')

    minx, maxx = bounds[0]
    miny, maxy = bounds[1]

    for r in range(restarts):
        # seed each restart
        if seed is not None:
            random.seed(seed + r)
            np.random.seed(seed + r)
        else:
            random.seed(None)

        # initialize added points inside polygon
        added = []
        while len(added) < n:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if polygon.contains(Point(x, y)):
                added.append((x, y))

        current_loss = None

        # Lloyd iterations
        for _ in range(params["maxiter"]):
            all_pts = params["fixed"] + added
            cells, _ = vu.get_voronoi_cells_and_areas(all_pts, polygon)
            added_cells = cells[-n:]
            # compute centroids
            centroids = [tuple(cell.centroid.coords)[0] for cell in added_cells]
            # average with previous positions
            added = [((added[i][0] + centroids[i][0]) * 0.5,
                      (added[i][1] + centroids[i][1]) * 0.5)
                     for i in range(n)]

            # flatten and evaluate loss
            x_flat = np.array([coord for pt in added for coord in pt])
            current_loss = loss(x_flat)

            # early exit if feasible
            if -current_loss >= A_min:
                break

        # record best across restarts
        if current_loss is not None and current_loss < best_loss:
            best_loss = current_loss
            best_x_flat = x_flat.copy()

    return best_x_flat, best_loss

