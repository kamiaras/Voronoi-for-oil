import random
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
import voronoi_utils as vu
from scipy.optimize import differential_evolution as scipy_de

def compute_metrics(x_flat, fixed, polygon):
    """
    Given a flat vector x_flat = [x0,y0,x1,y1,...], compute:
      - min_area: the smallest Voronoi cell area for seeds fixed + added
      - min_dist: the minimum l2 distance between any added point and any other point
                  (added–added or added–fixed; fixed–fixed excluded)
    """
    # reconstruct added points
    pts = [(x_flat[i], x_flat[i+1]) for i in range(0, len(x_flat), 2)]
    # Voronoi areas
    _, areas = vu.get_voronoi_cells_and_areas(fixed + pts, polygon)
    min_area = min(areas)
    # distances
    min_dist = float("inf")
    # added–added
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
            if d < min_dist:
                min_dist = d
    # added–fixed
    for p in pts:
        for f in fixed:
            d = np.linalg.norm(np.array(p) - np.array(f))
            if d < min_dist:
                min_dist = d
    return min_area, min_dist

def differential_evolution_custom(loss, bounds, polygon, A_min, params):
    fixed   = params["fixed"]
    l2_min  = params["l2_min"]
    reg     = params["reg"]
    seed    = params.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # unpack DE hyperparams
    dim     = len(bounds)
    minx, maxx = bounds[0]
    miny, maxy = bounds[1]
    maxiter = params["maxiter"]
    popsize = params["popsize"]
    F       = params["F"]
    CR      = params["CR"]

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

    # Evaluate initial fitness (objective)
    fitness = [loss(ind) for ind in pop]
    # Track best
    best_idx = int(np.argmin(fitness))
    best_flat, best_fit = pop[best_idx].copy(), fitness[best_idx]

    # Early exit if already meets both thresholds
    ma, md = compute_metrics(best_flat, fixed, polygon)
    if ma >= A_min and md >= l2_min:
        return best_flat, best_fit

    for _ in range(maxiter):
        for i in range(popsize):
            # mutation
            idxs = [j for j in range(popsize) if j != i]
            a, b, c = random.sample(idxs, 3)
            v = pop[a] + F * (pop[b] - pop[c])

            # project back into polygon
            projected = []
            for p in range(dim//2):
                xi, yi = v[2*p], v[2*p+1]
                pt = Point(xi, yi)
                if not polygon.contains(pt):
                    proj_pt = nearest_points(polygon, pt)[0]
                    xi, yi = proj_pt.x, proj_pt.y
                projected.extend([xi, yi])
            v = np.array(projected)

            # pairwise crossover
            u = pop[i].copy()
            pairs = dim // 2
            jr = random.randrange(pairs)
            for p in range(pairs):
                if random.random() < CR or p == jr:
                    u[2*p:2*p+2] = v[2*p:2*p+2]

            fu = loss(u)
            if fu < fitness[i]:
                pop[i], fitness[i] = u, fu
                # update best
                if fu < best_fit:
                    best_fit, best_flat = fu, u.copy()
                    # check thresholds
                    ma, md = compute_metrics(best_flat, fixed, polygon)
                    if ma >= A_min and md >= l2_min:
                        return best_flat, best_fit

    return best_flat, best_fit

def qpso_solver(loss, bounds, polygon, A_min, params):
    fixed   = params["fixed"]
    l2_min  = params["l2_min"]
    reg     = params["reg"]
    seed    = params.get("seed", None)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    dim      = len(bounds)
    swarm_sz = params["popsize"]
    maxiter  = params["maxiter"]
    alpha    = params["alpha"]

    # init swarm
    swarm = []
    for _ in range(swarm_sz):
        pos = []
        while len(pos) < dim:
            x = random.uniform(bounds[0][0], bounds[0][1])
            y = random.uniform(bounds[1][0], bounds[1][1])
            if polygon.contains(Point(x,y)):
                pos.extend([x,y])
        swarm.append(np.array(pos))

    pbest   = [s.copy() for s in swarm]
    pbest_f = [loss(s) for s in swarm]
    gbest_i = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[gbest_i].copy(), pbest_f[gbest_i]

    # early exit?
    ma, md = compute_metrics(gbest, fixed, polygon)
    if ma >= A_min and md >= l2_min:
        return gbest, gbest_f

    for _ in range(maxiter):
        mbest = sum(pbest)/len(pbest)
        for i, xi in enumerate(swarm):
            new_x = np.zeros(dim)
            # per-dimension update
            for d in range(dim):
                u   = random.random()
                phi = random.random()
                Pi  = phi*pbest[i][d] + (1-phi)*gbest[d]
                sign= 1 if random.random()>0.5 else -1
                new_x[d] = Pi + sign * alpha * abs(mbest[d]-xi[d]) * np.log(1/u)

            # project and evaluate
            proj = []
            for p in range(dim//2):
                xd, yd = new_x[2*p], new_x[2*p+1]
                pt = Point(xd, yd)
                if not polygon.contains(pt):
                    pp = nearest_points(polygon, pt)[0]
                    xd, yd = pp.x, pp.y
                proj.extend([xd, yd])
            new_x = np.array(proj)

            fu = loss(new_x)
            swarm[i] = new_x
            if fu < pbest_f[i]:
                pbest[i], pbest_f[i] = new_x.copy(), fu
            if fu < gbest_f:
                gbest, gbest_f = new_x.copy(), fu
                # check thresholds
                ma, md = compute_metrics(gbest, fixed, polygon)
                if ma >= A_min and md >= l2_min:
                    return gbest, gbest_f

    return gbest, gbest_f

def qpso_pairwise_solver(loss, bounds, polygon, A_min, params):
    # similar to above but block‐update pairs…
    fixed   = params["fixed"]
    l2_min  = params["l2_min"]
    reg     = params["reg"]
    seed    = params.get("seed", None)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    dim      = len(bounds)
    swarm_sz = params["popsize"]
    maxiter  = params["maxiter"]
    alpha    = params["alpha"]

    swarm = []
    for _ in range(swarm_sz):
        pos=[]
        while len(pos)<dim:
            x=random.uniform(bounds[0][0],bounds[0][1])
            y=random.uniform(bounds[1][0],bounds[1][1])
            if polygon.contains(Point(x,y)):
                pos.extend([x,y])
        swarm.append(np.array(pos))

    pbest   = [s.copy() for s in swarm]
    pbest_f = [loss(s) for s in swarm]
    gbest_i = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[gbest_i].copy(), pbest_f[gbest_i]

    ma, md = compute_metrics(gbest, fixed, polygon)
    if ma>=A_min and md>=l2_min:
        return gbest, gbest_f

    for _ in range(maxiter):
        mbest = sum(pbest)/len(pbest)
        for i, xi in enumerate(swarm):
            new_x = np.zeros(dim)
            # block update pairs
            for p in range(dim//2):
                u   = random.random()
                phi = random.random()
                Pi_x = phi*pbest[i][2*p]   + (1-phi)*gbest[2*p]
                Pi_y = phi*pbest[i][2*p+1] + (1-phi)*gbest[2*p+1]
                sign = 1 if random.random()>0.5 else -1
                new_x[2*p]   = Pi_x   + sign*alpha*abs(mbest[2*p]-xi[2*p])   * np.log(1/u)
                new_x[2*p+1] = Pi_y + sign*alpha*abs(mbest[2*p+1]-xi[2*p+1]) * np.log(1/u)

            proj=[]
            for p in range(dim//2):
                xd, yd=new_x[2*p],new_x[2*p+1]
                pt=Point(xd,yd)
                if not polygon.contains(pt):
                    pp=nearest_points(polygon,pt)[0]
                    xd,yd=pp.x,pp.y
                proj.extend([xd,yd])
            new_x=np.array(proj)

            fu=loss(new_x)
            swarm[i]=new_x
            if fu<pbest_f[i]:
                pbest[i],pbest_f[i]=new_x.copy(),fu
            if fu<gbest_f:
                gbest,gbest_f=new_x.copy(),fu
                ma, md = compute_metrics(gbest, fixed, polygon)
                if ma>=A_min and md>=l2_min:
                    return gbest, gbest_f

    return gbest, gbest_f

def ga_solver(loss, bounds, polygon, A_min, params):
    from deap import base, creator, tools
    fixed   = params["fixed"]
    l2_min  = params["l2_min"]
    reg     = params["reg"]
    seed    = params.get("seed", None)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    dim  = len(bounds)
    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox=base.Toolbox()
    def init_ind():
        ind=[]
        while len(ind)<dim:
            x=random.uniform(bounds[0][0],bounds[0][1])
            y=random.uniform(bounds[1][0],bounds[1][1])
            if polygon.contains(Point(x,y)):
                ind.extend([x,y])
        return creator.Individual(ind)
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", loss)
    toolbox.register("mate",    tools.cxTwoPoint)
    toolbox.register("mutate",  tools.mutGaussian, mu=0,
                     sigma=params.get("sigma",0.1), indpb=params.get("indpb",0.1))
    toolbox.register("select",  tools.selTournament, tournsize=3)

    pop = toolbox.population(n=params["popsize"])
    for ind in pop:
        ind.fitness.values = (loss(ind),)
    best = tools.selBest(pop,1)[0]
    ma, md = compute_metrics(best, fixed, polygon)
    if ma>=A_min and md>=l2_min:
        return list(best), best.fitness.values[0]

    ngen = params["ngen"]
    cxpb, mutpb = params.get("cxpb",0.5), params.get("mutpb",0.2)
    for _ in range(ngen):
        offs = toolbox.select(pop, len(pop))
        offs = list(map(toolbox.clone,offs))
        # crossover
        for c1,c2 in zip(offs[::2],offs[1::2]):
            if random.random()<cxpb:
                toolbox.mate(c1,c2)
                del c1.fitness.values, c2.fitness.values
        # mutation
        for m in offs:
            if random.random()<mutpb:
                toolbox.mutate(m)
                del m.fitness.values
        invalid = [ind for ind in offs if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = (loss(ind),)
        pop[:] = offs
        best = tools.selBest(pop,1)[0]
        ma, md = compute_metrics(best, fixed, polygon)
        if ma>=A_min and md>=l2_min:
            return list(best), best.fitness.values[0]

    best = tools.selBest(pop,1)[0]
    return list(best), best.fitness.values[0]







from shapely.geometry import Point
from shapely.ops import nearest_points
import random
import numpy as np
import solvers  # assumes compute_metrics is defined there

def spsa_solver(loss, bounds, polygon, A_min, params):
    """
    SPSA-based zero-order solver with:
      - one-sided directional approximation ghat = normalize((f_plus - f_x)*(x_plus - x))
      - random initialization inside `polygon`
      - multiple restarts
      - early exit when both area ≥ A_min and distance ≥ l2_min

    Args:
        loss: callablе taking flat x vector, returning combined objective = –
              (min_area + reg*min_dist)
        bounds: list of (min, max) for each coordinate [ (x0_min,x0_max),(y0_min,y0_max), ... ]
        polygon: Shapely polygon
        A_min: minimum required Voronoi cell area
        params: dict with keys:
            fixed: list of fixed seed points
            l2_min: minimum required pairwise distance
            restarts: number of random restarts
            maxiter: iterations per restart
            a, c, alpha, gamma: SPSA hyperparameters
            seed: optional RNG seed
    Returns:
        best_flat, best_obj  (flat vector and its loss value)
    """
    seed = params.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # unpack SPSA hyperparameters
    a      = params.get("a", 0.1)
    c      = params.get("c", 0.1)
    alpha  = params.get("alpha", 0.602)
    gamma  = params.get("gamma", 0.101)
    maxiter= params.get("maxiter", 100)
    restarts = params.get("restarts", 5)

    dim = len(bounds)
    # assume bounds = [(minx,maxx),(miny,maxy)] * n_added
    minx, maxx = bounds[0]
    miny, maxy = bounds[1]
    n_added = dim // 2

    best_global = None
    best_global_obj = np.inf

    def project(x_vec):
        # project any out-of-polygon points back onto the polygon boundary
        proj = []
        for i in range(n_added):
            xi, yi = x_vec[2*i], x_vec[2*i+1]
            pt = Point(xi, yi)
            if not polygon.contains(pt):
                p = nearest_points(polygon, pt)[0]
                xi, yi = p.x, p.y
            proj.extend([xi, yi])
        return np.array(proj)

    for _ in range(restarts):
        # random initialization of added points inside polygon
        added = []
        while len(added) < n_added:
            x0 = random.uniform(minx, maxx)
            y0 = random.uniform(miny, maxy)
            if polygon.contains(Point(x0, y0)):
                added.append((x0, y0))
        x = np.array([coord for pt in added for coord in pt])
        x = project(x)

        # initial objective and raw metrics
        obj_x = loss(x)
        ma, md = solvers.compute_metrics(x, params["fixed"], polygon)
        best, best_obj = x.copy(), obj_x
        if ma >= A_min and md >= params["l2_min"]:
            return x, obj_x
        if obj_x < best_global_obj:
            best_global, best_global_obj = x.copy(), obj_x

        for k in range(1, maxiter+1):
            ak = a / (k ** alpha)
            ck = c / (k ** gamma)

            # one-sided directional approximation
            delta = np.random.choice([-1, 1], size=dim)
            x_plus = project(x + ck * delta)
            f_plus = loss(x_plus)

            # directional vector scaled by objective change
            v = (f_plus - obj_x) * (x_plus - x)
            norm_v = np.linalg.norm(v)
            ghat = v / norm_v if norm_v > 0 else np.zeros(dim)

            # update and project
            x = project(x - ak * ghat)
            obj_x = loss(x)

            # check improvement & early exit
            ma, md = solvers.compute_metrics(x, params["fixed"], polygon)
            if ma >= A_min and md >= params["l2_min"]:
                return x, obj_x
            if obj_x < best_obj:
                best, best_obj = x.copy(), obj_x
                if best_obj < best_global_obj:
                    best_global, best_global_obj = best.copy(), best_obj

    return best_global, best_global_obj
