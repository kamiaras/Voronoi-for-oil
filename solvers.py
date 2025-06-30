import random
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
from scipy.optimize import differential_evolution as scipy_de

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
