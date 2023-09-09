#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/5 14:09
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : LMOCSO.py
# @Statement : Large-scale multi-objective competitive swarm optimizer (LMOCSO)
# @Reference : Tian Y, Zheng X, Zhang X, et al. Efficient large-scale multiobjective optimization based on a competitive swarm optimizer[J]. IEEE Transactions on Cybernetics, 2019, 50(8): 3696-3708.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cdist


def cal_obj(pop, nobj):
    # 0 <= x <= 1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, dim):
    # calculate approximately npop uniformly distributed reference points on dim dimensions
    h1 = 0
    while combination(h1 + dim, dim - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + dim), dim - 1))) - np.arange(dim - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < dim:
        h2 = 0
        while combination(h1 + dim - 1, dim - 1) + combination(h2 + dim, dim - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + dim), dim - 1))) - np.arange(dim - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * dim)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def calSDE(objs):
    # calculate shift-based density estimation
    npop = objs.shape[0]
    fmax = np.max(objs, axis=0)
    fmin = np.min(objs, axis=0)
    objs = (objs - fmin) / (fmax - fmin)
    dis = np.full((npop, npop), np.inf)
    for i in range(npop):
        temp_objs = np.max((objs, np.tile(objs[i], (npop, 1))), axis=0)
        for j in range(npop):
            if i != j:
                dis[i, j] = np.sqrt(np.sum((objs[i] - temp_objs[i]) ** 2))
    return np.min(dis, axis=1)


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    site = np.random.random((npop, dim)) < 1 / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def environmental_selection(pop, objs, vel, V, theta, gamma):
    # RVEA environmental selection
    # Step 1. ND sort
    pfs, rank = nd_sort(objs)
    pop = pop[pfs[1]]
    objs = objs[pfs[1]]
    vel = vel[pfs[1]]

    # Step 2. Objective translation
    t_objs = objs - np.min(objs, axis=0)

    # Step 4. Population partition
    angle = np.arccos(1 - cdist(t_objs, V, 'cosine'))
    association = np.argmin(angle, axis=1)

    # Step 5. Angle-penalized distance calculation
    Next = np.full(V.shape[0], -1)
    for i in np.unique(association):
        current = np.where(association == i)[0]
        if current.size != 0:
            APD = (1 + objs.shape[1] * theta * angle[current, i] / gamma[i]) * np.sqrt(np.sum(t_objs[current] ** 2, axis=1))
            best = np.argmin(APD)
            Next[i] = current[best]
    ind = np.where(Next != -1)[0]
    next_pop = pop[Next[ind]]
    next_objs = objs[Next[ind]]
    next_vel = vel[Next[ind]]
    return next_pop, next_objs, next_vel


def main(npop, iter, lb, ub, nobj=3, eta_m=20, alpha=2):
    """
    The main loop
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_m: perturbance factor distribution index (default = 20)
    :param alpha: the parameter to control the change rate of APD (default = 2)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    V = reference_points(npop, nobj)  # reference points
    npop = V.shape[0]  # population size
    pos = np.random.uniform(lb, ub, (npop, nvar))  # positions
    objs = cal_obj(pos, nobj)  # objectives
    vel = np.zeros((npop, nvar))  # velocities
    cosine = 1 - cdist(V, V, 'cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(cosine), axis=1)
    pos, objs, vel = environmental_selection(pos, objs, vel, V, (1 / iter) ** alpha, gamma)

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 200 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Find winners and losers
        fitness = calSDE(objs)
        if len(pos) >= 2:
            rank = np.random.choice(len(pos), int(len(pos) / 2) * 2, replace=False)
        else:
            rank = np.array([0, 0])
        loser = rank[:int(len(rank) / 2)]
        winner = rank[int(len(rank) / 2):]
        change = fitness[loser] >= fitness[winner]
        temp = loser[change].copy()
        loser[change] = winner[change]
        winner[change] = temp

        # Step 2.2. Update velocities and positions
        loser_pos = pos[loser]
        loser_vel = vel[loser]
        winner_pos = pos[winner]
        winner_vel = vel[winner]
        r1 = np.tile(np.random.random((loser.shape[0], 1)), (1, nvar))
        r2 = np.tile(np.random.random((loser.shape[0], 1)), (1, nvar))
        off_vel = r1 * loser_vel + r2 * (winner_pos - loser_pos)
        off_pos = loser_pos + off_vel + r1 * (off_vel - loser_vel)
        off_pos = np.concatenate((off_pos, winner_pos), axis=0)
        off_vel = np.concatenate((off_vel, winner_vel), axis=0)
        off_pos = mutation(off_pos, lb, ub, eta_m)
        off_objs = cal_obj(off_pos, nobj)

        # Step 2.3. Environmental selection
        pos, objs, vel = environmental_selection(np.concatenate((pos, off_pos), axis=0), np.concatenate((objs, off_objs), axis=0), np.concatenate((vel, off_vel), axis=0), V, ((t + 1) / iter) ** alpha, gamma)

    # Step 3. Sort the results
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in objs]
    y = [o[1] for o in objs]
    z = [o[2] for o in objs]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 2000, np.array([0] * 7), np.array([1] * 7))
