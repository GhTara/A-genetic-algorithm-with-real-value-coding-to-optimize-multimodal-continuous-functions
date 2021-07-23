import numpy as np
from ga import mutation, select_roul_wheel


def initialize_pop(interval, n_dim, m, n_sub_pop, th_h=0.2):
    pop = []
    # size of each population
    m_itr = m
    # number of sub population
    n_sub_pop_itr = n_sub_pop
    for _ in range(n_sub_pop):
        sub_pop = np.empty(shape=(m, n_dim))
        sub_pop[:] = np.nan
        for i in range(m):
            choromosom = np.random.uniform(
                low=interval[0], high=interval[1], size=n_dim)
            sub_pop[i, :] = choromosom
            if i > 0:
                if antropy_total(n_dim, sub_pop, interval, i+1) > th_h:
                    None
                else:
                    i -= 1
        pop.append(sub_pop)
    return pop


def antropy_total(n_dim, sub_pop, interval, m):
    out = 0
    for j in range(n_dim):
        out += antropy_j(sub_pop, j, interval, m)
    return out / n_dim


def antropy_j(sub_pop, j, interval, m):
    out = 0
    for i in range(0, m):
        for k in range(i+1, m):
            cal_p_ik = cal_p(sub_pop[i, :], sub_pop[k, :], j, interval)
            out += (-(cal_p_ik)*np.log(cal_p_ik))

    return out


def cal_p(chor1, chor2, index, interval):
    return 1 - (np.abs(chor1[index] - chor2[index])) / (interval[1]-interval[0])


def crossover(parents, p_c):
    offsprings = []
    num_gens = parents.shape[1]
    for i in range(0, parents.shape[0], 2):
        p1_i = i % parents.shape[0]
        p2_i = (i+1) % parents.shape[0]
        p1 = parents[p1_i, :]
        p2 = parents[p2_i, :]

        offspring = np.empty(shape=(num_gens))
        offspring[:] = np.nan
        if np.random.rand() < p_c:
            crossover_point = np.random.randint(1, num_gens-2)
            offspring = np.array(list(p1[0:crossover_point]) +
                                 list((p1[crossover_point:]+p2[crossover_point:])/2))
            offsprings.append(offspring)
            offspring = np.array(list(p2[0:crossover_point]) +
                                 list((p1[crossover_point:]+p2[crossover_point:])/2))
            offsprings.append(offspring)
        else:
            offspring = p1
            offsprings.append(offspring)
            offspring = p2
            offsprings.append(offspring)

    return np.array(offsprings)

# mutation for a child


def change_p_m(k, MaxGen):
    p_m_0 = 1
    alpha = MaxGen/np.log(float(p_m_0)/10**(-3))
    p_m_k = p_m_0*np.exp(-k/alpha)
    return p_m_k


def gentic_subpop_step2(interval, n_dim, m, n_sub_pop, MaxGen1, fitness_func, p_c):
    low, high = interval[0], interval[1]
    population = initialize_pop(
        interval=interval, n_dim=n_dim, m=m, n_sub_pop=n_sub_pop)
    best_fit = -float('inf')
    best_choromosom = population[0][0]
    for l in range(n_sub_pop):
        new_population = population[l]
        for k in range(MaxGen1):
            p_m = change_p_m(k=k, MaxGen=MaxGen1)
            # ------- second step: calculate fitnees function(1) -------
            fitness = fitness_func(
                population=new_population, a=f'aaaaaaaaaaaaaaaaaaaa{l}')
            # ------- second step: select best parents(2) -------
            selected_parents = select_roul_wheel(
                population=new_population, fitness=fitness)
            # ------- third step: crossover on parents -------
            n_pop = len(new_population)
            childs = np.empty(shape=new_population.shape)
            childs[:] = np.nan
            for i in range(0, len(new_population), 2):
                two_parents = np.array(
                    [selected_parents[i % n_pop], selected_parents[(i+1) % n_pop]])
                offsprings = crossover(parents=two_parents, p_c=p_c)
                # ------- forth step: mutation -------
                childs[i % n_pop, :] = mutation(
                    low=low, high=high, offspring_crossover=offsprings[0, :], p_m=p_m)
                childs[(i+1) % n_pop, :] = mutation(
                    low=low, high=high, offspring_crossover=offsprings[1, :], p_m=p_m)

            new_population = childs
        population[l] = new_population
        fitness = fitness_func(new_population, a='sssssssssssssssssssssss',)
        best_match_idx = np.where(fitness == np.max(fitness))[0][0]
        if best_fit < np.max(fitness):
            best_choromosom = new_population[best_match_idx]
            best_fit = np.max(fitness)
    return best_choromosom


def change_v_domain(k, MaxGen, v0):
    alpha = MaxGen/np.log(float(v0)/10**(-3))
    vk = v0*np.exp(-k/alpha)
    return vk


def initial_population(origins,  interval, v, n_dim, n_pop):
    dims = np.empty(shape=(n_dim, n_pop))
    dims[:] = np.nan
    for i in range(n_dim):
        dims[i, :] = np.random.uniform(
            low=max(origins[i]-v, interval[0]), high=min(origins[i]+v, interval[1]), size=(n_pop))
    return dims.T


def mutation_interval(offspring_crossover, p_m, origins, interval, v, n_dim):
    for i in range(n_dim):
        if np.random.rand() < p_m:
            random_val = np.random.uniform(
                low=max(origins[i]-v, interval[0]), high=min(origins[i]+v, interval[1]), size=1)
            offspring_crossover[i] = random_val

    return offspring_crossover


def gentic_subpop_step3(interval, n_dim, m, n_sub_pop, MaxGen1, fitness_func, p_c, MaxGen2, v0, n_pop):
    best_choromosom = gentic_subpop_step2(
        interval, n_dim, m, n_sub_pop, MaxGen1, fitness_func, p_c)

    vk = v0
    # NbPr ~ 2 times of n_dim
    best_fitness = -float('inf')
    new_population = np.empty(shape=(n_pop, n_dim))
    new_population[:] = np.nan
    progress = []
    progress_fit = []
    for k in range(MaxGen2):
        new_population = initial_population(
            origins=best_choromosom, interval=interval, v=vk, n_dim=n_dim, n_pop=n_pop)
        # change the neighborhood for next generation
        vk = change_v_domain(k, MaxGen=MaxGen2, v0=v0)
        # ------------------------------------- calulate current best choromosom
        # embedd genetic operator on population
        p_m = change_p_m(k=k, MaxGen=MaxGen1)
        # ------- second step: calculate fitnees function(1) -------
        fitness = fitness_func(population=new_population,
                               a='ddddddddddddddddddddddddd',)
        # ------- second step: select best parents(2) -------
        selected_parents = select_roul_wheel(
            population=new_population, fitness=fitness)
        # ------- third step: crossover on parents -------
        childs = np.empty(shape=new_population.shape)
        childs[:] = np.nan
        for i in range(0, len(new_population), 2):
            two_parents = np.array(
                [selected_parents[i % n_pop], selected_parents[(i+1) % n_pop]])
            offsprings = crossover(parents=two_parents, p_c=p_c)
            # ------- forth step: mutation -------
            childs[i % n_pop, :] = mutation_interval(
                offspring_crossover=offsprings[0, :], p_m=p_m, origins=best_choromosom, interval=interval, v=vk, n_dim=n_dim)
            childs[(i+1) % n_pop, :] = mutation_interval(
                offspring_crossover=offsprings[1, :], p_m=p_m, origins=best_choromosom, interval=interval, v=vk, n_dim=n_dim)
        # final population in gen_itr k
        new_population = childs
        fitness = fitness_func(new_population, a='fffffffffffffffffffffffff',)
        best_match_idx = np.where(fitness == np.max(fitness))[0][0]
        best_choromosom = new_population[best_match_idx]
        # NbBm criteria
        the = 10**(-4)
        tmp = np.empty(shape=(1, n_dim))
        tmp[0, :] = best_choromosom
        # if np.abs(best_fitness - fitness_func(tmp, a='ggggggggggggggg',)) < the:
        #     break
        # best_fitness = fitness_func(tmp, a='hhhhhhhhhhhhhhhhh', )
        progress.append(
            np.min(fitness_func(new_population, mode=1, a='jjjjjjjjjjjjjjjjjjjjjjjjj',)))
        progress_fit.append(np.max(fitness_func(new_population, mode=0)))

    return new_population, fitness_func(new_population, a='kkkkkkkkkkkkkkkkkkkkkkkkkkk',), progress, progress_fit
