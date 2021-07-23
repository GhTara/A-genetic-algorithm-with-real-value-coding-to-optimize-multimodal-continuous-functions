import numpy as np


def select_parents(num_par, population, fitness):
    parents = np.empty(shape=(num_par, population.shape[1]))
    for p_i in range(num_par):
        i_max_fit = np.where(np.max(fitness) == fitness)[0][0]
        parents[p_i, :] = population[i_max_fit, :]
        fitness[i_max_fit] = -float('inf')
    return parents


# roulette wheel selection algorithm
def select_roul_wheel(population, fitness):
    pop_fit = [list(arr) for arr in list(
        zip(*sorted(zip(population, fitness), key=lambda t:-t[1])))]
    population, fitness = np.array(pop_fit[0]), np.array(pop_fit[1])
    total_fit = np.sum(fitness)
    prob_fitness = [fit/total_fit for fit in fitness]
    occu_fitness = [np.sum(prob_fitness[:i+1]) for i in range(len(fitness))]
    new_population = []
    n_pop = len(population)
    while n_pop:
        for i in range(len(fitness)):
            if np.random.rand() <= occu_fitness[i]:
                new_population.append(population[i, :])
                n_pop -= 1
                break
    return new_population


def crossover(parents, p_c):
    # cross over on two parents => two child
    # p_c : large value close to 1
    offsprings = []
    num_gens = parents.shape[1]
    for i in range(0, parents.shape[0], 2):
        p1_i = i % parents.shape[0]
        p2_i = (i+1) % parents.shape[0]
        p1 = parents[p1_i, :]
        p2 = parents[p2_i, :]

        offspring = np.empty(shape=(num_gens))
        if np.random.rand() < p_c:
            crossover_point = np.random.randint(1, num_gens-2)
            offspring = np.array(list(p1[0:crossover_point]) +
                                 list(p2[crossover_point:]))
            offsprings.append(offspring)
            offspring = np.array(list(p2[0:crossover_point]) +
                                 list(p1[crossover_point:]))
            offsprings.append(offspring)
        else:
            offspring = p1
            offsprings.append(offspring)
            offspring = p2
            offsprings.append(offspring)

    return np.array(offsprings)

# mutation for a child


def mutation(low, high, offspring_crossover, p_m):
    for i in range(len(offspring_crossover)):
        if np.random.rand() < p_m:

            random_val = np.random.uniform(low=low, high=high, size=1)
            offspring_crossover[i] = random_val

    return offspring_crossover


def optimize_by_ga(fitness_func, new_population, num_generation, num_parents, p_c, p_m):
    progress = []
    for _ in range(num_generation):
        # ------- second step: calculate fitnees function(1) -------
        fitness = fitness_func(population=new_population)
        # print(fitness)
        # ------- second step: select best parents(2) -------
        selected_parents = select_parents(
            num_par=num_parents, population=new_population, fitness=fitness)
        # print(selected_parents)
        # ------- third step: crossover on parents -------
        offsprings = crossover(parents=selected_parents, p_c=p_c)
        # print(offsprings)
        # ------- forth step: mutation -------
        for i in range(len(offsprings)):
            offsprings[i, :] = mutation(
                offspring_crossover=offsprings[i, :], p_m=p_m)
        # print(offsprings)
        # ------- new population -------
        fitness = fitness_func(offsprings)
        child_fit = [list(arr) for arr in list(
            zip(*sorted(zip(offsprings, fitness), key=lambda t:t[1])))]
        offsprings, ch_fitness = child_fit[0], child_fit[1]
        fitness = fitness_func(new_population)
        pop_fit = [list(arr) for arr in list(
            zip(*sorted(zip(new_population, fitness), key=lambda t:t[1])))]
        new_population, po_fitness = pop_fit[0], pop_fit[1]
        # fitness = fitness_func(equation_inputs, offsprings)

        # for replacement best fitness ramian in population
        for i in range(len(offsprings)):
            if ch_fitness[i] > po_fitness[i]:
                # print(offsprings[i].shape)
                new_population[i] = offsprings[i]
        new_population = np.array(new_population)
        progress.append(np.max(fitness_func(new_population)))

    fitness = fitness_func(new_population)

    return new_population, fitness, progress


def optimize_by_ga_main(low, high, fitness_func, new_population, num_generation, num_parents, p_c, p_m):
    progress = []
    progress_fit = []
    for _ in range(num_generation):
        # ------- second step: calculate fitnees function(1) -------
        fitness = fitness_func(population=new_population)
        # print(fitness)
        # ------- second step: select best parents(2) -------
        selected_parents = select_roul_wheel(
            population=new_population, fitness=fitness)
        # print(selected_parents)
        # ------- third step: crossover on parents -------
        n_pop = len(new_population)
        childs = np.empty(shape=new_population.shape)
        for i in range(0, len(new_population), 2):
            two_parents = np.array(
                [selected_parents[i % n_pop], selected_parents[(i+1) % n_pop]])
            offsprings = crossover(parents=two_parents, p_c=p_c)
            # print(offsprings)
            # ------- forth step: mutation -------
            childs[i % n_pop, :] = mutation(
                low=low, high=high, offspring_crossover=offsprings[0, :], p_m=p_m)
            childs[(i+1) % n_pop, :] = mutation(
                low=low, high=high, offspring_crossover=offsprings[1, :], p_m=p_m)

        new_population = childs
        progress.append(np.min(fitness_func(new_population, mode=1)))
        progress_fit.append(np.max(fitness_func(new_population, mode=0)))

    fitness = fitness_func(new_population)

    return new_population, fitness, progress, progress_fit
