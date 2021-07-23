import numpy as np
import matplotlib.pyplot as plt
import ga
from RCGA import *
from sphere_benchmark import Sphere
from bent_Cigar_benchmark import BentCigar
from rastrigins_benchmark import Rastrigins
from ackley_benchmark import ACKLEY

benchmark_name = ['Sphere', 'Bent Cigar', 'rastrigins', 'ackley']
error_classic = [0]*4
error_RCGA = [0]*4

# for itr in range(51):
for itr in range(1):
    for i in range(4):
        # ------- first step: initialize the population -------
        # number of weights to optimize the equation (number of gens)
        num_var = 30
        # number of solution per population
        sol_per_pop = 5
        num_par = 2
        p_c = 0.9
        # p_m = 1.0 / float(num_var)
        p_m = 0.1
        if i == 0:
            benchmark = Sphere(num_var=num_var, n_pop=sol_per_pop)
            # number of sub populations
            num_sub_pop = 5
            # size of each sub populations
            m = 5
            MaxGen1 = 150
            MaxGen2 = 300
            v0 = benchmark.interval[1]-benchmark.interval[0]
            num_generation = 300
        elif i == 1:
            benchmark = BentCigar(num_var=num_var, n_pop=sol_per_pop)
            # number of sub populations
            num_sub_pop = 10
            # size of each sub populations
            m = 5
            MaxGen1 = 200
            MaxGen2 = 300
            v0 = 10
            num_generation = 300
        elif i == 2:
            benchmark = Rastrigins(num_var=num_var, n_pop=sol_per_pop)
            # number of sub populations
            num_sub_pop = 5
            # size of each sub populations
            m = 5
            MaxGen1 = 150
            MaxGen2 = 300
            v0 = benchmark.interval[1]-benchmark.interval[0]
            num_generation = 300
        else:
            benchmark = ACKLEY(num_var=num_var, n_pop=sol_per_pop)
            # number of sub populations
            num_sub_pop = 5
            # size of each sub populations
            m = 5
            MaxGen1 = 150
            MaxGen2 = 300
            v0 = benchmark.interval[1]-benchmark.interval[0]
            num_generation = 300

        # Sphere ------------------------------------------------------------
        new_population = benchmark.random_solutions
        new_population, fitness, progress, progress_fit = ga.optimize_by_ga_main(
            low=benchmark.interval[0], high=benchmark.interval[1], fitness_func=benchmark.fitness_func, new_population=new_population, num_generation=num_generation, num_parents=num_par, p_c=0.9, p_m=p_m)
        # for RCGA--------------------------------------------------
        new_populationImproved, fitnessImproved, progressImproved, progress_fitImproved = gentic_subpop_step3(interval=benchmark.interval, n_dim=num_var, m=m, n_sub_pop=num_sub_pop, MaxGen1=MaxGen1,
                                                                                                              fitness_func=benchmark.fitness_func, p_c=p_c, MaxGen2=MaxGen2, v0=v0, n_pop=20)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(fitness == np.max(fitness))[0][0]
        print(best_match_idx)
        print(f"{benchmark_name[i]}: Best solution : ",
              new_population[best_match_idx, :])
        tmp = np.empty(shape=(1, num_var))
        tmp[0, :] = new_population[best_match_idx, :]
        print(f"{benchmark_name[i]}: Best solution output : ", np.round(
            benchmark.fitness_func(tmp, mode=1), 10))
        error_classic[i] += np.round(
            benchmark.fitness_func(tmp, mode=1), 10)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(
            fitnessImproved == np.max(fitnessImproved))[0][0]
        print(best_match_idx)
        print(f"{benchmark_name[i]}: Best solution : ",
              new_populationImproved[best_match_idx, :])
        tmp = np.empty(shape=(1, num_var))
        tmp[0, :] = new_populationImproved[best_match_idx, :]
        print(f"{benchmark_name[i]}: Best solution output : ", np.round(
            benchmark.fitness_func(tmp, mode=1), 10))

        error_RCGA[i] += np.round(
            benchmark.fitness_func(tmp, mode=1), 10)

        plt.subplot(2, 2, i+1)
        # plt.plot(progress, 'b', label='classic function')
        # plt.plot(progressImproved, 'r', label='RCGA function')
        plt.plot(progress_fit, 'b--', label='classic fitness')
        plt.plot(progress_fitImproved, 'r--', label='RCGA fitness')
        plt.title(benchmark_name[i])
        plt.ylabel('fitness')
        plt.xlabel('Generation')
        plt.legend()
    plt.show()

# for i in range(len(benchmark_name)):
#     print(
#         f'classic method: average error for {benchmark_name[i]}: {error_classic[i]/51}')
#     print(
#         f'improved method: average error for {benchmark_name[i]}: {error_RCGA[i]/51}')
