import numpy as np
from sphere_benchmark import Sphere


class BentCigar(Sphere):
    def __init__(self, num_var, n_pop):
        super().__init__(num_var, n_pop)
        self.interval = (-100, 100)
        self.weights = [10**6 for i in range(num_var-1)]

    def fitness_func(self, population, mode=0, a=''):
        # print(a)
        population = np.array(population, dtype=np.float64)
        bias = population[:, 0]
        fitness = (np.power(bias, 2) +
                   np.dot(np.power(population[:, 1:], 2), self.weights))
        # print(fitness.shape)
        if mode == 0:
            return 1/(1+np.abs(fitness))
        elif mode == 1:
            return np.abs(fitness)
