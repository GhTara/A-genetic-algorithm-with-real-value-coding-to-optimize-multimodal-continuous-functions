import numpy as np


class Sphere:
    def __init__(self, num_var, n_pop):
        self.num_var = num_var
        self.interval = (-5.12, 5.12)
        self.weights = [1 for i in range(num_var)]
        self.random_solutions = np.random.uniform(
            low=self.interval[0], high=self.interval[1], size=(n_pop, num_var))

    def fitness_func(self, population, mode=0, a=''):
        # print(a)
        out = np.dot(np.power(population, 2), self.weights)
        fitness = 1/(np.abs(out)+1)
        if mode == 0:
            return fitness
        elif mode == 1:
            return np.abs(out)
