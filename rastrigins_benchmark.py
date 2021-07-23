import numpy as np
from math import pi
from sphere_benchmark import Sphere


class Rastrigins(Sphere):
    def __init__(self, num_var, n_pop):
        super().__init__(num_var, n_pop)

    def fitness_func(self, population, mode=0, a=''):
        # print(a)
        A = 10
        fitness = (A*self.num_var + np.sum(np.power(population, 2) -
                                           A*np.cos(2*pi*population), axis=1))
        if mode == 0:
            return 1/(1+np.abs(fitness))
        elif mode == 1:
            return np.abs(fitness)
