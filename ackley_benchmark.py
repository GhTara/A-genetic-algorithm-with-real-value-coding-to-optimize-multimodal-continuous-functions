import numpy as np
from math import pi
from sphere_benchmark import Sphere


class ACKLEY(Sphere):
    def __init__(self, num_var, n_pop):
        super().__init__(num_var, n_pop)
        self.interval = (-32.768, 32.768)

    def fitness_func(self, population, mode=0, a=''):
        # print(a)
        # print(np.pow
        # er(population, 2).shape)
        a, b, c, d = 20, 0.2, 2*pi, self.num_var
        fitness = -a*(np.exp(-b*np.sqrt((1/d)*np.sum(np.power(population, 2), axis=1)))) -\
            np.exp((1/d)*np.dot(np.cos(c*population), np.ones(shape=(self.num_var)))) +\
            a + np.exp(1)
        # print(fitness.shape)
        if mode == 0:
            return 1/(1+np.abs(fitness))
        elif mode == 1:
            return np.round(np.abs(fitness), 2)
