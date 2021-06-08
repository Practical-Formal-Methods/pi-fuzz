import numpy as np
from Seed import Seed
from abc import ABC, abstractmethod

class Scheduler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose(self, population):
        population = [seed for seed in population if seed.energy > 0]
        return population

class QueueScheduler(Scheduler):
    def choose(self, population):
        population = super().choose(population)
        if not population:
            return None
        else:
            seed = population[0]  # set its energy to 0 so that never use it again
            seed.energy = 0
            return seed


class RandomScheduler(Scheduler):
    def choose(self, population):
        population = super().choose(population)
        if not population:
            return None
        else:
            seed = np.random.choice(population)
            seed.energy = 0
            return seed


class PowerScheduler(Scheduler):
    def normalized_weight(self, population):
        """Normalize weight"""
        weights = list(map(lambda seed: seed.weight, population))
        sum_weights = sum(weights)  # Add up all values in weight
        if sum_weights == 0:
            norm_weight = [1/len(weights) for _ in weights]
        else:
            norm_weight = list(map(lambda nrg: nrg / sum_weights, weights))
        return norm_weight

    def filter_population(self, population):
        """Filter out seeds whose energy became zero from the population"""
        population = [seed for seed in population if seed.energy > 0]
        return population

    def choose(self, population):
        """Choose weighted by normalized weight."""
        population = self.filter_population(population)
        norm_weight = self.normalized_weight(population)
        seed = np.random.choice(population, p=norm_weight)
        return seed