import numpy as np
from Seed import Seed
from abc import ABC, abstractmethod

class Scheduler(ABC):
    def __init__(self):
        super().__init__()
        self.cycles = 0

    # @abstractmethod
    def filter(self, pool):
        filter_pool = []
        for seed in pool:
            if seed.energy > 0:
                filter_pool.append(seed)

        # if all seed energy is 0 start over
        if len(filter_pool) == 0:
            self.cycles += 1
            for seed in pool:
                seed.energy = 1
            return pool

        return filter_pool

class QueueScheduler(Scheduler):
    def choose(self, pool):
        pool = super().filter(pool)

        if not pool:
            return None

        seed = pool[0]
        seed.energy = 0  # set its energy to 0 so that never use it again
        return seed


class RandomScheduler(Scheduler):
    # using random scheduler can be problematic. check pool population code
    def choose(self, pool, rng):
        pool = super().filter(pool)
        if not pool:
            return None
        else:
            seed = rng.choice(pool)
            # seed.energy = 0
            return seed


class PowerScheduler(Scheduler):
    def normalized_weight(self, pool):
        """Normalize weight"""
        weights = list(map(lambda seed: seed.weight, pool))
        sum_weights = sum(weights)  # Add up all values in weight
        if sum_weights == 0:
            norm_weight = [1/len(weights) for _ in weights]
        else:
            norm_weight = list(map(lambda nrg: nrg / sum_weights, weights))
        return norm_weight

    def filter_pool(self, pool):
        """Filter out seeds whose energy became zero from the pool"""
        pool = [seed for seed in pool if seed.energy > 0]
        return pool

    def choose(self, pool):
        """Choose weighted by normalized weight."""
        pool = self.filter_pool(pool)
        norm_weight = self.normalized_weight(pool)
        seed = np.random.choice(pool, p=norm_weight)
        return seed