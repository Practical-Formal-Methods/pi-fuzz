from abc import ABC

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
    def choose(self, pool, rng):
        pool = super().filter(pool)
        if not pool:
            return None
        else:
            seed = rng.choice(pool)
            return seed