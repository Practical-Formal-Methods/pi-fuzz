import time
import numpy as np

from Seed import Seed
from fuzz_config import *


class Fuzzer:
    def __init__(self, fuzz_game, schedule, oracle, mutator):

        self.schedule = schedule
        self.mutator = mutator
        self.game = fuzz_game
        self.oracle = oracle

        self.pool = []
        self.epochs = 0
        self.warning_cnt = 0

    # @profile
    def fuzz(self):

        self.populate_pool()

        fuzz_start = time.time()
        cur_time = fuzz_start

        random_seed = 0
        warnings = []
        while cur_time - fuzz_start < TIME_BOUND:

            self.epochs += 1
            fuzz_seed = self.schedule.choose(self.pool)
            self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])
            num_warning = self.oracle.explore(fuzz_seed, random_seed)
            warnings.append(num_warning)

            random_seed += 1
            cur_time = time.time()

        return warnings

    def populate_pool(self):
        # for _ in range(30):
        self.game.env.reset(None)
        state_nn, state_env = self.game.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
        seed = Seed(state_nn, state_env)
        self.pool.append(seed)

        for cnt in range(POOL_BUDGET):

            if np.random.rand() < 0.7:
                seed = self.schedule.choose(self.pool)
                cand_env, cand_nn = self.mutator.mutate(seed)
                if cand_nn is None: continue

                d_shortest = np.inf
                for ex_sd in self.pool:
                    dist = np.linalg.norm(cand_nn - ex_sd.data, ord=2)
                    if dist < d_shortest:
                        d_shortest = dist

                if d_shortest > COV_DIST_THOLD:
                    self.pool.append(Seed(cand_nn, cand_env))
            else:
                self.game.env.reset(None)
                state_nn, state_env = self.game.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
                seed = Seed(state_nn, state_env)
                self.pool.append(seed)
        return self.pool