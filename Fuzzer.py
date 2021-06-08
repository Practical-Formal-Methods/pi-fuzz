import time
import numpy as np

from Seed import Seed
from fuzz_config import *


class Fuzzer:
    def __init__(self, fuzz_game, fuzz_type, schedule, oracle):

        self.fuzz_type = fuzz_type
        self.schedule = schedule

        self.pool = []
        self.explored_seeds = []
        self.epochs = 0

        self.game = fuzz_game
        self.oracle = oracle

        self.coverage = []  # list of tuples of length 2048
        self.wrng_prefixes = []
        self.snapshopts = []
        self.deviations = []

        self.warning_cnt = 0
        self.random_expl_divider = 1  # randomness decreases when increased

    # @profile
    def fuzz(self):
        fuzz_start = time.time()
        cur_time = fuzz_start

        random_seed = 0
        while cur_time - fuzz_start < TIME_BOUND:
            self.game.env.reset(random_seed=random_seed)

            self.epochs += 1
            fuzz_seed = self.schedule.choose(self.pool)
            if fuzz_seed is None:
                state_nn, state_env = self.game.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)
                fuzz_seed = Seed(state_nn, state_env)

            self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])

            num_warning, devs = self.oracle.explore(fuzz_seed, random_seed)
            self.cov_populate(devs)

            random_seed += 1
            cur_time = time.time()

            print(cur_time - fuzz_start, num_warning)

    def cov_populate(self, cands):
        for (cand_nn, cand_env) in cands:
            d_shortest = np.inf
            if not self.pool:
                self.pool.append(Seed(cand_nn, cand_env))
                continue
            for seed in self.pool:
                dist = np.linalg.norm(cand_nn - seed.data, ord=2)
                if dist < d_shortest:
                    d_shortest = dist

            if d_shortest > COV_DIST_THOLD:
                self.pool.append(Seed(cand_nn, cand_env))

        return self.pool