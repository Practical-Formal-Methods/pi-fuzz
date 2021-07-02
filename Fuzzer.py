import time
import torch
import logging
import numpy as np
from Seed import Seed
from fuzz_config import FUZZ_RNG, POOL_BUDGET, COV_DIST_THOLD, ORACLE_RNG

logger = logging.getLogger('fuzz_logger')

class Fuzzer:
    def __init__(self, fuzz_type, fuzz_game, schedule, la_oracle, mm_oracle, mutator, coverage):

        self.fuzz_type = fuzz_type
        self.schedule = schedule
        self.mutator = mutator
        self.game = fuzz_game
        self.la_oracle = la_oracle
        self.mm_oracle = mm_oracle
        self.cov_type = coverage

        self.pool = []
        self.epochs = 0
        self.warning_cnt = 0

    # @profile
    def fuzz(self):
        print(FUZZ_RNG.random())
        print(ORACLE_RNG.random())
        exit()
        # time start
        population_summary = self.populate_pool()
        # time end

        logger.info("Pool Budget: %d, Size of the Pool: %d" % (POOL_BUDGET, len(self.pool)))

        # warnings_la = []
        warnings_mm = []
        # fuzz_seed = self.schedule.choose(self.pool)
        for idx, fuzz_seed in enumerate(self.pool): #cur_time - fuzz_start < TIME_BOUND:
            self.epochs += 1

            self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])

            # num_warn_la = self.la_oracle.explore(fuzz_seed)
            num_warn_mm_e, num_warn_mm_h = self.mm_oracle.explore(fuzz_seed)
            num_warn_mm_tot = num_warn_mm_e + num_warn_mm_h

            # fuzz_seed.num_warn_la = num_warn_la
            fuzz_seed.num_warn_mm_hard = num_warn_mm_h
            fuzz_seed.num_warn_mm_easy = num_warn_mm_e

            # warnings_la.append(num_warn_la)
            warnings_mm.append(num_warn_mm_tot)

            # logger.info("\nLookahead Oracle has found %d warnings in state %d" % (num_warn_la, idx))
            logger.info("Metamorphic Oracle has found %d(E) + %d(H) = %d warnings in state %d" % (num_warn_mm_e, num_warn_mm_h, num_warn_mm_tot, idx))

        return warnings_mm, population_summary  # warnings_la,

    def populate_pool(self):
        population_summary = []
        self.game.env.reset()
        state_nn, state_env = self.game.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
        seed = Seed(state_nn, state_env, 0)
        self.pool.append(seed)

        start_time = time.perf_counter()
        trial = 0
        while (time.perf_counter() - start_time) < POOL_BUDGET:
            trial += 1
            rnd = FUZZ_RNG.random()

            if not self.fuzz_type == "bbox" and rnd < 0.7:  # for BB if False
                seed = self.schedule.choose(self.pool)
                cand_env, cand_nn = self.mutator.mutate(seed)
            else:
                self.game.env.reset()
                cand_nn, cand_env = self.game.env.get_state(one_hot=True, linearize=True, window=True, distance=True)

            # time start
            if cand_nn is not None and self.is_interesting(cand_nn):
                self.pool.append(Seed(cand_nn, cand_env, trial, time.perf_counter()-start_time))

            population_summary.append((time.perf_counter()-start_time, trial, len(self.pool)))
            # time end

        return population_summary

    def is_interesting(self, cand):
        if self.cov_type == "abs":
            cand = torch.tensor(cand).float()
            cand = self.game.model.qnetwork_target.hidden(cand)
            cand = cand.detach().numpy()

        d_shortest = np.inf
        for ex_sd in self.pool:
            ex_sd_data = ex_sd.data
            if self.cov_type == "abs":
                ex_sd_data = torch.tensor(ex_sd_data).float()
                ex_sd_data = self.game.model.qnetwork_target.hidden(ex_sd_data)
                ex_sd_data = ex_sd_data.detach().numpy()
            dist = np.linalg.norm(cand - ex_sd_data, ord=2)
            if dist < d_shortest:
                d_shortest = dist

        if d_shortest > COV_DIST_THOLD:
            return True

        return False
