import time
import torch
import logging
import numpy as np

import Mutator
import Scheduler
from Seed import Seed
from fuzz_config import POOL_BUDGET, COV_DIST_THOLD

logger = logging.getLogger('fuzz_logger')

class Fuzzer:
    def __init__(self, rng, fuzz_type, fuzz_game, coverage):

        self.rng = rng
        self.fuzz_type = fuzz_type
        self.game = fuzz_game
        self.cov_type = coverage

        self.pool = []
        self.epochs = 0
        self.warning_cnt = 0

        self.random_action_mutator = Mutator.RandomActionMutator(self.game)
        self.seed_policy_mutator = Mutator.SeedPolicyMutator(self.game)
        self.schedule = Scheduler.QueueScheduler()

    # @profile
    def fuzz(self):
        population_summary = []
        self.game.env.reset()
        nn_state, hi_lvl_state = self.game.get_state()
        seed = Seed(nn_state, hi_lvl_state, 0, 0)
        self.pool.append(seed)

        start_time = time.perf_counter()
        trial = 0
        while (time.perf_counter() - start_time) < POOL_BUDGET:
            trial += 1
            rnd = self.rng.random()

            if rnd < 0.8:
                if self.fuzz_type == "gbox":
                    seed = self.schedule.choose(self.pool)
                else:  # bbox
                    self.game.env.reset()  # rng=self.rng)
                    cand_nn, cand_hi_lvl = self.game.get_state()
                    seed = Seed(cand_nn, cand_hi_lvl, trial, time.perf_counter()-start_time)

                if rnd < 0.4:
                    cand_nn, cand_hi_lvl = self.seed_policy_mutator.mutate(self.game.seed_policy, seed, self.rng)
                else:  # iow ->  rnd >= 0.4 and rnd < 0.8:
                    cand_nn, cand_hi_lvl = self.random_action_mutator.mutate(seed, self.rng)
            else:
                self.game.env.reset()  # rng=self.rng)
                cand_nn, cand_hi_lvl = self.game.get_state()

            # time start
            if self.is_interesting(cand_nn):
                self.pool.append(Seed(cand_nn, cand_hi_lvl, trial, time.perf_counter()-start_time))

            population_summary.append([trial, time.perf_counter()-start_time, len(self.pool)])
            # time end

        logger.info("Pool Budget: %d, Size of the Pool: %d" % (POOL_BUDGET, len(self.pool)))

        return population_summary

    def is_interesting(self, cand):
        if cand is None: return False

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
            dist = np.linalg.norm(np.array(cand) - np.array(ex_sd_data), ord=2)
            if dist < d_shortest:
                d_shortest = dist

        if d_shortest > COV_DIST_THOLD:
            return True

        return False
