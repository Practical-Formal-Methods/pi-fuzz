import time
import torch
import logging
import numpy as np

import Mutator
import Scheduler
from Seed import Seed
from config import FUZZ_BUDGET

logger = logging.getLogger('fuzz_logger')

class Fuzzer:
    def __init__(self, rand_seed, fuzz_type, fuzz_game, inf_prob, coverage, coverage_thold, mut_budget):

        self.rng = np.random.default_rng(rand_seed)
        self.fuzz_type = fuzz_type
        self.game = fuzz_game
        self.cov_type = coverage
        self.cov_thold = coverage_thold
        self.inf_prob = inf_prob

        self.pool = []
        self.epochs = 0
        self.warning_cnt = 0

        self.random_action_mutator = Mutator.RandomActionMutator(self.game, mut_budget)
        self.seed_policy_mutator = Mutator.SeedPolicyMutator(self.game, mut_budget)
        self.schedule = Scheduler.RandomScheduler()

    # @profile
    def fuzz(self):
        population_summary = []
        self.game.env.reset()
        nn_state, hi_lvl_state, rand_state = self.game.get_state()
        seed = Seed(nn_state, hi_lvl_state, rand_state, 0, 0)
        seed.identifier = len(self.pool)
        self.pool.append(seed)

        time_wout_cov = 0
        start_time = time.perf_counter()
        trial = 0
        while (time.perf_counter() - start_time) < FUZZ_BUDGET:
            rnd = self.rng.random()

            s = time.time()
            if self.fuzz_type == "inc" and rnd < 0.8:
                seed = self.schedule.choose(self.pool, self.rng)
            else:
                self.game.env.reset()  # rng=self.rng)
                cand_nn, cand_hi_lvl, rand_state = self.game.get_state()
                seed = Seed(cand_nn, cand_hi_lvl, rand_state, trial, time.perf_counter()-start_time)

            rnd = self.rng.random()
            if rnd < self.inf_prob:
                cand_nn, cand_hi_lvl, rand_state = self.seed_policy_mutator.mutate(seed, self.rng)
            else:
                cand_nn, cand_hi_lvl, rand_state = self.random_action_mutator.mutate(seed, self.rng)
        
            if cand_nn is None: continue

            time_wout_cov += time.time() - s
        
            is_int = self.is_interesting(cand_nn)

            trial += 1
            if cand_nn is not None and is_int:
                cur_time = time.perf_counter()-start_time
                new_seed = Seed(cand_nn, cand_hi_lvl, rand_state, trial, cur_time)
                new_seed.identifier = len(self.pool)
                self.pool.append(new_seed)
                logger.info("New seed found at trial %d and time %s. Pool size: %d." % (trial, str(cur_time), len(self.pool)))
                population_summary.append([trial, cur_time, len(self.pool)])

        logger.info("Avg time spent in each fuzzer loop iteration excluding coverage: %f", time_wout_cov / trial) 
        logger.info("Avg time spent in each fuzzer loop iteration including coverage: %f", FUZZ_BUDGET / trial) 

        logger.info("Pool Budget: %d, Size of the Pool: %d" % (FUZZ_BUDGET, len(self.pool)))
        # note that this gets bigger and bigger for larger FUZZ_BUDGET
        logger.info("Avg time needed add one more seed to the pool: %f", FUZZ_BUDGET / len(self.pool)) 

        self.total_trials = trial
        self.summary = population_summary
        self.time_wout_cov = time_wout_cov


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
            dist = np.linalg.norm(np.array(cand) - np.array(ex_sd_data), ord=2)
            if dist < d_shortest:
                d_shortest = dist

        if d_shortest > self.cov_thold:
            return True

        return False
