import logging
import itertools
import numpy as np
import Mutator
from fuzz_config import SEARCH_BUDGET
from abc import ABC, abstractmethod

logger = logging.getLogger("fuzz_logger")


class Oracle(ABC):

    def __init__(self, game, mode, r_seed, delta, de_dup):
        super().__init__()
        self.game = game
        self.mode = mode
        self.r_seed = r_seed
        self.rng = np.random.default_rng(r_seed)
        self.delta = delta
        self.de_dup = de_dup

    def set_deviations(self):
        deviations = list(itertools.product(self.game.action_space, repeat=0))

        if len(deviations) > SEARCH_BUDGET:
            deviations = self.rng.choice(deviations, SEARCH_BUDGET, replace=False)

        self.deviations = deviations

    @abstractmethod
    def explore(self, fuzz_seed):
        pass


class MetamorphicOracle(Oracle):
    def __init__(self, game, mode, r_seed, delta=None, orcl_mut_bdgt=None, de_dup=False):
        super().__init__(game, mode, r_seed, delta, de_dup)
        if game.env_iden == "linetrack":
            self.mutator = Mutator.LinetrackOracleMutator(game, orcl_mut_bdgt)
        elif game.env_iden == "lunar":
            self.mutator = Mutator.LunarOracleMoonHeightMutator(game)
        elif game.env_iden == "bipedal":
            self.mutator = Mutator.BipedalEasyOracleMutator(game)

    def explore(self, fuzz_seed):
        self.game.set_state(fuzz_seed.hi_lvl_state)  # [fuzz_seed.state_env, fuzz_seed.data[-1]])
        agent_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, self.mode)

        num_rejects = 0
        num_warning_easy = 0
        num_warning_hard = 0
        bug_states = []
        for idx in range(SEARCH_BUDGET):
            if self.game.env_iden == "linetrack":
                exp_rng = np.random.default_rng(self.r_seed)
                self.game.env.reset(exp_rng)
            else:
                self.game.env.seed(self.r_seed)
            # make map EASIER
            if idx % 2 == 0:
                mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='easy')
                if mut_state is None:
                    num_rejects += 1
                    continue

                self.game.set_state(mut_state)  # linetrack: [street, v])
                nn_state, _ = self.game.get_state()

                mut_reward, _, visited_states = self.game.run_pol_fuzz(nn_state, self.mode)
                if self.de_dup and list(visited_states) in bug_states: continue
                if agent_reward - mut_reward > self.delta:
                    num_warning_easy += 1
                    bug_states.append(list(visited_states))

            # make map HARDER
            else:
                mut_state = self.mutator.mutate(fuzz_seed, self.rng, mode='hard')
                if mut_state is None:
                    num_rejects += 1
                    continue

                self.game.set_state(mut_state)  # linetrack: [street, v])
                nn_state, _ = self.game.get_state()

                mut_reward, _, visited_states = self.game.run_pol_fuzz(nn_state, mode=self.mode)
                if self.de_dup and list(visited_states) in bug_states: continue
                if mut_reward - agent_reward > self.delta:
                    num_warning_hard += 1
                    bug_states.append(list(visited_states))

        return num_warning_easy, num_warning_hard, num_rejects


class OptimalOracle(Oracle):
    def explore(self, seed):
        pass

###################
# # LEGACY CODE # #
###################
class LookAheadOracle(Oracle):
    def __init__(self, game, mode, rng, delta=None, de_dup=False):
        super().__init__(game, mode, rng, de_dup, delta)

    def explore(self, fuzz_seed):
        super().set_deviations()
        self.game.env.reset(rng=self.rng)
        num_warning = 0
        self.game.env.set_state(fuzz_seed)
        agent_reward, _, fp = self.game.run_pol_fuzz(fuzz_seed.data, mode=self.mode)
        # if agent does not crash originally, nothing to do in this mode
        if self.mode == "qualitative" and agent_reward > 0:
            return num_warning  # iow 0

        for deviation in self.deviations:
            self.game.env.set_state(fuzz_seed)
            dev_reward, _, fp = self.game.run_pol_fuzz(fuzz_seed.data, lahead_seq=deviation, mode=self.mode)

            if dev_reward - agent_reward > self.delta:
                num_warning += 1

        return num_warning, 0
