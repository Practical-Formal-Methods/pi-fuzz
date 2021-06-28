import copy
import logging
import itertools
import numpy as np
from fuzz_config import DEVIATION_DEPTH, SEARCH_BUDGET, MM_MUT_MAGNITUDE, DELTA, RNG
from abc import ABC, abstractmethod

logger = logging.getLogger("fuzz_logger")


class Oracle(ABC):

    def __init__(self, game, mode):
        super().__init__()
        self.game = game
        self.mode = mode

    def set_deviations(self):
        deviations = list(itertools.product(self.game.action_space, repeat=DEVIATION_DEPTH))

        if len(deviations) > SEARCH_BUDGET:
            deviations = RNG.choice(deviations, SEARCH_BUDGET, replace=False)

        self.deviations = deviations

    @abstractmethod
    def explore(self, fuzz_seed):
        pass


class LookAheadOracle(Oracle):
    def __init__(self, game, mode):
        super().__init__(game, mode)

    def explore(self, fuzz_seed):
        super().set_deviations()
        num_warning = 0
        self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])
        agent_reward, _, fp = self.game.run_pol_fuzz(fuzz_seed.data, mode=self.mode)
        # if agent does not crash originally, nothing to do in this mode
        if self.mode == "qualitative" and agent_reward > 0:
            return num_warning  # iow 0

        for deviation in self.deviations:
            self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])
            dev_reward, _, fp = self.game.run_pol_fuzz(fuzz_seed.data, lahead_seq=deviation, mode=self.mode)

            if dev_reward - agent_reward > DELTA:
                num_warning += 1

        return num_warning


class MetamorphicOracle(Oracle):
    def __init__(self, game, mode):
        super().__init__(game, mode)

    def explore(self, fuzz_seed):
        num_warning_easy = 0
        num_warning_hard = 0
        self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])
        agent_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, mode=self.mode)
        v = fuzz_seed.data[-1]
        street = copy.deepcopy(fuzz_seed.state_env)

        car_positions = []
        free_positions = []
        for lane_id, lane in enumerate(street):
            for spot_id, spot in enumerate(lane):
                if (spot is not None) and (str(spot) != "A"):
                    car_positions.append((lane_id, spot_id))
                if spot is None:
                    free_positions.append((lane_id, spot_id))

        for idx in range(SEARCH_BUDGET):
            # make map EASIER
            if idx % 2 == 0:
                # if we make the map easier and the agent is crashing we cant claim any bug in this mode
                if self.mode == "qualitative" and agent_reward < 0:
                    continue

                mut_ind = RNG.choice(len(car_positions), MM_MUT_MAGNITUDE, replace=False)
                mut_positions = np.array(car_positions)[mut_ind]

                # remove cars
                for pos in mut_positions:
                    street[pos[0]][pos[1]] = None

                self.game.env.set_state(street, v)
                state_nn, _ = self.game.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
                mut_reward, _, _ = self.game.run_pol_fuzz(state_nn, mode=self.mode)

                if agent_reward - mut_reward > DELTA:
                    num_warning_easy += 1
            # make map HARDER
            else:
                # if we make the map harder and the agent is winning we cant claim any bug in this mode
                if self.mode == "qualitative" and agent_reward > 0:
                    continue

                mut_ind = RNG.choice(len(free_positions), MM_MUT_MAGNITUDE, replace=False)
                mut_positions = np.array(free_positions)[mut_ind]

                for pos in mut_positions:
                    street[pos[0]][pos[1]] = self.game.env.get_new_car(pos[0])

                self.game.env.set_state(street, v)
                state_nn, _ = self.game.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)
                mut_reward, _, _ = self.game.run_pol_fuzz(state_nn, mode=self.mode)

                if agent_reward - mut_reward > DELTA:
                    num_warning_hard += 1

            street = copy.deepcopy(fuzz_seed.state_env)

        return num_warning_easy, num_warning_hard

class OptimalOracle(Oracle):
    def explore(self, seed):
        pass
