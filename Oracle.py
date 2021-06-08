import copy
import itertools
import numpy as np
from fuzz_config import MAX_DEVIATION_DEPTH, DEVIATION_SAMPLE_SIZE, MM_MUT_MAGNITUDE
from abc import ABC, abstractmethod

class Oracle(ABC):

    def __init__(self, game):
        super().__init__()
        self.game = game

    def set_deviations(self):
        deviations = list(itertools.product(self.game.action_space, repeat=MAX_DEVIATION_DEPTH))

        if len(deviations) > DEVIATION_SAMPLE_SIZE:
            deviations = np.random.choice(deviations, DEVIATION_SAMPLE_SIZE, replace=False)

        self.deviations = deviations

    @abstractmethod
    def explore(self, fuzz_seed, random_seed):
        pass


class LookAheadOracle(Oracle):
    def __init__(self, game):
        super().__init__(game)

    def explore(self, fuzz_seed, random_seed):
        super().set_deviations()
        num_warning = 0
        devs = []
        agent_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, lahead_seq=[])
        for deviation in self.deviations:
            # self.game.env.reset(random_seed=random_seed)
            self.game.env.set_state(fuzz_seed.state_env, fuzz_seed.data[-1])
            dev_reward, dev_state, _ = self.game.run_pol_fuzz(fuzz_seed.data, lahead_seq=deviation)

            if dev_state is not None:
                devs.append(dev_state)
            if dev_reward > agent_reward:
                num_warning += 1

        return num_warning, devs


class MetamorphicOracle(Oracle):
    def __init__(self, game):
        super().__init__(game)

    def explore(self, fuzz_seed, random_seed):

        agent_reward, _, _ = self.game.run_pol_fuzz(fuzz_seed.data, lahead_seq=[])

        v = fuzz_seed.data[-1]
        street = copy.deepcopy(fuzz_seed.state_env)
        car_positions = []
        for lane_id, lane in enumerate(street):
            for spot_id, spot in enumerate(lane):
                if (spot is not None) and (str(spot) != "A"):
                    car_positions.append((lane_id, spot_id))

        num_warning = 0
        for i in range(DEVIATION_SAMPLE_SIZE):
            mut_ind = np.random.choice(len(car_positions), MM_MUT_MAGNITUDE, replace=False)
            mut_positions = np.array(car_positions)[mut_ind]

            for pos in mut_positions:
                street[pos[0]][pos[1]] = None

            self.game.env.set_state(street, v)
            state_nn, _ = self.game.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)
            mut_reward, _, _ = self.game.run_pol_fuzz(state_nn, lahead_seq=[])
            if agent_reward > mut_reward:
                num_warning += 1

            street = copy.deepcopy(fuzz_seed.state_env)

        return num_warning, []

class OptimalOracle(Oracle):
    def explore(self, seed, random_seed):
        pass
