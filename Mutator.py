import time
import copy
import numpy as np
from abc import ABC
from fuzz_config import POOL_POP_MUT, MM_MUT_MAGNITUDE

class Mutator(ABC):
    def __init__(self, wrapper):
        self.wrapper = wrapper

class RandomActionMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng):
        self.wrapper.set_state(seed.hi_lvl_state)

        for _ in range(POOL_POP_MUT):
            act = rng.choice(self.wrapper.action_space, 1)[0]
            _, nn_state, done = self.wrapper.env_step(act)
            if done:
                return None, None

        nn_state, hi_lvl_state = self.wrapper.get_state()

        return nn_state, hi_lvl_state

class LinetrackOracleMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="easy"):
        car_positions = []
        free_positions = []
        street = copy.deepcopy(seed.hi_lvl_state)
        for lane_id, lane in enumerate(street):
            for spot_id, spot in enumerate(lane):
                if (spot is not None) and (str(spot) != "A"):
                    car_positions.append((lane_id, spot_id))
                if spot is None:
                    free_positions.append((lane_id, spot_id))

        if mode == "easy":        # remove cars
            mut_ind = rng.choice(len(car_positions), MM_MUT_MAGNITUDE, replace=False)
            mut_positions = np.array(car_positions)[mut_ind]
            for pos in mut_positions:
                street[pos[0]][pos[1]] = None
        else:
            mut_ind = rng.choice(len(free_positions), MM_MUT_MAGNITUDE, replace=False)
            mut_positions = np.array(free_positions)[mut_ind]

            for pos in mut_positions:
                street[pos[0]][pos[1]] = self.wrapper.env.get_new_car(pos[0])

        return street


class LunarOracleMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, mode="easy"):
        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)
        _, lander_vel, _, _, _, _, _, _, _, _ = hi_lvl_state
        if mode == "easy":
            mut_lander_vel = (lander_vel[0], 0.7 * lander_vel[1])
        else:
            mut_lander_vel = (lander_vel[0], 1.3 * lander_vel[1])

        hi_lvl_state[1] = mut_lander_vel

        return hi_lvl_state
