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
            # act = rng.choice(self.wrapper.action_space, 1)[0]
            act = rng.uniform(-1, 1, (4))
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


class BipedalHCOracleStumpMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="easy"):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)

        _, _, _, _, _, _, _, _, _, _, terrain_type_poly, _, _, _ = hi_lvl_state

        poly_list = []
        grass_ind = []
        stump_ind = []
        for idx, tt in enumerate(terrain_type_poly):
            if tt[0] == GRASS:
                grass_ind.append(idx)
            elif tt[0] == STUMP:
                stump_ind.append(idx)

            poly_list.append(tt[1])

        if mode == "easy":
            mut_ind = rng.choice(stump_ind)
            for m_ind in mut_ind:
                ttp = terrain_type_poly[m_ind]
                x, y = ttp[2], ttp[3]
                mut_terrain = (GRASS, None, x, y)
                terrain_type_poly[m_ind] = mut_terrain
        else:
            SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
            TERRAIN_STEP = 14/SCALE

            mut_ind = rng.choice(grass_ind)

            for m_ind in mut_ind:
                ttp = terrain_type_poly[m_ind]
                x, y = ttp[2], ttp[3]
                counter = rng.randint(1, 3)

                stump_poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                ]

                mut_terrain = (STUMP, stump_poly, x, y)
                terrain_type_poly[m_ind] = mut_terrain

        hi_lvl_state[10] = terrain_type_poly

        return hi_lvl_state

class BipedalEasyOracleStumpMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="easy"):
        VIEWPORT_H = 400
        SCALE = 30.0
        TERRAIN_STEP   = 14/SCALE
        TERRAIN_LENGTH = 200     # in steps
        TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
        TERRAIN_STARTPAD = 20    # in steps
        y = TERRAIN_HEIGHT
        velocity = 0.0

        if mode == "easy":
            vel_coeff = 0.6
            rough_coeff = 1
        else:
            vel_coeff = 0.9
            rough_coeff = 1.5

        mut_terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            velocity = vel_coeff*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
            if i > TERRAIN_STARTPAD: velocity += rng.uniform(-rough_coeff, rough_coeff)/SCALE   #1
            y += velocity
            mut_terrain_y.append(y)

        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)
        hi_lvl_state[-5] = mut_terrain_y

        return hi_lvl_state

class LunarOracleVelMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="easy"):
        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)
        _, lander_vel, _, _, _, _, _, _, _ = hi_lvl_state
        diff = rng.random()
        if mode == "easy":
            mut_lander_vel = (lander_vel[0], (1-diff) * lander_vel[1])
        else:
            mut_lander_vel = (lander_vel[0], (1+diff)  * lander_vel[1])

        hi_lvl_state[1] = mut_lander_vel

        return hi_lvl_state

class LunarOracleMoonMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="easy"):
        SCALE = 30.0
        VIEWPORT_H = 400
        H = VIEWPORT_H/SCALE

        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)
        _, _, _, _, _, _, _, _, height = hi_lvl_state
        mut_height = []
        if mode == "hard":
            for i in range(len(height)):
                if i+1 == len(height):
                    sp = 0
                else:
                    sp = i+1
                m_h = height[i]
                if height[i] > height[i-1] and height[i] > height[sp]:
                    m_h = rng.uniform(height[i], H/2)
                if height[i] < height[i-1] and height[i] < height[sp]:
                    m_h = rng.uniform(0, height[i])
                mut_height.append(m_h)
        else:
            for i in range(len(height)):
                if i+1 == len(height):
                    sp = 0
                else:
                    sp = i+1
                m_h = height[i]
                if height[i] > height[i-1] and height[i] > height[sp]:
                    if height[i-1] < height[sp]:
                        m_h = rng.uniform(height[i-1], height[sp])
                    else:
                        m_h = rng.uniform(height[sp], height[i-1])
                if height[i] < height[i-1] and height[i] < height[sp]:
                    if height[i-1] < height[sp]:
                        m_h = rng.uniform(height[i-1], height[sp])
                    else:
                        m_h = rng.uniform(height[sp], height[i-1])

                mut_height.append(m_h)

        hi_lvl_state[-1] = mut_height

        return hi_lvl_state