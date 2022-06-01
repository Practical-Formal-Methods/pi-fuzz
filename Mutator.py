import copy
import numpy as np
from abc import ABC

class Mutator(ABC):
    def __init__(self, wrapper):
        self.wrapper = wrapper

class SeedPolicyMutator(Mutator):
    def __init__(self, wrapper, fuzz_mut_bdgt):
        super().__init__(wrapper)
        self.fuzz_mut_bdgt = fuzz_mut_bdgt

    def mutate(self, seed, rng):
        self.wrapper.set_state(seed.hi_lvl_state, seed.rand_state)
        nn_state, hi_lvl_state, _ = self.wrapper.get_state()
        mut_bdgt = rng.integers(self.fuzz_mut_bdgt)

        next_state = nn_state
        for _ in range(mut_bdgt):
            act = self.wrapper.model_step(next_state, deterministic=False)  # Deliberately stochastic
            _, next_state, done = self.wrapper.env_step(act)
            if done:
                return None, None, None

        nn_state, hi_lvl_state, rand_state = self.wrapper.get_state()
        return nn_state, hi_lvl_state, rand_state

class RandomActionMutator(Mutator):
    def __init__(self, wrapper, fuzz_mut_bdgt):
        super().__init__(wrapper)
        self.fuzz_mut_bdgt = fuzz_mut_bdgt

    def mutate(self, seed, rng):
        self.wrapper.set_state(seed.hi_lvl_state, seed.rand_state)
        mut_bdgt = rng.integers(self.fuzz_mut_bdgt)
        for _ in range(mut_bdgt):
            if self.wrapper.env_iden == "bipedal":
                act = rng.uniform(-1, 1, (4))
            else:
                act = rng.choice(self.wrapper.action_space, 1)[0]
            _, nn_state, done = self.wrapper.env_step(act)
            if done:
                return None, None, None

        nn_state, hi_lvl_state, rand_state = self.wrapper.get_state()

        return nn_state, hi_lvl_state, rand_state

class HighwayOracleMutator(Mutator):
    def __init__(self, wrapper, orcl_mut_bdgt):
        super().__init__(wrapper)
        self.orcl_mut_bdgt = orcl_mut_bdgt

    def mutate(self, seed, rng, mode="relax"):
        car_positions = []
        free_positions = []
        street, v = seed.hi_lvl_state
        street = copy.deepcopy(street)
        for lane_id, lane in enumerate(street):
            for spot_id, spot in enumerate(lane):
                if (spot is not None) and (str(spot) != "A"):
                    car_positions.append((lane_id, spot_id))
                if spot is None:
                    free_positions.append((lane_id, spot_id))

        if mode == "relax":        # remove cars
            mut_ind = rng.choice(len(car_positions), self.orcl_mut_bdgt, replace=False)
            mut_positions = np.array(car_positions)[mut_ind]
            for pos in mut_positions:
                street[pos[0]][pos[1]] = None
        else:
            mut_ind = rng.choice(len(free_positions), self.orcl_mut_bdgt, replace=False)
            mut_positions = np.array(free_positions)[mut_ind]

            for pos in mut_positions:
                street[pos[0]][pos[1]] = self.wrapper.env.get_new_car(pos[0])

        mut_hi_lvl_state = [street, v]

        return mut_hi_lvl_state


class BipedalHCOracleStumpMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="relax"):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)

        terrain_type_poly = hi_lvl_state[-4]

        poly_list = []
        grass_ind = []
        stump_ind = []
        for idx, tt in enumerate(terrain_type_poly):
            if tt[0] == GRASS:
                grass_ind.append(idx)
            elif tt[0] == STUMP:
                stump_ind.append(idx)

            poly_list.append(tt[1])

        if mode == "relax":
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

class BipedalEasyOracleMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="relax"):
        VIEWPORT_H = 400
        SCALE = 30.0
        TERRAIN_STEP   = 14/SCALE
        TERRAIN_LENGTH = 200     # in steps
        TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
        TERRAIN_STARTPAD = 20    # in steps
        y = TERRAIN_HEIGHT
        velocity = 0.0

        if mode == "relax":
            vel_coeff = 0.7
            rough_coeff = 1
        else:
            vel_coeff = 0.9
            rough_coeff = 1

        mut_terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            velocity = vel_coeff*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
            if i > TERRAIN_STARTPAD: velocity += rng.uniform(-rough_coeff, rough_coeff)/SCALE   #1
            y += velocity
            mut_terrain_y.append(y)

        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)
        if len(hi_lvl_state) == 17:
            hi_lvl_state[-5] = mut_terrain_y
        elif len(hi_lvl_state) == 14:
            hi_lvl_state[-2] = mut_terrain_y


        return hi_lvl_state

class LunarOracleMoonHeightMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng, mode="relax"):
        SCALE = 30.0
        VIEWPORT_H = 400
        H = VIEWPORT_H/SCALE
        ORG_HELIPAD_H = H/4
        MARGIN = H/6
        CHUNKS = 11

        hi_lvl_state = copy.deepcopy(seed.hi_lvl_state)
        lander_pos = hi_lvl_state[0]
        _, lander_height = lander_pos

        if mode == "unrelax":
            if lander_height-MARGIN < ORG_HELIPAD_H:
                return None
            mut_helipad_height = rng.uniform(ORG_HELIPAD_H, lander_height-MARGIN)
        else:
            mut_helipad_height = rng.uniform(H/12, ORG_HELIPAD_H)

        mut_height = rng.uniform(0, mut_helipad_height*2, size=(CHUNKS+1,))
        mut_height[CHUNKS//2-2] = mut_helipad_height
        mut_height[CHUNKS//2-1] = mut_helipad_height
        mut_height[CHUNKS//2+0] = mut_helipad_height
        mut_height[CHUNKS//2+1] = mut_helipad_height
        mut_height[CHUNKS//2+2] = mut_helipad_height

        hi_lvl_state[-1] = list(mut_height)

        return hi_lvl_state
