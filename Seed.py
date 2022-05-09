import copy
import numpy as np

class Seed(object):
    def __init__(self, nn_state, hi_lvl_state, rand_state, gen_trial, gen_time):
        self.identifier = None
        self.data = nn_state
        self.hi_lvl_state = copy.copy(hi_lvl_state)
        self.rand_state = rand_state
        self.reward = 0
        self.energy = 1
        self.weight = 0
        self.parent_seed = None
        self.gen_time = gen_time
        self.gen_trial = gen_trial
        self.is_crash = False
