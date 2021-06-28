import copy
import numpy as np

class Seed(object):
    def __init__(self, state_nn, state_env):
        self.data = state_nn
        self.state_env = copy.copy(state_env)
        self.reward = 0
        self.energy = 1
        self.weight = 0
        self.parent_seed = None
        self.num_warn_la = 0
        self.num_warn_mm_hard = 0
        self.num_warn_mm_easy = 0