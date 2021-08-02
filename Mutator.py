import time
from abc import ABC
from fuzz_config import POOL_POP_MUT

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
