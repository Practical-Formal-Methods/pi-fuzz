from abc import ABC
from fuzz_config import POOL_POP_MUT

class Mutator(ABC):
    def __init__(self, wrapper):
        self.wrapper = wrapper

class RandomActionMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed, rng):
        nn_state = seed.data
        self.wrapper.env.set_state(seed.state_env, nn_state[-1])
        for _ in range(POOL_POP_MUT):
            act = rng.choice(self.wrapper.env.action_space, 1)
            _, nn_state, done = self.wrapper.env.step(act)
            if done:
                return None, None

        nn_state, env_state = self.wrapper.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)

        return env_state, nn_state
