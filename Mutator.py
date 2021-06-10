from abc import ABC, abstractmethod
from fuzz_config import POOL_POP_MUT

class Mutator(ABC):
    def __init__(self, wrapper):
        self.wrapper = wrapper

class RandomActionMutator(Mutator):
    def __init__(self, wrapper):
        super().__init__(wrapper)

    def mutate(self, seed):
        self.wrapper.env.set_state(seed)
        nn_state = seed.data
        for _ in range(POOL_POP_MUT):
            act = self.wrapper.model.act(nn_state)
            _, nn_state, done = self.wrapper.env.step(act)
            if done:
                return None, None

        env_state = self.wrapper.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)

        return env_state, nn_state
