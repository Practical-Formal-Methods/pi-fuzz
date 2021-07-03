from linetrack.dqn.agent import Pseudo_Agent, Agent
from linetrack.model.model import Linetrack

class Wrapper():
    def __init__(self, load_path):
        self.load_path = load_path
        self.action_space = range(5)

        self.env = None
        self.initial_state = None
        self.model = None

    def create_linetrack_model(self, rng):
        ag = Agent(self.env, rng, n_episodes=10000, l_episodes=300, checkpoint_name='Unnamed', eps_start=1.0, eps_end=0.0001,
                  eps_decay=0.999, learning_count=0)
        ag.load(self.load_path, None)  # second parameter is useless here
        self.model = ag
        print('Model created.')


    def create_linetrack_environment(self, rng):
        # environment parameters
        num_lines = 2
        length_lines = 100
        ratios = [0.02, 0.1]
        env = Linetrack(num_lines=num_lines, length_lines=length_lines, rng=rng, mode='line_ratio', ratios=ratios, input_stripe=True)
        self.env = env
        print('Environment created.')

    # @profile
    def run_pol_fuzz(self, init_state, lahead_seq=None, mode="quantitative"):
        if lahead_seq is None:
            lahead_seq = []
        next_state = init_state
        dev_state = None
        idx = 0
        full_play = []
        while True:
            if idx < len(lahead_seq):
                act = lahead_seq[idx]
            else:
                act = self.model.act(next_state)

            if idx == len(lahead_seq):
                dev_state = self.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)  # BEWARE: position of this line is important.

            _, next_state, done = self.env.step(act)

            full_play.append(act)
            if done:
                total_reward = self.env.get_discounted_return()
                if mode == "qualitative":
                    total_reward = int(total_reward > 0) * 100  # if no crash 100 else 0
                return total_reward, dev_state, full_play

            idx += 1

    # @profile
    def run_env_fuzz(self, seed=None):  # environment fuzzer
        self.env.reset()
        init_state = self.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)
        next_state = init_state

        coverage = []
        idx = 0
        full_play = []
        # total_reward = 0
        while True:
            if idx == seed:
                self.env.mutate_street(0.2, 3)
                next_state = self.env.get_state(one_hot=True, linearize=True,  window=True, distance=True)

            act = self.model.act(next_state)
            # print(act, end=' ')

            _, next_state, done = self.env.step(act)
            full_play.append(act)
            # total_reward += reward

            if done:
                total_reward = self.env.get_discounted_return()
                return {'usage': True, 'data': (coverage, total_reward, full_play)}

            idx += 1
