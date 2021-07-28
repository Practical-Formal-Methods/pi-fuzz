from mod_gym import gym

from linetrack.dqn.agent import Pseudo_Agent, Agent
from linetrack.model.model import Linetrack
from mod_stable_baselines3.stable_baselines3 import DQN

class Wrapper():
    def __init__(self, env_identifier):
        self.env_iden = env_identifier
        self.env = None
        self.initial_state = None
        self.model = None

    def create_lunar_model(self, load_path):
        model = DQN.load(load_path, env=self.env)
        self.model = model

    def create_lunar_environment(self, seed):
        env = gym.make('LunarLander-v2')
        env.seed(seed)
        self.env = env
        self.action_space = range(env.action_space.n)  # Discrete(4)

    def create_linetrack_model(self, load_path, r_seed):
        ag = Agent(self.env, r_seed, n_episodes=10000, l_episodes=300, checkpoint_name='Unnamed', eps_start=1.0, eps_end=0.0001,
                  eps_decay=0.999, learning_count=0)
        ag.load(load_path, None)  # second parameter is useless here
        self.model = ag

    def create_linetrack_environment(self, rng):
        # environment parameters
        num_lines = 2
        length_lines = 100
        ratios = [0.02, 0.1]
        env = Linetrack(num_lines=num_lines, length_lines=length_lines, rng=rng, mode='line_ratio', ratios=ratios, input_stripe=True)
        self.env = env
        self.action_space = env.action_space

    def create_environment(self, rng=None, env_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_environment(env_seed)
        elif self.env_iden == "linetrack":
            self.create_linetrack_environment(rng)

    def create_model(self, load_path, r_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_model(load_path)
        elif self.env_iden == "linetrack":
            self.create_linetrack_model(load_path, r_seed)

    def get_state(self):
        nn_state, hi_lvl_state = None, None

        if self.env_iden == "lunar":
            nn_state, hi_lvl_state = self.env.get_state()
        elif self.env_iden == "linetrack":
            nn_state, hi_lvl_state = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)

        return nn_state, hi_lvl_state

    def set_state(self, state_inf):
        hi_lvl_state, v = state_inf
        if self.env_iden == "lunar":
            self.env.set_state(hi_lvl_state)
        elif self.env_iden == "linetrack":
            self.env.set_state(hi_lvl_state, v)


    def model_step(self, state):
        act = None
        if self.env_iden == "lunar":
            act = self.model.predict(state, deterministic=True)
        elif self.env_iden == "linetrack":
            act = self.model.act(state)

        return act

    def env_step(self, action):
        reward, next_state, done = None, None, None
        if self.env_iden == "lunar":
            reward, next_state, done, info = self.env.step(action)
        elif self.env_iden == "linetrack":
            next_state, reward, done = self.env.step(action)

        return reward, next_state, done

    def run_pol_fuzz(self, init_state, mode="quantitative"):
        next_state = init_state
        full_play = []
        total_reward = 0
        while True:
            act = self.model_step(next_state)
            reward, next_state, done = self.env_step(act)
            total_reward += reward
            full_play.append(act)
            if done:
                if mode == "qualitative":
                    total_reward = int(total_reward > 0) * 100  # if no crash 100 else 0
                return total_reward, full_play

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
