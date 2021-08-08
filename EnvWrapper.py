import time

from mod_gym import gym

from linetrack.dqn.agent import Pseudo_Agent, Agent
from linetrack.model.model import Linetrack
from mod_stable_baselines3.stable_baselines3 import DQN, PPO


class Wrapper():
    def __init__(self, env_identifier):
        self.env_iden = env_identifier
        self.env = None
        self.initial_state = None
        self.model = None
        self.seed_policy = None

    def create_seed_policy(self, load_path):
        model = None
        if self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            model = PPO.load(load_path, env=self.env)
        elif self.env_iden == "lunar":
            model = DQN.load(load_path, env=self.env)

        self.seed_policy = model

    def create_bipedal_environment(self, seed, hardcore=False):
        if hardcore:
            env = gym.make('BipedalWalkerHardcore-v3')
        else:
            env = gym.make('BipedalWalker-v3')
        env.seed(seed)
        self.env = env
        # self.action_space = range(env.action_space.n)  # Discrete(4)

    def create_bipedal_model(self, load_path):
        model = PPO.load(load_path, env=self.env)
        self.model = model

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
        elif self.env_iden == "bipedal":
            self.create_bipedal_environment(env_seed)
        elif self.env_iden == "bipedal-hc":
            self.create_bipedal_environment(env_seed, hardcore=True)
        elif self.env_iden == "linetrack":
            self.create_linetrack_environment(rng)

    def create_model(self, load_path, r_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_model(load_path)
        elif self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            self.create_bipedal_model(load_path)
        elif self.env_iden == "linetrack":
            self.create_linetrack_model(load_path, r_seed)

    def get_state(self):
        nn_state, hi_lvl_state = None, None

        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            nn_state, hi_lvl_state = self.env.get_state()
        elif self.env_iden == "linetrack":
            nn_state, hi_lvl_state = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)

        return nn_state, hi_lvl_state

    def set_state(self, hi_lvl_state, extra=None):
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            self.env.reset(hi_lvl_state=hi_lvl_state)
        elif self.env_iden == "linetrack":
            self.env.set_state(hi_lvl_state, extra)


    def model_step(self, state):
        act = None
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            act, _ = self.model.predict(state, deterministic=True)
        elif self.env_iden == "linetrack":
            act = self.model.act(state)

        return act

    def env_step(self, action):
        reward, next_state, done = None, None, None
        if self.env_iden == "lunar" or self.env_iden == "bipedal"  or self.env_iden == "bipedal-hc":
            next_state, reward, done, info = self.env.step(action)
        elif self.env_iden == "linetrack":
            next_state, reward, done = self.env.step(action)

        return reward, next_state, done

    def run_pol_fuzz(self, init_state, mode="quantitative", render=False):
        next_state = init_state
        full_play = []
        total_reward = 0
        all_rews = []
        while True:
            act = self.model_step(next_state)
            reward, next_state, done = self.env_step(act)
            if render:
                self.env.render()
                time.sleep(0.02)

            total_reward += reward
            all_rews.append(reward)

            full_play.append(act)
            if done:
                if mode == "qualitative":
                    if -100 in all_rews:
                        total_reward = 0 # walker fell before reaching end
                    else:
                        total_reward = 100  # walker reached end
                    # total_reward = int(total_reward > 0) * 100  # if no crash 100 else 0
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
