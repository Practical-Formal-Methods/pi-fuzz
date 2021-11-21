import time

import numpy as np

from mod_gym import gym

from linetrack.dqn.agent import Agent as LinetrackAgent
from linetrack.model.model import Linetrack
from mod_racetrack.racetrack_dqn import Agent as RacetrackAgent
from mod_racetrack.environment import Environment as Racetrack
from mod_racetrack.argument_parser import Racetrack_parser
from mod_stable_baselines3.stable_baselines3 import DQN, PPO
from mod_stable_baselines3.stable_baselines3.common.policies import ActorCriticPolicy


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
            model = PPO.load(load_path, env=self.env)

        self.seed_policy = model

    def create_bipedal_environment(self, seed, hardcore=False):
        if hardcore:
            env = gym.make('BipedalWalkerHardcore-v3')
        else:
            env = gym.make('BipedalWalker-v3')
        env.seed(seed)
        self.env = env
        # self.action_space = range(env.action_space.n)  # Discrete(4)

    def create_bipedal_model(self, load_path, r_seed):
        ppo = PPO(env=self.env, seed=r_seed, policy=ActorCriticPolicy)
        model = ppo.load(load_path, env=self.env)
        self.model = model

    def create_lunar_model(self, load_path, r_seed):
        ppo = PPO(env=self.env, seed=r_seed, policy=ActorCriticPolicy)
        model = ppo.load(load_path, env=self.env)
        # model = PPO.load(load_path, env=self.env)
        # PPO.set_random_seed(r_seed)
        self.model = model

    def create_lunar_environment(self, seed):
        env = gym.make('LunarLander-v2')
        env.seed(seed)
        self.env = env
        self.action_space = range(env.action_space.n)  # Discrete(4)

    def create_linetrack_model(self, load_path, r_seed):
        ag = LinetrackAgent(self.env, r_seed, n_episodes=10000, l_episodes=300, checkpoint_name='Unnamed', eps_start=1.0, eps_end=0.0001,
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

    def create_racetrack_model(self, load_path, r_seed):
        rparser = Racetrack_parser()
        namespace = rparser.parse(["racetrack", "ring", "-s", str(r_seed), "-n", "-rs", "-nr", "-100"])
        ag = RacetrackAgent(self.env, namespace)
        ag.load(load_path)
        self.model = ag

    def create_racetrack_environment(self, r_seed):
        rparser = Racetrack_parser()
        namespace = rparser.parse(["racetrack", "ring", "-s", str(r_seed), "-n", "-rs", "-nr", "-100"])
        env = Racetrack(rt_args=namespace)
        self.env = env
        self.action_space = range(9) 

    def create_environment(self, env_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_environment(env_seed)
        elif self.env_iden == "bipedal":
            self.create_bipedal_environment(env_seed)
        elif self.env_iden == "bipedal-hc":
            self.create_bipedal_environment(env_seed, hardcore=True)
        elif self.env_iden == "linetrack":
            rng = np.random.default_rng(env_seed)
            self.create_linetrack_environment(rng)
        elif self.env_iden == "racetrack":
            self.create_racetrack_environment(env_seed)

    def create_model(self, load_path, r_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_model(load_path, r_seed)
        elif self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            self.create_bipedal_model(load_path, r_seed)
        elif self.env_iden == "linetrack":
            self.create_linetrack_model(load_path, r_seed)
        elif self.env_iden == "racetrack":
            self.create_racetrack_model(load_path, r_seed)

    def get_state(self):
        nn_state, hi_lvl_state = None, None

        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            nn_state, hi_lvl_state = self.env.get_state()
        elif self.env_iden == "linetrack":
            nn_state, street = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
            hi_lvl_state = [street, nn_state[-1]]
        elif self.env_iden == "racetrack":
            nn_state = self.env.get_state()
            hi_lvl_state = self.env.get_high_level_state()

        return nn_state, hi_lvl_state

    def set_state(self, hi_lvl_state):
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            self.env.reset(hi_lvl_state=hi_lvl_state)
        elif self.env_iden == "linetrack":
            self.env.set_state(hi_lvl_state)
        elif self.env_iden == "racetrack":
            position, velocity, path, map_obj, _, _ = hi_lvl_state
            self.env.reset_to_state(position, velocity, map_obj, path)

    def model_step(self, state, deterministic=True):
        act = None
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            act, _ = self.model.predict(state, deterministic=deterministic)
        elif self.env_iden == "linetrack" or self.env_iden == "racetrack":
            act = self.model.act(state)

        return act

    def env_step(self, action):
        reward, next_state, done = None, None, None
        if self.env_iden == "lunar" or self.env_iden == "bipedal"  or self.env_iden == "bipedal-hc":
            next_state, reward, done, info = self.env.step(action)
        elif self.env_iden == "linetrack" or self.env_iden == "racetrack":
            reward, next_state, done = self.env.step(action)

        return reward, next_state, done

    def run_pol_fuzz(self, init_state, mode="qualitative", render=False):
        next_state = init_state
        full_play = []
        all_rews = []
        visited_states = []
        total_reward = 0
        while True:
            
            _, hls = self.get_state()
            n_hls = []
            for elm in hls:
                if isinstance(elm, np.ndarray):
                    elm = list(elm)
                n_hls.append(elm)
           
            visited_states.append(n_hls)
            act = self.model_step(next_state)
            reward, next_state, done = self.env_step(act)
            if render:
                self.env.render()
                time.sleep(0.01)

            if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "racetrack":
                total_reward += reward
            else:
                total_reward = self.env.acc_return  # += np.power(GAMMA, num_steps) * reward

            all_rews.append(reward)
            full_play.append(act)

            # racetrack agent can stuck, thus prevent this
            if self.env_iden == "racetrack" and len(full_play) == 200: 
                return 0, full_play, visited_states

            if done:
                if mode == "qualitative":
                    if -100 in all_rews:
                        total_reward = 0 # walker fell before reaching end, lander crashed
                    else:
                        total_reward = 100  # walker reached end, lander didnt crash
                return total_reward, full_play, visited_states
