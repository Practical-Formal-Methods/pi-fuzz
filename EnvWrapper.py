import time

import numpy as np

from mod_gym import gym

from linetrack.dqn.agent import Agent as LinetrackAgent
from linetrack.model.model import Linetrack
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

    def create_highway_model(self, load_path, r_seed):
        ag = LinetrackAgent(self.env, r_seed, n_episodes=10000, l_episodes=300, checkpoint_name='Unnamed', eps_start=1.0, eps_end=0.0001,
                  eps_decay=0.999, learning_count=0)
        ag.load(load_path, None)  # second parameter is useless here
        self.model = ag

    def create_highway_environment(self, rng):
        # environment parameters
        num_lines = 2
        length_lines = 100
        ratios = [0.02, 0.1]
        env = Linetrack(num_lines=num_lines, length_lines=length_lines, rng=rng, mode='line_ratio', ratios=ratios, input_stripe=True)
        self.env = env
        self.action_space = env.action_space

    def create_environment(self, env_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_environment(env_seed)
        elif self.env_iden == "bipedal":
            self.create_bipedal_environment(env_seed)
        elif self.env_iden == "bipedal-hc":
            self.create_bipedal_environment(env_seed, hardcore=True)
        elif self.env_iden == "highway":
            rng = np.random.default_rng(env_seed)
            self.create_highway_environment(rng)

    def create_model(self, load_path, r_seed=None):
        if self.env_iden == "lunar":
            self.create_lunar_model(load_path, r_seed)
        elif self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            self.create_bipedal_model(load_path, r_seed)
        elif self.env_iden == "highway":
            self.create_highway_model(load_path, r_seed)

    def get_state(self):
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            nn_state, hi_lvl_state, rand_state = self.env.get_state()
        elif self.env_iden == "highway":
            # in highway, rand_state is default_rng
            nn_state, street, rand_state = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
            hi_lvl_state = [street, nn_state[-1]]

        return nn_state, hi_lvl_state, rand_state

    def set_state(self, hi_lvl_state, rand_state=None):
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            self.env.reset(hi_lvl_state=hi_lvl_state, rand_state=rand_state)
        elif self.env_iden == "highway":
            # in highway, rand_state is default_rng
            self.env.set_state(hi_lvl_state, rand_state)

    def model_step(self, state, deterministic=True):
        act = None
        if self.env_iden == "lunar" or self.env_iden == "bipedal" or self.env_iden == "bipedal-hc":
            act, _ = self.model.predict(state, deterministic=deterministic)
        elif self.env_iden == "highway":
            act = self.model.act(state)

        return act

    def env_step(self, action):
        reward, next_state, done = None, None, None
        if self.env_iden == "lunar" or self.env_iden == "bipedal"  or self.env_iden == "bipedal-hc":
            next_state, reward, done, info = self.env.step(action)
        elif self.env_iden == "highway":
            reward, next_state, done = self.env.step(action)

        return reward, next_state, done

    def play(self, init_state, render=False):
        next_state = init_state
        full_play = []
        all_rews = []
        while True:            
            act = self.model_step(next_state)

            reward, next_state, done = self.env_step(act)
            if render:
                self.env.render()
                time.sleep(0.01)

            # if self.env_iden == "highway":
            #     total_reward = self.env.acc_return  # += np.power(GAMMA, num_steps) * reward
            # else:
            #     total_reward += reward

            all_rews.append(reward)
            full_play.append(act)

            if done:
                # walker fell before reaching end, lander crashed, car crashed
                if -100 in all_rews:
                    final_rew = 0 
                # walker reached end, lander didnt crash, car didnt crash
                else:
                    final_rew = 100
                return final_rew, full_play, all_rews # visited_states

    def test(self):
        c = 0
        next_state = self.env.reset()
        while c < 200:
            act = [0]  # self.model_step(next_state)
            reward, next_state, done = self.env_step(act)
            if done:
                print(c, reward, "finished")
            self.env.render()
            time.sleep(0.005)
            c += 1