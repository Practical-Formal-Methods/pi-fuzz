import os
import numpy as np
import matplotlib.pyplot as plt
from mod_gym import gym
from mod_stable_baselines3.stable_baselines3 import DQN, PPO, DDPG, A2C, SAC
from mod_stable_baselines3.stable_baselines3.common.evaluation import evaluate_policy
from mod_stable_baselines3.stable_baselines3.common import results_plotter
from mod_stable_baselines3.stable_baselines3.common.monitor import Monitor
from mod_stable_baselines3.stable_baselines3.common.results_plotter import load_results, ts2xy
from mod_stable_baselines3.stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model' + str(rseed))
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
              
                  self.model.save(self.save_path + '_%d' % self.n_calls)

        return True

log_dir = "policies/"
rseed = 55

# Create environment
#env = gym.make('BipedalWalker-v3')
env = gym.make('LunarLander-v2')
env = Monitor(env, log_dir + "/monitor%d.csv" % rseed)
env.seed(rseed)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

time_steps = int(1e6)
# Train the agent
callback = SaveOnBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir)
model.learn(total_timesteps=time_steps, callback=callback)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Lunar")
plt.savefig("LunarPPOTraining%d.pdf" % time_steps)

# Load the trained agent
model = PPO.load(log_dir + "/best_model", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
print(mean_reward, std_reward)

# Enjoy trained agent
env.seed(0)
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
