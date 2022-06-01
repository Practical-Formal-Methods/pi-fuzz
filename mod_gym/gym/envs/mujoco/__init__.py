from mod_gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from mod_gym.envs.mujoco.ant import AntEnv
from mod_gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from mod_gym.envs.mujoco.hopper import HopperEnv
from mod_gym.envs.mujoco.walker2d import Walker2dEnv
from mod_gym.envs.mujoco.humanoid import HumanoidEnv
from mod_gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from mod_gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from mod_gym.envs.mujoco.reacher import ReacherEnv
from mod_gym.envs.mujoco.swimmer import SwimmerEnv
from mod_gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from mod_gym.envs.mujoco.pusher import PusherEnv
from mod_gym.envs.mujoco.thrower import ThrowerEnv
from mod_gym.envs.mujoco.striker import StrikerEnv
