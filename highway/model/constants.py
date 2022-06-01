
## Constants

# don't change this!
KEEP_SPEED = 0
SPEED_UP = 1
CHANGE_RIGHT = 2
CHANGE_LEFT = 3
SLOW_DOWN = 4

AVAILABLE_ACTIONS = [KEEP_SPEED, SPEED_UP, CHANGE_RIGHT, CHANGE_LEFT, SLOW_DOWN]
###

LOOSE_REWARD = - 100
WIN_REWARD = 100
STEP_REWARD = 0


SPEEDLIMIT_AGENT = 4
SPEEDMIN_AGENT = 2

GRANDMA_SPEED = 1
MANIAC_SPEED = 5

INPUT_STRIPE = 10
WINDOW_STRIPE = 10

# Agent DQN Constants

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 0.001  # 1e-3              # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
