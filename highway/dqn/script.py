import sys

sys.path.append("..")
import argparse
from highway.dqn.agent import Agent
from highway.model.model import Highway
import numpy as np

# environment parameters
num_lines = 2
length_lines = 100
ratios = []
seed = 0

parser = argparse.ArgumentParser()
parser.add_argument("agent_id", type=int)
args = parser.parse_args()
agent_id = args.agent_id

for idx in [agent_id]:  # range(35):
    # set the seed
    env_rng = np.random.default_rng(idx)

    env = Highway(
        num_lines=num_lines,
        length_lines=length_lines,
        rng=env_rng,
        mode="line_ratio",
        ratios=[0.02, 0.1],
        random_start=False,
        input_stripe=True,
    )
    a = Agent(
        env,
        r_seed=idx,
        n_episodes=20000,
        l_episodes=300,
        checkpoint_name="final_policies/agent" + str(idx),
        eps_start=1.0,
        eps_end=0.0001,
        eps_decay=0.999,
        learning_count=0,
    )

    a.train()  # needed to train all agents on the same states

    # pic = env.show()
    # pic.show()

