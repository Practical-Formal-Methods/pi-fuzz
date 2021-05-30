import sys

sys.path.append("..")
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np
import highway.model.constants as c

from highway.model.model import Highway

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define neural network
class Network(nn.Module):
    def __init__(self, rseed):
        torch.manual_seed(rseed)
        super().__init__()
        self.fc1 = nn.Linear(170, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 5)
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def hidden(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


# Replay Buffer used for Experience Replay
class ReplayBuffer:
    # initialize the Replay buffer
    # Fix the seed for random sampling through the Replay Buffer
    # Batch size defines number of samples drawn at each learning operation
    # buffer is the actual buffer
    def __init__(self, buffer_size, batch_size, rng):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.rng = rng

    # add samples to the buffer
    # transfer the done value to an integer
    def add(self, state, action, reward, next_state, done):
        if done:
            done_value = 1
        else:
            done_value = 0
        self.buffer.append([state, action, reward, next_state, done_value])

    # sample from the database
    # the samples later need to be split into tensors of each part of the samples
    # thus, collects a sample and writes every part of the sample in the corresponding list
    # afterwards transforms this lists into tensors and returns them
    def sample(self):
        samples = self.rng.choice(self.buffer, self.batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for sample in samples:
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.tensor(states).float()
        actions = torch.LongTensor(actions)
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).float()

        return [states, actions, rewards, next_states, dones]

    # carries over the length of the buffer to the replay buffer
    def __len__(self):
        return len(self.buffer)


class Agent:
    # defines both, the local and the target network
    # defines the optimizer. Mostly, Adam is used
    # initializes buffer and update_counter
    # initialize hyper-parameters of learning process
    # def __init__(self, seed = 0, num_lines = 2, length_lines = 10, ratio=0.2, equally_spread=True, constant_distance=False, n_episodes=1000, l_episodes=100, checkpoint_name='Unnamed', eps_start=1.0, eps_end=0.0001, eps_decay=0.999, learning_count=0):
    def __init__(
        self,
        env,
        r_seed,
        n_episodes=1000,
        l_episodes=100,
        checkpoint_name="Unnamed",
        eps_start=1.0,
        eps_end=0.0001,
        eps_decay=0.999,
        learning_count=0,
    ):
        self.rng = np.random.default_rng(r_seed)
       
        self.qnetwork_target = Network(r_seed)
        self.qnetwork_local = Network(r_seed)
        self.qnetwork_target.to(device)
        self.qnetwork_local.to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=c.LR)
        self.buffer = ReplayBuffer(c.BUFFER_SIZE, c.BATCH_SIZE, self.rng)

        self.update_counter = 0

        # hyperparameters of learning:
        self.n_episodes = n_episodes
        self.l_episodes = l_episodes
        self.checkpoint_name = checkpoint_name
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.env = env
        
        # variables for learning
        self.learning_count = learning_count
        self.best_score = -float("inf")

    # used after init to load an existing network
    def load(self, file, best_score):
        net = Network(123)  # placeholder rseed nn weights loaded anyways
        net.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
        net.eval()
        self.qnetwork_local = net
        self.qnetwork_target = net
        self.best_score = best_score

        self.qnetwork_target.to(device)
        self.qnetwork_local.to(device)

    # carries out one step of the agent
    def step(self, state, action, reward, next_state, done):
        #state = torch.tensor(state)
        #next_state = torch.tensor(next_state)
        
        # add the sample to the buffer
        self.buffer.add(state, action, reward, next_state, done)

        # increment update counter
        self.update_counter = (self.update_counter + 1) % c.UPDATE_EVERY

        # if the update counter mets the requirement of UPDATE_EVERY,
        # sample and start the learning process
        if self.update_counter == 0:
            if (len(self.buffer)) > c.BATCH_SIZE:
                samples = self.buffer.sample()
                self.learn(samples, c.GAMMA)

    # act epsilon greedy according to the local network
    def act(self, state, eps=0):
        state = torch.tensor(state).float()
        state = state.to(device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        action_values = action_values.cpu()

        rnd = self.rng.random()
        if rnd > eps:
            return np.argmax(action_values.numpy())
        else:
            return self.rng.choice(range(len(action_values.numpy())))

    # learn method
    def learn(self, samples, gamma):
        states, actions, rewards, next_states, dones = samples
        
        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        # Implementation of dqn algorithm
        q_values_next_states = self.qnetwork_target.forward(next_states).max(dim=1)[0]
        targets = rewards + (gamma * (q_values_next_states) * (1 - dones))
        q_values = self.qnetwork_local.forward(states)

        actions = actions.view(actions.size()[0], 1)
        predictions = torch.gather(q_values, 1, actions).view(actions.size()[0])

        # calculate loss between targets and predictions
        loss = F.mse_loss(predictions, targets)

        # make backward step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # perform a soft-update to the network
        for target_weight, local_weight in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_weight.data.copy_(
                c.TAU * local_weight.data + (1.0 - c.TAU) * target_weight.data
            )

        # heart of the agent, learning process

    def train(self):
        self.learning_count += 1
        f = open(self.checkpoint_name + ".scores", "a")
        f.write(
            "Start Training Run: eps_start = "
            + str(self.eps_start)
            + " eps_end = "
            + str(self.eps_end)
            + " eps_decay: "
            + str(self.eps_decay)
            + "ReplayBuffer size: "
            + str(c.BUFFER_SIZE)
        )
        f.close()

        # initialize arrays and values
        means = []
        scores = []
        scores_window = deque(maxlen=100)
        eps = self.eps_start

        # initialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        test_pool = []
        for _ in range(100):
            self.env.reset()
            nn_state, env_state = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)

            test_pool.append((env_state, nn_state[-1]))

        score = self.play(test_pool)  # np.mean(scores_window)

        bad_cnt = 0
        mod_cnt = 0
        good_cnt = 0
        # iterate for initialized number of episodes
        for i_episode in range(1, self.n_episodes + 1):
            # reset state and score
            self.env.reset()

            score = 0
            # make at most max_t steps
            for t in range(self.l_episodes):
                state, _ = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
                action = self.act(state, eps)
                reward, _, done = self.env.step(action)  # send the action to the environment and observe
                next_state, _ = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
                self.step(state, action, reward, next_state, done)
                score += reward * np.power(c.GAMMA, t)
                if done:
                    break
            
            # scores_window.append(score)
            # scores.append(score)
            eps = max(self.eps_end, self.eps_decay * eps)

            if i_episode % 200 == 0:
                score = self.play(test_pool)  # np.mean(scores_window)
                means.append(score)

                # if current score is better, save the network weights and update best seen score
                if score > self.best_score:
                    self.best_score = score
                    torch.save(self.qnetwork_local.state_dict(), self.checkpoint_name + ".pth")

                print(
                    "\rEpisode {}\tAverage Score: {:.2f}\tBest Score: {:.2f}".format(
                        i_episode, score, self.best_score
                    )
                )

                f = open(self.checkpoint_name + ".scores", "a")
                for score in scores_window:
                    f.write(str(score) + "\n")
                f.close()
    
                if i_episode % 1000 == 0:
                    torch.save(
                        self.qnetwork_local.state_dict(), self.checkpoint_name + "_" + str(bad_cnt) + ".pth"
                    )
                    bad_cnt += 1

                '''
                if score > -60 and score < -2.5:
                    torch.save(
                        self.qnetwork_local.state_dict(), self.checkpoint_name + "_bad" + str(bad_cnt) + ".pth"
                    )
                    bad_cnt += 1
                if score > 2.5 and score < 7.5:
                    torch.save(
                        self.qnetwork_local.state_dict(), self.checkpoint_name + "_mod" + str(mod_cnt) + ".pth"
                    )
                    mod_cnt += 1
                if score > 20:
                    torch.save(
                        self.qnetwork_local.state_dict(), self.checkpoint_name + "_good" + str(good_cnt) + ".pth"
                    )
                    good_cnt += 1
                '''

        plt.plot(np.arange(len(means)), means, label="Mean", color="r")
        # plt.scatter(np.arange(len(scores)), scores, label="scores", color="c")

        plt.ylabel("Mean Score")
        plt.xlabel("Episode #")
        plt.savefig(self.checkpoint_name + "_run_" + str(self.learning_count) + ".png")

        return self

    def play(self, test_pool):
        scores = []
        for idx, state_pair in enumerate(test_pool):
            play_rng = np.random.default_rng(idx)
            self.env.reset(rng=play_rng)
            self.env.set_state(state_pair[0], state_pair[1])
            state, _ = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
            
            score = 0
            for t in range(self.l_episodes):
                action = self.act(state)
                reward, _, done = self.env.step(action)
                next_state, _ = self.env.get_state(one_hot=True, linearize=True, window=True, distance=True)
                self.step(state, action, reward, next_state, done)
                score += reward * np.power(c.GAMMA, t)
                state = next_state

                if done:
                    break
            scores.append(score)
        return np.mean(scores)


class Pseudo_Agent:
    def __init__(self, checkpoint_name="Unnamed"):
        self.net = Network()
        self.net.load_state_dict(torch.load(checkpoint_name))
        self.net.eval()

    def act(self, state):
        state = torch.tensor(state).float()
        with torch.no_grad():
            action_values = self.net(state)
        return np.argmax(action_values.numpy())

