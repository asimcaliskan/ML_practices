import BanditEnvironment
import DQN
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def np_to_torch(np_array):
    return torch.Tensor(np_array)

EPOCHS = 2000
EPSILON = 0.2
LEARNING_RATE = 0.01
ACTION_SIZE = 10
STATE_SIZE = 10

agent = DQN.DQN(10, 100, 10, LEARNING_RATE)
env = BanditEnvironment.BanditEnv(STATE_SIZE, ACTION_SIZE)

rewards = []
for _ in tqdm(range(EPOCHS)):
    state = env.get_one_hot_state()#it is a vector like -> [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
    action = agent.get_action(np_to_torch(state), use_softmax=False)
    reward = env.make_action(action)
    rewards.append(reward)
    one_hot_reward = env.get_one_hot_state(reward)
    agent.backward(agent.action, np_to_torch(one_hot_reward))

mean_rewards = []
mean = 0
for t in range(len(rewards)):
    mean = (mean * t + rewards[t]) / (t + 1)
    mean_rewards.append(mean)


plt.scatter(np.arange(len(rewards)), rewards, color = '#123456')
plt.title("Contextual Bandit")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.show()
