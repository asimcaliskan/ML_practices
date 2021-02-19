import numpy as np
import random 
class BanditEnv:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.init_environment()
        self.update_state()

    def init_environment(self):
        self.space_action_value_mat = np.random.rand(self.state_size, self.action_size)

    def update_state(self):
        self.state = np.random.randint(self.state_size)

    def get_state(self):
        return self.state
    
    def get_one_hot_state(self, value=1):
        result = np.zeros(self.state_size)
        result[self.state] = value
        return result

    def update_reward(self, state, action):
        reward = 0
        for _ in range(self.action_size):
            if random.random() < self.space_action_value_mat[state][action]:
                reward += 1
        self.reward = reward

    def make_action(self, action):
        self.update_reward(self.state, action)
        self.update_state()
        return self.reward
