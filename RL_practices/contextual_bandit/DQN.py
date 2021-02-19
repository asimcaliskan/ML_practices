import torch
import numpy as np
import BanditEnvironment
import random

class DQN(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size, learning_rate):
        super(DQN, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.learning_rate = learning_rate
        self.init_model()

    def init_model(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, self.hid_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hid_size, self.out_size),
            torch.nn.ReLU()
        )
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_parameter(self):
        return self.model.parameters()

    def forward(self, state):
        self.action = self.model(state)

    def get_action(self, state, use_softmax = False, epsilon = 0.2):
        self.forward(state)
        if(use_softmax):
            softmax_predict = self.softmax(self.action.data.numpy())
            return np.random.choice(self.out_size, p=softmax_predict)
        else:
            if random.random() < epsilon:
                return np.random.randint(self.out_size)
            else:
                return np.argmax(self.action.data.numpy())

    def backward(self, y_predict, y_truth):
        loss = self.loss_func(y_predict, y_truth)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def print_parameters(self):
        for p in self.model.parameters():
            if p.requires_grad:
                print(p.name, p.data)

    def softmax(self, vec, tau=1.00):
        return np.exp(vec / tau) / np.sum(np.exp(vec / tau))
