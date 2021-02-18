"""
To figure out dynamic of multi armed bandit problem
@segfaultian
"""
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
"""
    >we take a random value between 0 and 10
    >if this random value is smaller than epsilon, increases(+1) counter value
    >when N goes positive infinity, we see (counter / N) goes epsilon value
    
    this function acts like an armed bandit. Why?
    >Each play has probabilistic 
    >Maybe we get 10 or 0 
    >If we increase the number of tries on the armed bandit, we will see a mean reward 〜 epsilon value
    >mean gives us value of this machine(action)
"""
def epsilon_prob(epsilon, N):
    counter = 0
    for _ in range(N):
        if random.randint(0,10) < epsilon:
            counter += 1
    return counter / N


number_of_slots = 10
slot_probs = np.random.rand(number_of_slots)
epsilon = 0.2
record = np.zeros((number_of_slots, 2))#first column: average reward, second column: pulled times

def update_record(action, reward):
    total = (record[action][0] * record[action][1]) + reward
    record[action][1] += 1
    record[action][0] = total / record[action][1]

def get_best_bandit():
    return np.argmax(record[:,0], axis=0)

def get_action():
    rand_value = random.random()
    if rand_value < epsilon:
        return np.random.randint(10)
    else:
        return get_best_bandit()

def get_reward(action, n = 10):
    reward = 0
    for _ in range(n):
        if random.random() < slot_probs[action]:
            reward += 1
    return reward

def get_mean_reward():
    return np.sum(record[:,0]) / number_of_slots

"""
    What is the softmax?
    It is a probability distribution over the action values
    >take action vector
    <return probability distribution 
"""
def softmax(value, tau):
    return np.exp(value / tau) / np.sum(np.exp(value / tau))

def test(TEST_SIZE, test_with_softmax = False):
    mean_reward_record = [0]
    plt_x_axis = []
    action_vector = np.arange(10)
    for t in range(TEST_SIZE):
        if not test_with_softmax:
            action = get_action()
        else:
            action = np.random.choice(action_vector, p=softmax(record[:,0], 1.25))
        reward = get_reward(action)
        update_record(action, reward)
        """
        mean_reward = ((old_reward * #elements) + reward) / #elements
        (t+1) because initial #elements == 1 but initial t == 0, because of this t is increased by 1
        """
        mean_reward_record.append((mean_reward_record[t] * (t + 1) + reward) / (t + 2))
        plt_x_axis.append(t)
    plt.scatter(plt_x_axis, mean_reward_record[1:], color = '#88c999')
    plt.title("Multi Armed Bandit Test #:{}, epsilon:{}".format(TEST_SIZE, epsilon))
    plt.xlabel("Time")
    plt.ylabel("Average Reward")
    plt.show()

test(500, True)
#print(softmax(np.array([1, 2, 3, 4]), 1.25))