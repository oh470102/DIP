import user_env_gym.double_pendulum as dpend
import torch
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def test_env():
    env = dpend.DoublePendEnv(render_mode="human", reward_mode=1)

    n_episodes = 1
    for episode in range(n_episodes):
        state, _ = env.reset()

        terminated = False
        truncated = False
        action = env.action_space.sample()

        while not terminated and not truncated:
            state = torch.tensor(state, dtype=torch.float32)
            state, reward, terminated, truncated, _ = env.step(action)
            print(reward)
            
            
    env.close() 

def live_plot(scores):
    plt.clf
    plt.plot(scores)
    plt.xlabel('epochs')
    plt.ylabel('scores')
    plt.draw()
    plt.pause(0.001)
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: 
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0
