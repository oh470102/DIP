import user_env_gym.double_pendulum as dpend
from utils import *
from tqdm import tqdm
from collections import deque
from Model import *
import gym
import torch
import copy
import matplotlib.pyplot as plt
from datetime import datetime

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

### Train
#env = dpend.DoublePendEnv(reward_mode=0, render_mode='None')
env = gym.make('InvertedDoublePendulum-v4')
agent = SACAgent(env=env)

epochs = 1000#int(input("EPOCHS: "))
agent.train(epochs)

plt.ioff()
plt.plot(agent.save_epi_reward)