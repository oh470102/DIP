from utils import *
from tqdm import tqdm
from collections import deque
from Model import *
import gym
import matplotlib.pyplot as plt

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

# create env
env = gym.make('InvertedDoublePendulum-v4')

# initialize agent, set epoch length
agent = SACAgent(env=env)
epochs = int(input("EPOCHS: "))

# train the agent
scores = agent.train(epochs)
print("------Training Completed------")

# turn off plotting interactive mode
plt.ioff()
plt.plot(agent.save_epi_reward)

# plot moving average 
final_plot(scores)

# see how the trained agent performs
showcase(agent)
