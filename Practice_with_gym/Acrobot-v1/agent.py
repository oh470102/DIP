import gym
from utils import *
from Model import *
import matplotlib.pyplot as plt


### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

### Train
env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
agent = SACAgent(env=env)

epochs = 1000#int(input("EPOCHS: "))
trained_agent = agent.train(epochs)

showcase(trained_agent)

plt.ioff()
