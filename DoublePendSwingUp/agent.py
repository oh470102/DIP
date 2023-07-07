import user_env_gym.double_pendulum as dpend
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
env = dpend.DoublePendEnv(reward_mode=4)

# initialize agent, set epoch length
load = bool(input("Load Latest Agent? (True or False): "))
agent = SACAgent(env=env, load=load)

if load:
    n_showcase = int(input("n_showcase: "))
    showcase(agent, env=dpend.DoublePendEnv(reward_mode=4, render_mode='human'), n_showcase=n_showcase)

# train the agent if not load
if not load:
    epochs = int(input("EPOCHS: "))
    save = bool(input("Save Agent? (True or False): "))
    scores = agent.train(epochs, save=save)
    print("------Training Completed------")

    # turn off plotting interactive mode
    plt.ioff()
    plt.plot(agent.save_epi_reward)

    # plot moving average 
    final_plot(scores)

    #see how the trained agent performs
    showcase(agent, env=dpend.DoublePendEnv(reward_mode=4, render_mode='human'), n_showcase=5)