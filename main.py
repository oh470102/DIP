import gym
import numpy as np
from agents import SACAgent
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

### HYPERPARAMS
gamma = 0.99
alpha = 3e-4
beta = 3e-4
fc1_dim = 256
fc2_dim = 256
memory_size = 1_000_000
batch_size = 256
tau = 5e-3
updated_period = 2
reward_scale = 2
warmup = 100
reparam_noise_lim = 1e-6
epochs = int(input("EPOCHS:"))
showcase = True
save_model = False

###
plt.ion()
env = gym.make('InvertedDoublePendulum-v4')
agent = SACAgent(gamma=gamma, alpha=alpha, beta=beta, state_dims=env.observation_space.shape, 
              action_dims=env.action_space.shape, max_action=env.action_space.high[0],
              fc1_dim=fc1_dim, fc2_dim=fc2_dim, memory_size=memory_size, batch_size=batch_size,
              tau=tau, update_period=updated_period, reward_scale=reward_scale, warmup=warmup,
              reparam_noise_lim=reparam_noise_lim)

scores = [40]

for i in tqdm(range(epochs)):
    state, _ = env.reset()
    terminated, truncated = False, False
    score = 0

    if i % 50 == 0: live_plot(scores)

    while not truncated and not terminated:
        action = agent.choose_action(state, deterministic=False, reparameterize=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        score += reward
        agent.store_transition(state, action, reward, next_state, truncated or terminated)
        agent.learn()
        state = next_state
    
    scores.append(score)

resolve_matplotlib_error()
final_plot(scores)

if showcase: 
    env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
    for i in range(3):
        state, _ = env.reset()
        terminated, truncated = False, False

        while not truncated and not terminated:
            action = agent.choose_action(state, deterministic=True, reparameterize=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

if save_model: agent.save()
