import user_env_gym.double_pendulum as dpend

import torch
from collections import deque

# Testcode to check operation correctness of simulator

env = dpend.DoublePendEnv(render_mode="human")
# env = dpend.DoublePendEnv()

n_episodes = 1
for episode in range(n_episodes):
    state, _ = env.reset()

    terminated = False
    truncated = False

    for i in range(1000):
        state = torch.tensor(state, dtype=torch.float32)
        
        action = 0.0
        state = env.step(action)
        print(state)
        
        
env.close() 

