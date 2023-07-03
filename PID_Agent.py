import user_env_gym.double_pendulum as dpend

import torch
from collections import deque
import numpy as np

# Testcode to check operation correctness of simulator

env = dpend.DoublePendEnv(render_mode="human", reward_mode=3)
# env = dpend.DoublePendEnv()

n_episodes = 1
ref_th1 = np.pi
ref_th2 = np.pi
err_th1 = 0
err_th2 = 0
mv_1 = 0
mv_2 = 0

preverr_th1 = 0
dpreverr_th1 = 0
preverr_th2 = 0
dpreverr_th2 = 0

K_P = [600, 1000]
K_I = [1,0.5]
K_D = [100,300]

for episode in range(n_episodes):
    state, _ = env.reset()

    terminated = False
    truncated = False

    for i in range(1000):
        dpreverr_th1 = preverr_th1
        preverr_th1 = err_th1
        dpreverr_th2 = preverr_th2
        preverr_th2 = err_th2

        err_th1 = ref_th1 - state[0]
        err_th2 = ref_th2 - state[2]

        mv_1 = mv_1 + K_I[0] * err_th1 + K_P[0] * (err_th1 - preverr_th1) + K_D[0] * (err_th1 - 2 * preverr_th1 + dpreverr_th1)
        mv_2 = mv_2 + K_I[1] * err_th2 + K_P[1] * (err_th2 - preverr_th2) + K_D[1] * (err_th2 - 2 * preverr_th2 + dpreverr_th2)


        state = torch.tensor(state, dtype=torch.float32)
        
        action = mv_1 - mv_2

        state, reward, terminated, truncated, info = env.step(action)

        print(reward)

        if terminated or truncated:
            break
        
        
env.close() 

