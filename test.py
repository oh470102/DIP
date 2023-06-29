import gym
import numpy as np

env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
env.reset()

done = False

while not done:
    action = np.random.uniform(low=-1, high=1)
    action = np.array(action).reshape(1, )
    _, _, term, trunc, _ = env.step(action)
    done = term or trunc