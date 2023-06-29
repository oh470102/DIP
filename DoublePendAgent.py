import user_env_gym.double_pendulum as dpend
from utils import *
from tqdm import tqdm
from collections import deque
from Model import *
import torch
import copy
import matplotlib.pyplot as plt

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

class DDPGAgent:
    
    def __init__(self, hidden_size, output_size):
        self.epochs = 900#int(input("enter epochs: "))
        self.alpha_actor = 1e-4
        self.alpha_critic = 1e-3
        self.gamma = 0.99
        self.tau = 5e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(4, hidden_size, output_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.load_state_dict(self.actor_target.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        self.actor_loss_fn = None
        self.critic = Critic(5, hidden_size, output_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.load_state_dict(self.critic_target.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.alpha_critic)
        self.critic_loss_fn = torch.nn.MSELoss()

        self.replay_buffer = ExperienceReplay(max_length=100000, batch_size=64)

        self.best_models = dict()

    # Simulation with learned policy (includes graphic)
    def test_agent(self, model):
        env = dpend.DoublePendEnv(render_mode='human', reward_mode=0)
        n_episodes = 1

        for _ in range(n_episodes):
            curr_state, _ = env.reset()

            terminated = False
            truncated = False

            while not terminated and not truncated:
                curr_state = torch.from_numpy(curr_state).to(self.device)
                action = model(curr_state)
                next_state, reward, truncated, terminated, _ = env.step(action)
                
                curr_state = next_state
                
        env.close()  

    # Training the agent to learn the policy    
    def train_agent(self):
        env = dpend.DoublePendEnv(reward_mode=0)
        noise = OUNoise(env.action_space)
        scores = [0]

        for i in tqdm(range(self.epochs)):

            curr_state, _ = env.reset()
            noise.reset()
            truncated = False
            terminated = False
            j = 0

            if i %(self.epochs//30) == 0 or i == self.epochs-1:
                print(sum(scores[-self.epochs//30:])/len(scores[-self.epochs//30:]))
                # plt.clf
                # plt.plot(scores)
                # plt.draw()
                # plt.pause(0.001)

            while not truncated and not terminated:
                curr_state = torch.from_numpy(curr_state).to(self.device)
                action = self.actor(curr_state)
                action = noise.process_action(action, j)
                next_state, reward, truncated, terminated, _ = env.step(action)
                
                self.replay_buffer.append(curr_state, action, reward, torch.from_numpy(next_state), truncated or terminated)

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
                    state_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay_buffer.sample()

                    # Calculate Critic Loss 
                    Q = self.critic.forward(state_batch, action_batch)
                    action2_batch = self.actor_target.forward(state2_batch)
                    
                    with torch.no_grad():
                        Q_next = self.critic_target(state2_batch, action2_batch.detach())
                    
                    Q_target = reward_batch + self.gamma * (1-done_batch)*Q_next
                    critic_loss = self.critic_loss_fn(Q, Q_target)

                    # Calculate Actor Loss
                    actor_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()

                    # update networks
                    # NOTE: 반드시 actor를 먼저 update 
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # update target networks
                    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                curr_state = next_state
                j += 1

                if truncated or terminated: 
                    if j > max(scores): 
                        self.best_models[j] = copy.deepcopy(self.actor).state_dict()
                    scores.append(j)
                    break

        best_score = max(self.best_models.keys())
        best_model_params = self.best_models[best_score]
        best_model_copy = Actor(4, 128, 1).to(self.device)
        best_model_copy.load_state_dict(best_model_params)

        return scores, best_model_copy


ddpg_agent = DDPGAgent(hidden_size=128, output_size=1)
scores, best_model = ddpg_agent.train_agent()

plt.ioff()
plt.show()

ddpg_agent.test_agent(best_model)
print(scores)
