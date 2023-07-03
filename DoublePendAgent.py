import user_env_gym.double_pendulum as dpend
from utils import *
from tqdm import tqdm
from collections import deque
from Model import *
import torch
import copy
import matplotlib.pyplot as plt
from datetime import datetime

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

class DDPGAgent:
    
    def __init__(self, hidden_size, output_size):
        self.epochs = int(input("enter epochs: "))
        self.alpha_actor = 1e-4
        self.alpha_critic = 1e-3
        self.gamma = 0.99
        self.tau = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(4, hidden_size, output_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.load_state_dict(self.actor_target.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha_actor)
        self.actor_loss_fn = None
        self.critic = Critic(5, hidden_size, output_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.load_state_dict(self.critic_target.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.alpha_critic, weight_decay=0.2)
        self.critic_loss_fn = torch.nn.MSELoss()

        self.replay_buffer = ExperienceReplay(max_length=100000, batch_size=64)
        self.learning_starts = 150

        self.best_models = dict()

    # Simulation with learned policy (includes graphic)
    def test_agent(self, model):
        env = dpend.DoublePendEnv(render_mode='human', reward_mode=3)
        n_episodes = 5

        for _ in range(n_episodes):
            curr_state, _ = env.reset()

            terminated = False
            truncated = False

            score = 0 

            while not terminated and not truncated:
                self.actor.eval()
                curr_state = torch.from_numpy(curr_state).unsqueeze(0).to(self.device)
                action = model(curr_state).cpu().detach()
                next_state, reward, truncated, terminated, _ = env.step(action)

                print(reward)

                score += reward
                
                curr_state = next_state
                
        print(f"MODEL SCORE:{score}")
        env.close()  

    # Training the agent to learn the policy    
    def train_agent(self):
        env = dpend.DoublePendEnv(reward_mode=3)
        noise = OUNoise(env.action_space)
        scores = [40] # default score

        for i in tqdm(range(self.epochs)):

            curr_state, _ = env.reset()
            noise.reset()
            truncated = False
            terminated = False
            j = 0

            if i >= self.epochs//30 and i %(self.epochs//30) == 0 or i == self.epochs-1:
                print(f"average of last {self.epochs//30} scores: {sum(scores[-self.epochs//30:])/len(scores[-self.epochs//30:]): .2f}")
                live_plot(scores)

            while not truncated and not terminated:

                self.actor.eval()
                with torch.no_grad():
                    curr_state = torch.from_numpy(curr_state).unsqueeze(0).to(self.device)
                    action = self.actor(curr_state)
                    action = noise.process_action(action, j)
                self.actor.train()

                next_state, reward, truncated, terminated, info = env.step(action.cpu())
                
                self.replay_buffer.append(curr_state, action, reward, torch.from_numpy(next_state), truncated or terminated)

                if i > self.learning_starts:
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
                    # NOTE: need to update actor first.
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
        best_model_copy = Actor(4, 32, 1).to(self.device)
        best_model_copy.load_state_dict(best_model_params)

        return scores, best_model_copy


ddpg_agent = DDPGAgent(hidden_size=32, output_size=1)

scores, best_model = ddpg_agent.train_agent()

plt.ioff()
plt.show()

file_path = datetime.now().strftime("%-I:%M") +".pth"
torch.save(best_model.state_dict(), file_path)

ddpg_agent.test_agent(best_model)

# ddpg_agent.actor.load_state_dict(torch.load("2:51.pth"))
# ddpg_agent.test_agent(model=ddpg_agent.actor)