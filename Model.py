import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, state, action):
        x = torch.cat([self.leakyrelu(state), action], 1)
        x = self.l1(x)
        x = self.layernorm(self.l2(x))
        x = self.layernorm(self.l3(x))
        out = self.l4(x)

        return out.to(device)

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, state):
        x = self.leakyrelu(self.l1(state))
        x = self.layernorm(self.l2(x))
        x = self.layernorm(self.l3(x))
        out = 20 * self.tanh(self.l4(x))

        return out.to(device)