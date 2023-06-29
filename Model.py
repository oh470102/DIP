import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size//2)
        self.l4 = nn.Linear(hidden_size//2, output_size)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        out = self.l4(x)

        return out.to(device)

class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size//2)
        self.l4 = nn.Linear(hidden_size//2, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        out = 20 * self.tanh(self.l4(x))

        return out.to(device)