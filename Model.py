import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.l1 = nn.Linear(input_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        
        self.l2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)

        self.l3 = nn.Linear(300, output_size)

    def forward(self, action, state):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.bn1(self.l1(state_action)))
        q = F.relu(self.bn2(self.l2(q)))

        out = self.l3(q)

        return out.to(device)

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.l1 = nn.Linear(input_size, 400)
        self.bn1 = nn.BatchNorm1d(400)

        self.l2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)

        self.l3 = nn.Linear(300, output_size)


    def forward(self, state):
        a = F.relu(self.bn1(self.l1(state)))
        a = F.relu(self.bn2(self.l2(a)))
        out = 20 * torch.tanh(self.l3(a))

        return out.to(device)


