import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim=4):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.hidden_act = nn.ReLU()

        nn.init.kaiming_uniform_(self.layer1.weight)
        nn.init.kaiming_uniform_(self.layer2.weight)
        nn.init.kaiming_uniform_(self.layer3.weight)

    def forward(self, s):
        s = self.layer1(s)
        s = self.hidden_act(s)
        s = self.layer2(s)
        s = self.hidden_act(s)
        s = self.layer3(s)
        s = torch.tensor(2.0) * torch.tanh(s)
        return s

class Qnet(nn.Module):
    def __init__(self, input_dim=4, action_dim=1):
        super(Qnet, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.layer_s = nn.Linear(input_dim, 64)
        self.layer_a = nn.Linear(action_dim, 64)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.hidden_act = nn.ReLU()

    def forward(self, s, a):
        s_vec = self.hidden_act(self.layer_s(s))
        a_vec = self.hidden_act(self.layer_a(a))

        x = torch.cat([s_vec, a_vec], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        return x
