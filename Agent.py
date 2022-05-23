import torch
import torch.nn as nn

class DDPGagent(nn.Module):
    def __init__(self,
                 qnet:nn.Module,
                 qnet_target:nn.Module,
                 actor:nn.Module,
                 actor_target:nn.Module,
                 tau:float, lr:float,
                 discount_factor:float):

        super().__init__()
        self.qnet = qnet
        self.actor = actor
        self.qnet_target = qnet_target
        self.actor_target = actor_target
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.tau = tau
        self.discount_factor = discount_factor
        self.lr = lr

        self.huber = nn.HuberLoss()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.qnet_opt = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        with torch.no_grad():
            action = self.actor(state)[0]
            action = action.numpy()
        return action

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        with torch.no_grad():
            next_action = self.actor_target(ns)
            next_value = self.qnet_target(ns, next_action)
            target = r + self.discount_factor * next_value * (1-done)

        value = self.qnet(s, a)
        self.qnet_loss = self.huber(value, target)
        self.qnet_opt.zero_grad()
        self.qnet_loss.backward()
        self.qnet_opt.step()

        self.actor_loss = -self.qnet(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        self.actor_loss.backward()
        self.actor_opt.step()


    def soft_target_update(self, params, target_params):
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)