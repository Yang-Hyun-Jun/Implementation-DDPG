import torch
import gym
import argparse
import numpy as np

from Noise import OUProcess
from Memory import ReplayMemory
from Agent import DDPGagent
from Network import Actor
from Network import Qnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Pendulum-v1")
    parser.add_argument("--num_episode", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=10000)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    s_dim = 3
    a_dim = 1

    memory = ReplayMemory(args.memory_size)
    ou_noise = OUProcess(mu=np.zeros(1))
    actor = Actor(input_dim=s_dim)
    actor_target = Actor(input_dim=s_dim)
    actor_target.load_state_dict(actor.state_dict())

    qnet = Qnet(input_dim=s_dim, action_dim=a_dim)
    qnet_target = Qnet(input_dim=s_dim, action_dim=a_dim)
    qnet_target.load_state_dict(qnet.state_dict())

    agent = DDPGagent(qnet=qnet,
                      qnet_target=qnet_target,
                      actor=actor,
                      actor_target=actor_target,
                      lr=args.lr,
                      discount_factor=args.discount_factor,
                      tau=args.tau)


    def prepare_training_inputs(sampled_exps):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for sampled_exp in sampled_exps:
            states.append(sampled_exp[0])
            actions.append(sampled_exp[1])
            rewards.append(sampled_exp[2])
            next_states.append(sampled_exp[3])
            dones.append(sampled_exp[4])

        states = torch.cat(states, dim=0).float()
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0).float()
        next_states = torch.cat(next_states, dim=0).float()
        dones = torch.cat(dones, dim=0).float()
        return states, actions, rewards, next_states, dones


    def run():
        steps_done = 0
        for epi in range(args.num_episode):
            state = env.reset()
            state = torch.tensor(state).float().view(1,-1)
            cum_r = 0
            while True:
                env.render()
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                noise = ou_noise()[0]
                action = torch.tensor(action+noise).view(1,-1)
                reward = torch.tensor(reward).float().view(1,-1)
                next_state = torch.tensor(next_state).float().view(1,-1)
                done = torch.tensor(done).float().view(1,-1)

                steps_done += 1
                experience = (state, action, reward, next_state, done)
                memory.push(experience)

                cum_r += reward
                state = next_state

                if done:
                    break

                if len(memory) > args.batch_size:
                    sampled_exps = memory.sample(args.batch_size)
                    sampled_exps = prepare_training_inputs(sampled_exps)
                    agent.update(*sampled_exps)
                    agent.soft_target_update(agent.qnet.parameters(), agent.qnet_target.parameters())
                    agent.soft_target_update(agent.actor.parameters(), agent.actor_target.parameters())

                if steps_done % 50 == 0:
                    q = agent.qnet(state, action).detach().numpy()
                    a = action
                    print(f"epi:{epi} | q:{q} | a:{a} | cum_r:{cum_r}" )

    run()



