# Implementation: DDPG

My implementation code of DDPG algorithm
[Continous Control with Deep Reinforcement Learning, 2016 ICLR](https://arxiv.org/pdf/1509.02971.pdf)


# Overview

- DDPG (Deep Deterministic Policy Gradient) is an algorithm that follows the Actor-Critic style, where the Actor aims to optimize continuous control using the deep deterministic policy gradient algorithm, and the Critic is updated in a similar fashion to the DQN (Deep Q-Network) style.
- $$\bigtriangledown_{\phi} J \approx  \frac{1}{N}\sum\bigtriangledown_{\phi}Q_{\theta}(s, \pi_{\phi}(s))$$

