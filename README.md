# MADDPG

This is an implementation of the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm for multi-agent reinforcement learning scenarios. MADDPG extends the Deep Deterministic Policy Gradient (DDPG) algorithm to handle multiple agents interacting in the environment.

## Overview

The MADDPG algorithm consists of multiple individual agents, each with their own actor and critic networks. The agents learn from their own experiences and collaborate with each other to improve their performance. The algorithm combines elements of centralized training with decentralized execution.

The code provides a `MADDPG` class that encapsulates the MADDPG algorithm. The key methods and functionalities of the `MADDPG` class are as follows:

- `__init__(self, actor_dims, critic_dims, n_agents, n_actions, scenario, alpha, beta, fc1, fc2, gamma, tau, chkpt_dir)`: Initializes the MADDPG algorithm with the specified parameters. It creates individual agents based on the given dimensions and hyperparameters.

- `save_checkpoint(self)`: Saves the model checkpoints of all the agents.

- `load_checkpoint(self)`: Loads the saved model checkpoints of all the agents.

- `choose_action(self, raw_obs)`: Given the raw observations, selects actions for each agent using their respective actor networks.

- `learn(self, memory)`: Performs the learning step of the MADDPG algorithm. It samples experiences from the replay memory and updates the actor and critic networks of each agent using the DDPG update rules.

## Dependencies

The code requires the following dependencies:

- `torch`: PyTorch library for tensor computations.
- `torch.nn.functional`: Module for various activation functions.
- `agent`: Custom `Agent` class that represents an individual agent.

## Usage

To use the MADDPG algorithm, follow these steps:

1. Import the necessary libraries and the `Agent` class.
   
   ```python
   import torch as T
   import torch.nn.functional as F
   from agent import Agent
   ```

2. Create an instance of the `MADDPG` class, providing the necessary parameters.

   ```python
   maddpg = MADDPG(actor_dims, critic_dims, n_agents, n_actions, scenario, alpha, beta, fc1, fc2, gamma, tau, chkpt_dir)
   ```

3. Perform the learning loop by interacting with the environment, collecting experiences, and calling the `learn` method.

   ```python
   memory = ReplayMemory()
   
   for episode in range(num_episodes):
       state = env.reset()
       done = False

       while not done:
           actions = maddpg.choose_action(state)
           next_state, reward, done = env.step(actions)
           memory.add_experience(state, actions, reward, next_state, done)
           state = next_state

           maddpg.learn(memory)
   ```

4. Save or load the model checkpoints using the `save_checkpoint` and `load_checkpoint` methods.

   ```python
   maddpg.save_checkpoint()
   maddpg.load_checkpoint()
   ```

## References

- [Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., & Mordatch, I. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. In Advances in Neural Information Processing Systems (pp. 6379-6390).](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments)
