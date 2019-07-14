import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from agents.buffer import *
from agents.prioritized_buffer import *


class Actor(nn.Module):
  def __init__(self, state_size, action_size, action_min, action_max):
    super(Actor, self).__init__()
    self.action_min = float(action_min)
    self.action_max = float(action_max)

    self.fc_1 = nn.Linear(state_size, 64)
    self.ln_1 = nn.LayerNorm(64)  # layer normalization (https://arxiv.org/abs/1607.06450)
    self.fc_2 = nn.Linear(64, 64)
    self.ln_2 = nn.LayerNorm(64)
    self.fc_out = nn.Linear(64, action_size)

    init.xavier_uniform_(self.fc_1.weight, gain=np.sqrt(3))
    init.xavier_uniform_(self.fc_2.weight, gain=np.sqrt(3))
    init.xavier_uniform_(self.fc_out.weight, gain=np.sqrt(3))

  def forward(self, x):
    out = F.relu(self.ln_1(self.fc_1(x)))
    out = F.relu(self.ln_2(self.fc_2(out)))
    out = torch.sigmoid(self.fc_out(out))
    return out * (self.action_max - self.action_min) + self.action_min


class Critic(nn.Module):
  def __init__(self, state_size, action_size):
    super(Critic, self).__init__()
    self.fc_state = nn.Linear(state_size, 64)
    self.ln_state = nn.LayerNorm(64)
    self.fc1 = nn.Linear(64 + action_size, 64)
    self.ln_1 = nn.LayerNorm(64)
    self.fc2 = nn.Linear(64, 64)
    self.ln_2 = nn.LayerNorm(64)
    self.fc_value = nn.Linear(64, 1, bias=False)

    init.xavier_uniform_(self.fc_state.weight, gain=np.sqrt(3))
    init.xavier_uniform_(self.fc1.weight, gain=np.sqrt(3))
    init.xavier_uniform_(self.fc2.weight, gain=np.sqrt(3))
    init.xavier_uniform_(self.fc_value.weight, gain=np.sqrt(3))

  def forward(self, state, action):
    out = F.relu(self.ln_state(self.fc_state(state)))
    out = torch.cat([out, action], dim=1)
    out = F.relu(self.ln_1(self.fc1(out)))
    out = F.relu(self.ln_2(self.fc2(out)))
    out = self.fc_value(out)
    return out


class Agent:
  def __init__(self,
               state_size,
               action_size,
               max_epsiodes,
               num_workers=1,  # num workers for parallel training
               action_min=0.1, action_max=1.0,
               lr=1e-3, gamma=0.9, tau=1e-2,
               epsilon_start=1.0, epsilon_end=0.01,
               buffer_size=10000, min_buffer_size=10000, batch_size=64,
               cuda=False):

    self.state_size = state_size
    self.action_size = action_size
    self.episode_now = 0
    self.max_epsiodes = max_epsiodes
    self.num_workers = num_workers
    self.lr = lr
    self.gamma = gamma
    self.tau = tau
    self.buffer_size = buffer_size
    self.min_buffer_size = min_buffer_size
    self.batch_size = batch_size
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_now = epsilon_start

    # the decay factor of epsilon after each epsiode
    self.epsilon_rate = (epsilon_end / epsilon_start) ** (1 / max_epsiodes)
    self.cuda = cuda

    self.actor = Actor(state_size, action_size, action_min, action_max)
    self.actor_target = Actor(state_size, action_size, action_min, action_max)

    # pertubed actors for exploring, each worker has its own actor_pertub
    self.actor_pertub = \
      [Actor(state_size, action_size, action_min, action_max) for _ in range(num_workers)]

    self.critic = Critic(state_size, action_size)
    self.critic_target = Critic(state_size, action_size)

    self.actor_target.load_state_dict(self.actor.state_dict())
    self.critic_target.load_state_dict(self.critic.state_dict())
    for actor_p in self.actor_pertub:
      actor_p.load_state_dict(self.actor.state_dict())

    if self.cuda:
      self.actor.cuda()
      for actor_p in self.actor_pertub:
        actor_p.cuda()
      self.actor_target.cuda()
      self.critic.cuda()
      self.critic_target.cuda()

    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, amsgrad=True)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, amsgrad=True)

    self.buffer = Memory(batch_size=batch_size, memory_size=buffer_size)

    self.rewards = []
    self.moving_rewards = None

  # exploration before buffer size is large enough
  def is_explore(self):
    return len(self.buffer.memory) < self.min_buffer_size

  # get pertubed action from a worker
  def get_action(self, state, w_id=0):
    if self.cuda:
      state = state.cuda()
    return self.actor_pertub[w_id](state).detach().cpu().numpy()

  def record(self, item):
    self.buffer.append(item)

  # at the beginning of an epsiode
  # 1. decay the epsilon with epsilon_rate (all workers share one epsilon)
  # 2. generate the new pertubed actor according to the updated epsilon
  #    for the specified worker
  def episode_start(self, w_id=0):
    if not self.is_explore():
      self.epsilon_now = self.epsilon_now * self.epsilon_rate

    for (t_name, t_param), (name, param) in \
        zip(self.actor_pertub[w_id].named_parameters(), self.actor.named_parameters()):
      assert t_name == name
      if 'ln' in name:
        t_param.data.copy_(param.data)
      else:
        t_param.data.copy_(param.data + torch.randn(*param.data.shape) * self.epsilon_now)

  # at the end of an epsiode
  # 1. update the episode count
  # 2. update the moving average of reward
  def episode_end(self, reward):
    self.episode_now += 1

    self.rewards.append(reward)
    if len(self.rewards) < 100:
      self.moving_rewards = np.mean(self.rewards)
    else:
      self.moving_rewards = 0.95 * self.moving_rewards + 0.05 * reward

  def train(self):
    if self.is_explore():
      # do not update agent during exploration, only recalculate the epsilon_rate
      self.epsilon_rate = (self.epsilon_end / self.epsilon_start) ** \
                          (1 / (self.max_epsiodes - self.episode_now))
      return 0, 0, self.epsilon_now

    # split experience into states，actions, rewards, next_states and dones
    states, actions, rewards, next_states, dones = \
      zip(*self.buffer.sample_batch())
    states = torch.tensor(states).float()
    actions = torch.tensor(actions).float()[:, None]
    rewards = torch.tensor(rewards).float()[:, None]
    if self.moving_rewards is not None:
      rewards = rewards - self.moving_rewards
    next_states = torch.tensor(next_states).float()
    dones = torch.tensor(dones).float()[:, None]

    if self.cuda:
      states = states.cuda()
      actions = actions.cuda()
      rewards = rewards.cuda()
      next_states = next_states.cuda()
      dones = dones.cuda()

    # the ddpg algorithm
    next_actions = self.actor_target(next_states)
    returns = rewards + self.gamma * self.critic_target(next_states, next_actions).detach() * (1 - dones)

    self.actor.train()
    self.critic.train()

    # critic update
    q_values = self.critic(states, actions)
    # td_errors = torch.abs(q_values - returns).detach().cpu().numpy()
    critic_loss = torch.mean(((q_values - returns) ** 2))
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # actor update
    new_actions = self.actor(states)
    actor_loss = -torch.mean(self.critic(states, new_actions))
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # update actor_target and critic_target
    for t_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      t_param.data.copy_(t_param.data * (1 - self.tau) + param.data * self.tau)
    for t_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      t_param.data.copy_(t_param.data * (1 - self.tau) + param.data * self.tau)

    return critic_loss.item(), actor_loss.item(), self.epsilon_now
