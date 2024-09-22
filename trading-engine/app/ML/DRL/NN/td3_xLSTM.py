import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from .xLSTM.xLSTM import xLSTM

from .ounoise import OUNoise

from .off_policy_agent import OffPolicyAgent, Network

from ..settings import BATCH_SIZE, INDICATOR_NUM, get_device

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# Origin by: Seunghyeop
# TD3 with LSTM & CNN 1 lstm parallel architecture


class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        x_example = torch.zeros(BATCH_SIZE, INDICATOR_NUM, state_size).to(device=get_device())
        factor = 2
        depth = 4
        layer_num = 10
        layers = ('m'*7 + 's') * layer_num
        # layers = 'ms'
        self.xlstm = xLSTM(layers, x_example, factor=factor, depth=depth)

        self.fc1 = nn.Linear(state_size * INDICATOR_NUM, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

        self.activation = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        is_forward = False
        if len(states.size()) == 2:
            states = states.unsqueeze(0)
            is_forward = True
            cat_dim = 0
        else:
            cat_dim = 1

        xlstm_output= self.xlstm(states)

        x = torch.flatten(xlstm_output, start_dim = cat_dim)
        x = self.activation(self.fc1(x))
        action = torch.tanh(self.fc2(x))

        if is_forward:
            action = action.squeeze(0)

        # -- define layers to visualize until here ---
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.l1 = nn.Linear(state_size * INDICATOR_NUM, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        # Q2
        # --- define layers here ---
        self.l5 = nn.Linear(state_size * INDICATOR_NUM, int(hidden_size / 2))
        self.l6 = nn.Linear(action_size, int(hidden_size / 2))
        self.l7 = nn.Linear(hidden_size, hidden_size)
        self.l8 = nn.Linear(hidden_size, 1)

        self.silu = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, states, actions):
        states = torch.flatten(states, start_dim = len(states.size()) - 2)
        xs = self.silu(self.l1(states))
        xa = self.silu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = self.silu(self.l3(x))
        x1 = self.l4(x)

        xs = self.silu(self.l5(states))
        xa = self.silu(self.l6(actions))
        x = torch.cat((xs, xa), dim=1)
        x = self.silu(self.l7(x))
        x2 = self.l8(x)

        return x1, x2

    def Q1_forward(self, states, actions):
        states = torch.flatten(states, start_dim = len(states.size()) - 2)
        xs = self.silu(self.l1(states))
        xa = self.silu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = self.silu(self.l3(x))
        x1 = self.l4(x)
        return x1



class TD3(OffPolicyAgent):
    def __init__(self, device):
        super().__init__(device)

        # DRL parameters
        self.noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        # TD3 parameters
        self.policy_noise   = 0.2
        self.noise_clip     = 0.5
        self.policy_freq    = 2

        self.last_actor_loss = 0

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)
        # self.actor_lr_scheduler = self.create_lr_scheduler(self.actor_optimizer)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)
        # self.critic_lr_scheduler = self.create_lr_scheduler(self.critic_optimizer)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)



    def get_action(self, state, is_training, step, visualize=False):
        if self.device == torch.device('mps'):
            state = torch.from_numpy(np.asarray(state)).to(device=self.device, dtype=torch.float32)
        else:
            state = torch.from_numpy(np.asarray(state)).to(device=self.device)
        action = self.actor(state, visualize)
        if is_training:
            if self.device == torch.device('mps'):
                noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(device=self.device, dtype=torch.float32)
            else:
                noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(device=self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(0, 1.0), -1.0, 1.0), np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)]


    def train(self, state, action, reward, state_next, done):
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        action_next = (self.actor_target(state_next) + noise).clamp(-1.0, 1.0)
        Q1_next, Q2_next = self.critic_target(state_next, action_next)
        Q_next = torch.min(Q1_next, Q2_next)

        Q_target = reward + (1 - done) * self.discount_factor * Q_next
        Q1, Q2 = self.critic(state, action)

        loss_critic = self.loss_function(Q1, Q_target) + self.loss_function(Q2, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()
        # self.critic_lr_scheduler.step()

        if self.iteration % self.policy_freq == 0:
            # optimize actor
            loss_actor = -1 * self.critic.Q1_forward(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()
            # self.actor_lr_scheduler.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss]