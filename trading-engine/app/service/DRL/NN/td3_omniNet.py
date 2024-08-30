import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from omninet_pytorch import Omninet

from .ounoise import OUNoise

from .off_policy_agent import OffPolicyAgent, Network

from ..settings import INDICATOR_NUM

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# Origin by: Seunghyeop
# TD3 with LSTM & CNN 1 lstm parallel architecture


class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        # --- define layers here ---
        self.omninet = Omninet(
            dim = state_size,             # model dimension
            depth = 6,                     # depth
            dim_head = 64,                 # dimension per head
            heads = 6,                     # number of heads
            pool_layer_tokens_every = 3,   # key to this paper - every N layers, omni attend to all tokens of all layers
            attn_dropout = 0.1,            # attention dropout
            ff_dropout = 0.1,              # feedforward dropout
            feature_redraw_interval = 1000 # how often to redraw the projection matrix for omni attention net - Performer
        )

        self.linear = nn.Sequential(
            nn.Linear(state_size * INDICATOR_NUM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, int(hidden_size * 2)),
            nn.SiLU(),
            nn.Linear(int(hidden_size * 2), int(hidden_size * 2)),
            nn.SiLU(),
            nn.Linear(int(hidden_size * 2), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2 ** 1),
            nn.SiLU(),
            nn.Linear(hidden_size // 2 ** 1, hidden_size // 2 ** 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2 ** 2, hidden_size // 2 ** 3),
            nn.SiLU(),
            nn.Linear(hidden_size // 2 ** 3, action_size),
            nn.Sigmoid()
            )

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        is_target = False
        if len(states.size()) == 2:
            states = torch.unsqueeze(states, dim=0)
            is_target = True
        
        x = torch.flatten(self.omninet(states), start_dim=1)
        if is_target:
            x = x.squeeze(0)

        action = self.linear(x)

        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.omninet1 = Omninet(
                dim = state_size,              # model dimension
                depth = 3,                     # depth
                dim_head = 32,                 # dimension per head
                heads = 3,                     # number of heads
                pool_layer_tokens_every = 3,   # key to this paper - every N layers, omni attend to all tokens of all layers
                attn_dropout = 0.1,            # attention dropout
                ff_dropout = 0.1,              # feedforward dropout
                feature_redraw_interval = 1000 # how often to redraw the projection matrix for omni attention net - Performer
            )

        self.action_linear1 = nn.Linear(action_size, hidden_size)

        self.linear1 = nn.Sequential(
                nn.Linear(state_size * INDICATOR_NUM + hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size // 2 ** 1),
                nn.SiLU(),
                nn.Linear(hidden_size // 2 ** 1, hidden_size // 2 ** 3),
                nn.SiLU(),
                nn.Linear(hidden_size // 2 ** 3, action_size),
                nn.Sigmoid()
            )
        
        self.omninet2 = Omninet(
                dim = state_size,             # model dimension
                depth = 3,                     # depth
                dim_head = 32,                 # dimension per head
                heads = 3,                     # number of heads
                pool_layer_tokens_every = 3,   # key to this paper - every N layers, omni attend to all tokens of all layers
                attn_dropout = 0.1,            # attention dropout
                ff_dropout = 0.1,              # feedforward dropout
                feature_redraw_interval = 1000 # how often to redraw the projection matrix for omni attention net - Performer
            )

        self.action_linear2 = nn.Linear(action_size, hidden_size)

        self.linear2 = nn.Sequential(
                nn.Linear(state_size * INDICATOR_NUM + hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size // 2 ** 1),
                nn.SiLU(),
                nn.Linear(hidden_size // 2 ** 1, hidden_size // 2 ** 3),
                nn.SiLU(),
                nn.Linear(hidden_size // 2 ** 3, action_size),
                nn.Sigmoid()
            )

        self.apply(super().init_weights)

    def forward(self, states, actions):
        is_target = False
        if len(states.size()) == 2:
            states = torch.unsqueeze(states, dim=0)
            actions = torch.unsqueeze(actions, dim=0)
            is_target = True
            
        a = self.action_linear1(actions)
        x = torch.flatten(self.omninet1(torch.cat((states, a), dim=1)), start_dim=1)
        if is_target:
            x = x.squeeze(0)
        q1 = self.linear1(x)

        a = self.action_linear2(actions)
        x = torch.flatten(self.omninet2(torch.cat((states, a), dim=1)), start_dim=1)
        if is_target:
            x = x.squeeze(0)
        q2 = self.linear2(x)

        return q1, q2

    def Q1_forward(self, states, actions):
        is_target = False
        if len(states.size()) == 2:
            states.unsqueeze(dim=0)
            actions.unsqueeze(dim=0)
            is_target = True

        a = self.action_linear1(actions)
        x = torch.flatten(self.omninet1(torch.cat((states, a), dim=1)), start_dim=1)
        if is_target:
            x = x.squeeze(0)
        q1 = self.linear1(x)

        return q1

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