import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

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
        self.linears = []
        for _ in range(INDICATOR_NUM):
            self.linears.append(nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, int(hidden_size * 2)),
                nn.SiLU(),
                nn.Linear(int(hidden_size * 2), int(hidden_size * 2)),
                nn.SiLU(),
                nn.Linear(int(hidden_size * 2), 2 ** 4),
                nn.SiLU()
                )
            )

        # --- conv layers for feature extraction ---
        self.conv_iter = 3
        self.pooling_kernel_size = 5
        inner_channel_size = 2 ** 5
        fc_size = int(state_size / (self.pooling_kernel_size ** self.conv_iter)) * inner_channel_size

        self.conv = nn.Sequential(
            nn.Conv1d(1, inner_channel_size, 4, padding='same', padding_mode='circular'),
            # nn.BatchNorm1d(self.state_size),
            nn.SiLU(),
            nn.MaxPool1d(self.pooling_kernel_size),
            nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular'),
            # nn.BatchNorm1d(self.state_size // 2 ** 1),
            nn.SiLU(),
            nn.MaxPool1d(self.pooling_kernel_size),
            nn.Conv1d(inner_channel_size, inner_channel_size, 4, padding='same', padding_mode='circular'),
            # nn.BatchNorm1d(self.state_size // 2 ** 2),
            nn.Sigmoid(),
            nn.MaxPool1d(self.pooling_kernel_size)
        )
        self.conv_fc = nn.Linear(fc_size, hidden_size // 2 ** 2)

        # --- lstm ---
        self.lstm1 = nn.LSTM(input_size=state_size, hidden_size=hidden_size // 2 ** 2, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=hidden_size // 2 ** 2, hidden_size=hidden_size // 2 ** 2, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=hidden_size // 2 ** 2, hidden_size=hidden_size // 2 ** 2, num_layers=1)

        self.final = nn.Sequential(
            nn.Linear(int((2 ** 4) * INDICATOR_NUM) + hidden_size // 2, hidden_size // 2 ** 1),
            nn.SiLU(),
            nn.LayerNorm(hidden_size // 2 ** 1),
            nn.Linear(hidden_size // 2 ** 1, hidden_size // 2 ** 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_size // 2 ** 2),
            nn.Linear(hidden_size // 2 ** 2, hidden_size // 2 ** 3),
            nn.SiLU(),
            nn.LayerNorm(hidden_size // 2 ** 3),
            nn.Linear(hidden_size // 2 ** 3, action_size),
            nn.Sigmoid()
            )

        self.activation = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        cat_dim = 0 if len(states.size()) == 2 else 1
        xs = []
        for i in range(INDICATOR_NUM):
            if cat_dim == 0:
                xs.append(self.linears[i](states[i]))
            else:
                xs.append(self.linears[i](states[:, i]))

        feature = self.conv(states)
        feature = torch.flatten(feature, start_dim=cat_dim)
        feature = torch.sigmoid(self.conv_fc(feature))

        if cat_dim == 1:
            state = states.transpose(0, 1)
        lx, hid = self.lstm1(state)
        lx = self.silu(lx)
        lx, hid = self.lstm2(lx, hid)
        lx = self.silu(lx)
        lx, _ = self.lstm3(lx, hid)
        if cat_dim == 0:
            lx = torch.sigmoid(lx)[-1, 0, :]
            # lx = torch.sigmoid(lx)[0:, -1, :]
        else:
            lx = torch.sigmoid(lx)[-1, :, :]

        x = torch.cat((xs, feature, lx), dim=cat_dim)

        action = self.final(x)

        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.linears = [[], []]
        self.action_linears = []
        self.final = []

        for i in range(2):
            for _ in range(INDICATOR_NUM):
                self.linears[i].append(nn.Sequential(
                    nn.Linear(state_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, int(hidden_size * 2)),
                    nn.SiLU(),
                    nn.Linear(int(hidden_size * 2), int(hidden_size * 2)),
                    nn.SiLU(),
                    nn.Linear(int(hidden_size * 2), 2 ** 3),
                    nn.SiLU()
                    )
                )

                self.action_linears.append(nn.Linear(action_size, hidden_size // 2 ** 3))

                self.final.append(nn.Sequential(
                    nn.Linear(int((2 ** 4) * INDICATOR_NUM) + hidden_size // 2 ** 3, hidden_size // 2 ** 1),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_size // 2 ** 1),
                    nn.Linear(hidden_size // 2 ** 1, hidden_size // 2 ** 2),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_size // 2 ** 2),
                    nn.Linear(hidden_size // 2 ** 2, hidden_size // 2 ** 3),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_size // 2 ** 3),
                    nn.Linear(hidden_size // 2 ** 3, 1),
                    nn.Tanh()
                    )
                )

        self.activation = nn.SiLU()

        self.apply(super().init_weights)

    def forward(self, states, actions):
        cat_dim = 0 if len(states.size()) == 2 else 1
        xs = [[], []]
        out = []
        for j in range(2):
            for i in range(INDICATOR_NUM):
                if cat_dim == 0:
                    xs[j].append(self.linears[j][i](states[i]))
                else:
                    xs[j].append(self.linears[j][i](states[:, i]))

            xa = self.action_linears[j](actions)
            x = torch.cat((xs, xa), dim=1)
            out.append(self.final(x))

        return out[0], out[1]

    def Q1_forward(self, states, actions):
        cat_dim = 0 if len(states.size()) == 2 else 1
        xs = []
        for i in range(INDICATOR_NUM):
            if cat_dim == 0:
                xs.append(self.linears[0][i](states[i]))
            else:
                xs.append(self.linears[0][i](states[:, i]))

        xa = self.action_linears[0](actions)
        x1 = self.final(torch.cat((xs, xa), dim=1))
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