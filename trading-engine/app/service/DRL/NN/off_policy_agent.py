#!/usr/bin/env python3

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as torchf
from core_optimizer import CoRe

from ..settings import N_TRAIN, ACTION_SIZE, HIDDEN_SIZE, BATCH_SIZE, BUFFER_SIZE

class OffPolicyAgent(ABC):
    def __init__(self, device):

        self.device = device

        # Network structure
        self.state_size         = N_TRAIN # Could add more states on environment.py(also environment_real.py)
        self.action_size        = ACTION_SIZE
        self.hidden_size        = HIDDEN_SIZE
        self.input_size         = self.state_size
        # Hyperparameters
        self.batch_size         = BATCH_SIZE
        self.buffer_size        = BUFFER_SIZE
        self.discount_factor    = 0.99
        self.learning_rate      = 0.001
        self.tau                = 0.003
        # Other parameters
        self.step_time          = 0.001
        self.loss_function      = torchf.smooth_l1_loss
        self.epsilon            = 1.0
        self.epsilon_decay      = 0.9995
        self.epsilon_minimum    = 0.05
        self.reward_function    = 'A'
        self.stacking_enabled   = False
        self.stack_depth        = 3
        self.frame_skip         = 4
        # if ENABLE_STACKING:
        #     self.input_size *= self.stack_depth

        self.networks = []
        self.iteration = 0

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def get_action():
        pass

    @abstractmethod
    def get_action_random():
        pass

    def _train(self, replaybuffer):
        batch = replaybuffer.sample(self.batch_size)
        sample_s, sample_a, sample_r, sample_ns, sample_d = batch
        sample_s = torch.from_numpy(sample_s).to(self.device)
        sample_a = torch.from_numpy(sample_a).to(self.device)
        sample_r = torch.from_numpy(sample_r).to(self.device)
        sample_ns = torch.from_numpy(sample_ns).to(self.device)
        sample_d = torch.from_numpy(sample_d).to(self.device)
        result = self.train(sample_s, sample_a, sample_r, sample_ns, sample_d)
        self.iteration += 1
        if self.epsilon and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        return result

    def create_network(self, type, name):
        network = type(name, self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.networks.append(network)
        return network

    def create_optimizer(self, network):
        # return torch.optim.SGD(network.parameters(), self.learning_rate)
        # return torch.optim.RMSprop(network.parameters(), self.learning_rate)
        return torch.optim.AdamW(network.parameters(), self.learning_rate)
        # return CoRe(network.parameters(), self.learning_rate)

    def create_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.00001)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_model_configuration(self):
        configuration = ""
        for attribute, value in self.__dict__.items():
            if attribute not in ['actor', 'actor_target', 'critic', 'critic_target']:
                configuration += f"{attribute} = {value}\n"
        return configuration

    def get_model_parameters(self):
        parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size,
                            self.discount_factor, self.learning_rate, self.tau, self.step_time, 'A',
                            True, False, self.stack_depth, self.frame_skip]
        parameter_string = ', '.join(map(str, parameters))
        return parameter_string

    def attach_visual(self, visual):
        self.actor.visual = visual

class Network(torch.nn.Module, ABC):
    def __init__(self, name, visual=None):
        super(Network, self).__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0

    @abstractmethod
    def forward():
        pass

    def init_weights(n, m):
        if isinstance(m, torch.nn.Linear):
            # --- define weights initialization here (optional) ---
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)