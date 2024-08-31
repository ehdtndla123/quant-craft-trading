from .DRL.env.drl_backtesting import Strategy
# from backtesting import Strategy
from .DRL.NN.td3_simple import TD3
from .DRL.replaybuffer import ReplayBuffer
from .DRL.storagemanager import StorageManager
from .DRL.graph import Graph
from .DRL.settings import MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL, N_TRAIN, LONG, \
                        SHORT, CLOSE, HOLD, LIQUIFIED, DATA_DONE, DEMOCRATISATION

import torch
import numpy as np
# import pandas as pd
import time


class DRLStrategy(Strategy):
    def init(self):
        self.is_training = True
        self.device = self.get_device()
        self.init_balance = 1000 #USDT
        self.state_size = N_TRAIN

        self.is_data_done = False
        self.is_liquified = False

        self.previous_state = None
        self.previous_action = HOLD
        self.current_balance = 0
        self.previous_balance = 0

        # Create DRL Agent
        self.agent = TD3(self.device)

        # Create Storage Manager
        self.sm = StorageManager('td3', '', 0, self.device)
        self.sm.new_session_dir()
        self.sm.store_model(self.agent)

        # Create Graph Plotter
        self.graph = Graph()
        self.graph.session_dir = self.sm.session_dir

        # Create Replay Buffer
        self.replay_buffer = ReplayBuffer(self.agent.buffer_size)
        self.episode, self.step, self.loss_critic, self.loss_actor, self.total_step = 0, 0, 0, 0, 0
        self.reward_sum = 0.0
        self.action_history = [0, 0, 0, 0]
        self.episode_start = time.perf_counter()

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("MPS is available. Using GPU on macOS.")
        else:
            device = torch.device('cpu')
            print("No GPU found. Using CPU.")

        return device

    # def get_state(arr: pd.Series, n: int) -> pd.Series:
    #     return pd.Series(arr)[n:]
    def get_state(self):
        return np.append(np.array([self.data.Open[:self.state_size], self.data.Close[:self.state_size], self.data.High[:self.state_size],
                    self.data.Low[:self.state_size], self.data.Volume[:self.state_size]]).flatten(), self.previous_action)

    def reward(self, current_action, is_liq):
        rw = 0

        if current_action == LONG:
            if self.previous_action == current_action:
                rw -= 50
            else:
                rw += 10
        elif current_action == SHORT:
            if self.previous_action == current_action:
                rw -= 50
            else:
                rw += 10
        elif current_action == CLOSE or current_action == HOLD:
            rw -= 10

        rw += (self.current_balance - self.previous_balance) * 10

        if is_liq:
            print("Liquifided!!")
            rw += DEMOCRATISATION

        return rw


    def take_step(self, action):
        current_action = np.argmax(action)

        state = self.get_state()

        if current_action == LONG:
            self.buy()
            self.action_history[LONG] += 1
        elif current_action == SHORT:
            self.sell()
            self.action_history[SHORT] += 1
        elif current_action == CLOSE:
            self.position.close()
            self.action_history[CLOSE] += 1
        else:
            self.action_history[HOLD] += 1

        # if self.equity == 0:
        #     self.is_liquified = True

        rw = self.reward(current_action, self.is_liquified)

        return state, rw, current_action, (self.is_data_done or self.is_liquified)


    def next(self):
        if len(self.data) < N_TRAIN + 1:
            return
        # if self.previous_state is None:
        #     # state = self.I(self.get_state, self.price_state, self.agent.state_size)
        #     state = torch.tensor(self.get_state()).flatten().unsqueeze(0).detach().cpu().to(dtype=torch.float)
        #     self.current_balance = self.equity

        state = torch.tensor(self.get_state()).flatten().detach().cpu().to(dtype=torch.float)
        self.previous_balance = self.current_balance
        self.current_balance = self.equity
        action = self.agent.get_action(state, self.step, False)
        state, reward, current_action, is_episode_done = self.take_step(action)
        self.previous_action = current_action
        state = torch.tensor(state).flatten().detach().cpu().to(dtype=torch.float)
        self.reward_sum += reward
        self.action_history[current_action] += 1
        self.previous_state = state

        # Training Agent
        if self.is_training:
            self.replay_buffer.add_sample(self.previous_state, action, [reward], state, [1] if is_episode_done else [0])
            if self.replay_buffer.get_length() >= self.agent.batch_size:
                loss_c, loss_a, = self.agent._train(self.replay_buffer)
                self.loss_critic += loss_c
                self.loss_actor += loss_a

        self.step += 1
        # print(self.step, ', ', action, ', Liquified: ', self.is_liquified, ', Data Done: ', self.is_data_done)

        if is_episode_done:
            self.episode += 1
            duration = time.perf_counter() - self.episode_start
            final_balance = self.equity
            self.total_step += self.step

            print(f"'ep: {self.episode} balance: {final_balance} reward: {self.reward_sum:<6.2f}")
            print(f"buy: {self.action_history[LONG]} sell: {self.action_history[SHORT]} close:{self.action_history[CLOSE]} hold: {self.action_history[HOLD]}")
            print(f"steps: {self.step:<6} steps_total: {self.total_step:<7}time: {duration:<6.2f}\n")

            if self.is_training:
                self.graph.update_data(self.step, self.total_step, final_balance, self.reward_sum, self.loss_critic, self.loss_actor)

            self.reward_sum = 0.0
            self.step = 0
            self.action_history = [0, 0, 0, 0]
            self.current_balance = 0
            self.previous_balance = 0
            self.previous_state = None
            self.is_liquified = False
            self.is_data_done = False

            if (self.episode % MODEL_STORE_INTERVAL == 0) or (self.episode == 1):
                self.sm.save_session(self.episode, self.agent.networks, self.graph.graphdata, self.replay_buffer.buffer)
                # logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
            if (self.episode % GRAPH_DRAW_INTERVAL == 0) or (self.episode == 1):
                self.graph.draw_plots(self.episode)

            self.episode_start = time.perf_counter()

