from .DRL.env.drl_backtesting import Strategy
# from backtesting import Strategy
from .DRL.settings import MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL, BATCH_SIZE, LONG, \
                        SHORT, CLOSE, HOLD, LIQUIFIED, DATA_DONE, DEMOCRATISATION, INDICATOR_NUM, \
                            MIN_START_POINT, MAX_START_POINT_FROM, PRINT_STATUS_INTERVAL

import torch
import numpy as np
import pandas as pd
import time
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class DRLStrategy(Strategy):
    def init(self, agent, data_length, interval=10):
        self.agent = agent
        self.is_training = True
        self.interval = int(interval)

        self.reward_sum = 0.0
        self.step = -1
        self.action_history = [0, 0, 0]
        self.current_balance = 0
        self.previous_balance = 0
        self.state = None
        self.previous_state = None
        self.action = None
        self.previous_action = [HOLD] * self.agent.state_size
        self.is_liquified = False
        self.is_data_done = False
        self.episode_start = time.perf_counter()

        self.start_point = random.randint(MIN_START_POINT, data_length - MAX_START_POINT_FROM)
        self.is_episode_started = False

    def resample_and_align(self, freq, reference_index):
        d = self.data.df.copy()

        d['datetime'] = pd.to_datetime(d['datetime'])
        d.set_index('datetime', drop=True, inplace=True)

        resampled_data = d.resample(freq).agg({
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'
        }).dropna()

        aligned_data = resampled_data[resampled_data.index <= self.data.datetime[reference_index]].tail(self.agent.state_size)
        return aligned_data

    def get_state(self):
        one_minute_data = np.array([self.data.Open[-self.agent.state_size:],
                                    self.data.Close[-self.agent.state_size:],
                                    self.data.High[-self.agent.state_size:],
                                    self.data.Low[-self.agent.state_size:],
                                    self.data.Volume[-self.agent.state_size:]])

        reference_index = -1

        frequencies = ['10min', '30min', '1h', '4h']
        states = []

        for freq in frequencies:
            resampled_data = self.resample_and_align(freq, reference_index)
            states.append(np.array([resampled_data.Open,
                                    resampled_data.Close,
                                    resampled_data.High,
                                    resampled_data.Low,
                                    resampled_data.Volume]))

        result = np.concatenate([one_minute_data] + states + [np.array(self.previous_action).reshape(1, 300)], axis=0)
        return result


    def reward(self, current_action, is_liq):
        rw = 0

        if current_action == self.previous_action[-1]:
            rw -= 1

        rw += (self.current_balance - self.previous_balance)# * 10

        if is_liq:
            print("Liquifided!!")
            rw += DEMOCRATISATION

        return rw


    def take_step(self, action):
        current_action = np.argmax(action)
        state = self.get_state()

        if not (self.is_liquified or self.is_data_done):
            if self.previous_action[-1] == current_action:
                current_action = HOLD
            elif current_action == LONG:
                self.buy()
            elif current_action == SHORT:
                self.sell()

        # if self.equity <= 3:
        #     self.is_liquified = True

        rw = self.reward(current_action, self.is_liquified)

        return state, rw, current_action, (self.is_data_done or self.is_liquified)


    def next(self):
        if len(self.data) < self.start_point:
            return
        self.step += 1
        if self.step % self.interval != 0:
            return
        if not self.is_episode_started:
            print(f'Episode starting from {self.data.datetime[-1]}, Initial balance {self.equity}')
            self.previous_state = torch.tensor(self.get_state()).detach().cpu().to(dtype=torch.float)
            self.is_episode_started = True

        self.previous_balance = self.current_balance
        self.current_balance = self.equity

        # To train faster..
        if self.step < BATCH_SIZE:
            self.action = self.agent.model.get_action_random()
        else:
            self.action = self.agent.model.get_action(self.previous_state, self.step, False)
        self.state, reward, current_action, is_episode_done = self.take_step(self.action)
        # print(f'Current date time {self.data.datetime[-1]}, balance {self.current_balance}, Action {a[current_action]}')

        del self.previous_action[0]
        self.previous_action.append(np.argmax(self.action))
        self.state = torch.tensor(self.state).detach().cpu().to(dtype=torch.float)

        # Training Agent
        if self.is_training:
            self.train_model(reward, is_episode_done)

        self.reward_sum += reward
        self.action_history[current_action] += 1
        self.previous_state = self.state

        if self.step % (PRINT_STATUS_INTERVAL * self.interval) == 0:
            print(f'Current date time {self.data.datetime[-1]}, Current balance {self.current_balance}')


        if is_episode_done:
            self.finish_episode()

    def train_model(self, reward, is_episode_done):
        self.agent.replay_buffer.add_sample(self.previous_state, self.action, [reward], self.state, [1] if is_episode_done else [0])
        if self.agent.replay_buffer.get_length() >= self.agent.model.batch_size:
            loss_c, loss_a, = self.agent.model._train(self.agent.replay_buffer)
            self.agent.loss_critic += loss_c
            self.agent.loss_actor += loss_a

    def finish_episode(self):
        self.agent.episode += 1
        duration = time.perf_counter() - self.episode_start
        final_balance = self.equity
        self.agent.total_step += self.step

        print(f"ep: {self.agent.episode} balance: {final_balance} reward: {self.reward_sum:<6.2f}")
        print(f"buy: {self.action_history[LONG]} sell: {self.action_history[SHORT]} hold: {self.action_history[HOLD]}")
        print(f"steps: {self.step:<6} steps_total: {self.agent.total_step:<7}time: {duration:<6.2f}\n")

        if (not self.is_training):
            self.agent.logger.update_test_results(self.step, final_balance if self.is_liquified else "Liquifieded", duration)
            return

        self.agent.graph.update_data(self.step, self.agent.total_step, final_balance, self.reward_sum, self.agent.loss_critic, self.agent.loss_actor)
        self.agent.logger.file_log.write(f"{self.agent.episode}, {self.reward_sum}, {final_balance if final_balance > 0 else 'Liquifieded'}, {duration}, {self.step}, {self.agent.total_step}, \
                                {self.agent.replay_buffer.get_length()}, {self.agent.loss_critic / self.step}, {self.agent.loss_actor / self.step}\n")


        if (self.agent.episode % MODEL_STORE_INTERVAL == 0) or (self.agent.episode == 1):
            self.agent.sm.save_session(self.agent.episode, self.agent.model.networks, self.agent.graph.graphdata, self.agent.replay_buffer.buffer)
            # logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
        if (self.agent.episode % GRAPH_DRAW_INTERVAL == 0) or (self.agent.episode == 1):
            self.agent.graph.draw_plots(self.agent.episode)

        self.episode_start = time.perf_counter()

