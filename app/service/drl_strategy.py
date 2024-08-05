from .DRL.env.drl_backtesting import Strategy
# from backtesting import Strategy
from .DRL.settings import MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL, N_TRAIN, LONG, \
                        SHORT, CLOSE, HOLD, LIQUIFIED, DATA_DONE, DEMOCRATISATION

import torch
import numpy as np
# import pandas as pd
import time


class DRLStrategy(Strategy):
    def init(self, agent):
        self.agent = agent
        self.is_training = True

        self.reward_sum = 0.0
        self.step = 0
        self.action_history = [0, 0, 0, 0]
        self.current_balance = 0
        self.previous_balance = 0
        self.previous_state = None
        self.previous_action = HOLD
        self.is_liquified = False
        self.is_data_done = False
        self.episode_start = time.perf_counter()

    def get_state(self):
        return np.append(np.array([self.data.Open[:self.agent.state_size], self.data.Close[:self.agent.state_size], self.data.High[:self.agent.state_size],
                    self.data.Low[:self.agent.state_size], self.data.Volume[:self.agent.state_size]]).flatten(), self.previous_action)

    def reward(self, current_action, is_liq):
        rw = 0

        if current_action == LONG:
            if self.previous_action == current_action:
                rw -= 5
            else:
                rw += 10
        elif current_action == SHORT:
            if self.previous_action == current_action:
                rw -= 5
            else:
                rw += 10
        elif current_action == CLOSE or current_action == HOLD:
            rw -= 1

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

        if self.equity <= 3:
            self.is_liquified = True

        rw = self.reward(current_action, self.is_liquified)

        return state, rw, current_action, (self.is_data_done or self.is_liquified)


    def next(self):
        if len(self.data) < N_TRAIN + 1:
            return
        state = torch.tensor(self.get_state()).flatten().detach().cpu().to(dtype=torch.float)
        self.previous_balance = self.current_balance
        self.current_balance = self.equity
        action = self.agent.model.get_action(state, self.step, False)
        state, reward, current_action, is_episode_done = self.take_step(action)
        self.previous_action = current_action
        state = torch.tensor(state).flatten().detach().cpu().to(dtype=torch.float)
        self.reward_sum += reward
        self.action_history[current_action] += 1
        self.previous_state = state

        # Training Agent
        if self.is_training:
            self.agent.replay_buffer.add_sample(self.previous_state, action, [reward], state, [1] if is_episode_done else [0])
            if self.agent.replay_buffer.get_length() >= self.agent.model.batch_size:
                loss_c, loss_a, = self.agent.model._train(self.agent.replay_buffer)
                self.agent.loss_critic += loss_c
                self.agent.loss_actor += loss_a

        self.step += 1
        # print(self.step, ', ', action, ', Liquified: ', self.is_liquified, ', Data Done: ', self.is_data_done)

        if is_episode_done:
            self.finish_episode()

    def finish_episode(self):
        self.agent.episode += 1
        duration = time.perf_counter() - self.episode_start
        final_balance = self.equity
        self.agent.total_step += self.step

        print(f"'ep: {self.agent.episode} balance: {final_balance} reward: {self.reward_sum:<6.2f}")
        print(f"buy: {self.action_history[LONG]} sell: {self.action_history[SHORT]} close:{self.action_history[CLOSE]} hold: {self.action_history[HOLD]}")
        print(f"steps: {self.step:<6} steps_total: {self.agent.total_step:<7}time: {duration:<6.2f}\n")

        if (not self.is_training):
            self.agent.logger.update_test_results(self.step, final_balance if final_balance > 0 else "Liquifieded", duration)
            return

        self.agent.graph.update_data(self.step, self.agent.total_step, final_balance, self.reward_sum, self.agent.loss_critic, self.agent.loss_actor)
        self.agent.logger.file_log.write(f"{self.agent.episode}, {self.reward_sum}, {final_balance if final_balance > 0 else "Liquifieded"}, {duration}, {self.step}, {self.agent.total_steps}, \
                                {self.agent.replay_buffer.get_length()}, {self.agent.loss_critic / self.step}, {self.agent.lost_actor / self.step}\n")


        if (self.agent.episode % MODEL_STORE_INTERVAL == 0) or (self.agent.episode == 1):
            self.agent.sm.save_session(self.agent.episode, self.agent.model.networks, self.agent.graph.graphdata, self.agent.replay_buffer.buffer)
            # logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
        if (self.agent.episode % GRAPH_DRAW_INTERVAL == 0) or (self.agent.episode == 1):
            self.agent.graph.draw_plots(self.agent.episode)

        self.episode_start = time.perf_counter()

