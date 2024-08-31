from numpy.core.numeric import inf
# from .settings import COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM
import time
import os

class Logger():
    def __init__(self, training, machine_dir, session_dir, session, hyperparameters, model_config, symbol, algorithm, load_episode):
        self.test_entry = 0
        self.test_outcome = 0
        self.test_distance = []
        self.test_duration = []
        self.test_swerving = []
        self.is_training = training
        self.liquifid_cnt = 0

        self.session = session
        self.hyperparameters = hyperparameters
        self.model_config = model_config
        self.symbol = symbol
        self.algorithm = algorithm

        self.highest_reward = -inf
        self.best_episode_reward = 0
        self.highest_avg_profit = 0
        self.best_episode_avg_profit = 0

        datetime = time.strftime("%Y%m%d-%H%M%S")
        symbol = symbol.replace("/", "-")

        self.file_comparison = self.init_comparison_file(datetime, machine_dir, symbol, hyperparameters, algorithm, session, load_episode)
        if self.is_training:
            self.file_log = self.init_training_log(datetime, session_dir, symbol, model_config)
        else:
            self.file_log = self.init_testing_log(datetime, session_dir, symbol, load_episode)

    def update_test_results(self, step, outcome, episode_duration):
        self.test_entry += 1
        self.test_outcome = outcome

        self.file_log.write(f"{self.test_entry}, {outcome}, {step}, {episode_duration}, {self.test_outcome}\n")
        if self.test_entry > 0:
            print(f"Profit: {outcome} ({outcome/self.test_entry:.2%}), ")
            if outcome == "Liquifided":
                self.liquifid_cnt += 1
                self.file_log.write(f"Liquifided count = {self.liquifid_cnt}\n")
        if self.test_entry > 0 and self.test_entry % 100 == 0:
            self.update_comparison_file(self.test_entry, outcome / (self.test_entry / 100), 0)
            self.file_log.write(f"Profit: {outcome} ({outcome/self.test_entry:.2%}), ")
            if outcome == "Liquifided":
                self.file_log.write(f"Liquifided count = {self.liquifid_cnt}\n")


    def init_training_log(self, datetime, path, symbol, model_config):
        file_log = open(os.path.join(path, "_train_" + symbol + "_" + datetime + '.txt'), 'w+')
        file_log.write("episode, reward, balance, duration, steps, total_steps, memory length, avg_critic_loss, avg_actor_loss\n")
        with open(os.path.join(path, '_model_configuration_' + datetime + '.txt'), 'w+') as file_model_config:
            file_model_config.write(model_config + '\n')
        return file_log

    def init_testing_log(self, datetime, path, symbol, load_episode):
        file_log = open(os.path.join(path, "_test_stage" + symbol + "_eps" + str(load_episode) + "_" + datetime + '.txt'), 'w+')
        file_log.write("episode, outcome, step, episode_duration\n")
        return file_log

    def init_comparison_file(self, datetime, path, symbol, hyperparameters, algorithm, session, episode):
        prefix = "_training" if self.is_training else "_testing"
        with open(os.path.join(path, "__" + algorithm + prefix + "_comparison.txt"), 'a+') as file_comparison:
            file_comparison.write(datetime + ', ' + session + ', ' + str(episode) + ', ' + symbol + ', ' + hyperparameters + '\n')
        return file_comparison

    def update_comparison_file(self, episode, avg_prifit, average_reward=0):
        if average_reward > self.highest_reward and episode != 1:
            self.highest_reward = average_reward
            self.best_episode_reward = episode
        if avg_prifit > self.highest_avg_profit and episode != 1:
            self.highest_avg_profit = avg_prifit
            self.best_episode_avg_profit = episode
        datetime = time.strftime("%Y%m%d-%H%M%S")
        with open(self.file_comparison.name, 'a+') as file_comparison:
            file_comparison.seek(0)
            lines = file_comparison.readlines()
            file_comparison.seek(0)
            file_comparison.truncate()
            file_comparison.writelines(lines[:-1])
            file_comparison.write(datetime + ', ' + self.session + ', ' + self.symbol + ', ' + self.hyperparameters)
            if self.is_training:
                file_comparison.write(', results, ' + str(episode) + ', ' + str(self.best_episode_avg_profit) + ': ' + str(self.highest_avg_profit) + '%, ' + str(self.best_episode_reward) + ': ' + str(self.highest_reward) + '\n')
            else:
                file_comparison.write(', results, ' + str(episode) + ', ' + str(self.best_episode_avg_profit) + ', ' + str(self.highest_avg_profit) + '%\n')
