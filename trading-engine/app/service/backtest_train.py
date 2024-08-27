from app.service.DRL.env.drl_backtesting import Backtest
# from backtesting import Backtest
import pprint
import importlib
from app.services.data_loader_service import DataLoaderService as DataLoader


from .DRL.NN.td3_xLSTM import TD3
from .DRL.replaybuffer import ReplayBuffer
from .DRL.storagemanager import StorageManager
from .DRL.graph import Graph
from .DRL.logger import Logger
from .DRL.settings import MODEL_STORE_INTERVAL, N_TRAIN, get_device

import time
import pandas as pd
import os


class BacktestManager:
    @staticmethod
    def run_backtest(
            exchange_name: str,
            symbol: str,
            timeframe: str,
            start_time: str,
            end_time: str,
            timezone: str,
            strategy_name: str,
            commission: float,
            cash: float,
            exclusive_orders: bool,
            margin: float,
            **kwargs
    ):
        file_name = f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_time}_{end_time}.csv"
        file_path = os.path.join('data', file_name)
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            DataLoader.save_data_from_ccxt(
                exchange_name=exchange_name,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                timezone=timezone
            )
        finally:
            data = pd.read_csv(file_path)

        # 동적으로 전략 클래스 불러오기
        # strategy_module = importlib.import_module(f"app.service.{strategy_name}")
        strategy_module = importlib.import_module(f"app.service.drl_simple_strategy")
        StrategyClass = getattr(strategy_module, strategy_name)

        bt = Backtest(data, StrategyClass, commission=commission, cash=cash,
                    exclusive_orders=exclusive_orders, margin=margin, **kwargs)

        # try:
        A = Agent(symbol)
        episode = 0
        while True:
            stats = bt.run(A)
            episode += 1
            if episode % (MODEL_STORE_INTERVAL) == 0 or episode == 1:
                pprint.pprint(stats)
                print(episode)
                bt.plot()
        # except KeyboardInterrupt:
        #     return stats

class Agent:
    def __init__(self, symbol):
        self.training = True
        self.state_size = N_TRAIN
        self.device = get_device()

        # Create DRL Agent
        self.model = TD3(self.device)

        # Create Storage Manager
        self.sm = StorageManager('td3', '', 0, self.device)
        self.sm.new_session_dir()
        self.sm.store_model(self.model)

        # Create Graph Plotter
        self.graph = Graph()
        self.graph.session_dir = self.sm.session_dir

        # Create Replay Buffer
        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.episode, self.loss_critic, self.loss_actor, self.total_step = 0, 0, 0, 0
        self.episode_start = time.perf_counter()

        # Create Logger
        self.logger = Logger(self.training, self.sm.machine_dir, self.sm.session_dir, self.sm.session, self.model.get_model_parameters(), self.model.get_model_configuration(), symbol, "td3", self.episode)
