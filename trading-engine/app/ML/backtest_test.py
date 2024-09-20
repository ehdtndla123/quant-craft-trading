from app.ML.DRL.env.drl_backtesting import Backtest
# from backtesting import Backtest
import pprint
import importlib
from app.services.data_loader_service import DataLoaderService as DataLoader

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
        # strategy_module = importlib.import_module(f"app.ML.{strategy_name}")
        strategy_module = importlib.import_module("app.ML.drl_test_stratgy")
        StrategyClass = getattr(strategy_module, strategy_name)

        bt = Backtest(data, StrategyClass, commission=commission, cash=cash,
                    exclusive_orders=exclusive_orders, margin=margin, **kwargs)


        while True:
            stats = bt.run()
            pprint.pprint(stats)
            bt.plot()