from app.service.DRL.env.drl_backtesting import Backtest
# from backtesting import Backtest
import pprint
import importlib
from app.service.data_loader import DataLoader


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
        data = DataLoader.load_data_from_ccxt(
            exchange_name=exchange_name,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone
        )

        # 동적으로 전략 클래스 불러오기
        # strategy_module = importlib.import_module(f"app.service.{strategy_name}")
        strategy_module = importlib.import_module(f"app.service.drl_strategy")
        StrategyClass = getattr(strategy_module, strategy_name)

        bt = Backtest(data, StrategyClass, commission=commission, cash=cash,
                      exclusive_orders=exclusive_orders, margin=margin,**kwargs)
        stats = bt.run()
        pprint.pprint(stats)
        bt.plot()

        return stats