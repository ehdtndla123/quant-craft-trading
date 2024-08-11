from app.model.backtest.Backtest import Backtest
import pprint
import importlib
import pandas as pd


class BacktestManager:
    @staticmethod
    def run_backtest(
            data: str or pd.DataFrame,
            strategy_name: str,
            commission: float,
            cash: float,
            exclusive_orders: bool,
            **kwargs
    ):

        # data가 문자열이면 CSV 파일 경로로, DataFrame으로 로드
        if isinstance(data, str):
            data = pd.read_csv(data, index_col=0, parse_dates=True)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be either a DataFrame or a path to a CSV file.")

        # 동적으로 전략 클래스 불러오기
        strategy_module = importlib.import_module(f"app.strategy.strategy")
        StrategyClass = getattr(strategy_module, strategy_name)

        bt = Backtest(data, StrategyClass, commission=commission, cash=cash,
                      exclusive_orders=exclusive_orders)
        stats = bt.run()
        pprint.pprint(stats)
        bt.plot()

        return stats