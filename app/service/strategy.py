# from backtesting import Strategy
from app.model.Strategy import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd


class MyStrategy(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

    def update_data(self, new_data: pd.DataFrame):
        super().update_data(new_data)
        self.init()
