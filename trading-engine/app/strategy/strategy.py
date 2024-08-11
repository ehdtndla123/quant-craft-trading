from app.model.Strategy import Strategy
from backtesting.test import SMA
import pandas as pd


class MyStrategy(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if len(self.data) % 2 == 0:
            self.buy(size=2)
        else:
            self.sell(size=2)

    def update_data(self, new_data: pd.DataFrame):
        super().update_data(new_data)
        self.init()
