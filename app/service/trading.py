import ccxt.pro as ccxtpro
import asyncio
import pandas as pd
from abc import ABC, abstractmethod
import importlib


class Strategy(ABC):
    def __init__(self, data):
        self.data = data
        self.position = None
        self.init()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def next(self):
        pass


class TradingService:
    def __init__(self, strategy_name, symbol='BTC/USDT', timeframe='1m', initial_balance=10000, is_dry_run=True,
                 api_key=None, secret_key=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.is_dry_run = is_dry_run
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        strategy_module = importlib.import_module("app.service.strategy")
        self.StrategyClass = getattr(strategy_module, strategy_name)
        self.strategy = None

        self.exchange = ccxtpro.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        self.is_running = False
        self.task = None

    async def fetch_ohlcv(self):
        if self.exchange.has['watchOHLCV']:
            while self.is_running:
                try:
                    trades = await self.exchange.watch_trades(self.symbol)
                    ohlcvc = self.exchange.build_ohlcvc(trades, self.timeframe)

                    for candle in ohlcvc:
                        timestamp, open_price, high, low, close, volume = candle
                        new_row = pd.DataFrame({
                            'timestamp': [pd.to_datetime(timestamp, unit='ms')],
                            'open': [open_price],
                            'high': [high],
                            'low': [low],
                            'close': [close],
                            'volume': [volume]
                        })
                        self.data = pd.concat([self.data, new_row]).reset_index(drop=True)
                        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
                        self.data = self.data.tail(100)

                        if len(self.data) >= 20:
                            if self.strategy is None:
                                self.strategy = self.StrategyClass(self.data)
                            signal = self.strategy.next()
                            if signal:
                                await self.execute_trade(signal)

                except Exception as e:
                    print(f"An error occurred: {e}")
                    await asyncio.sleep(1)

    async def execute_trade(self, signal):
        current_price = self.data['close'].iloc[-1]

        if self.is_dry_run:
            if signal == 'buy' and self.balance > 0:
                amount = (self.balance * 0.95) / current_price
                cost = amount * current_price
                self.balance -= cost
                self.position += amount
                print(f"Dry Run: Buy {amount} {self.symbol} at {current_price}")
            elif signal == 'sell' and self.position > 0:
                revenue = self.position * current_price
                self.balance += revenue
                print(f"Dry Run: Sell {self.position} {self.symbol} at {current_price}")
                self.position = 0
        else:
            if signal == 'buy':
                amount = (self.balance * 0.95) / current_price
                order = await self.exchange.create_market_buy_order(self.symbol, amount)
                print(f"Live: Buy order executed: {order}")
            elif signal == 'sell':
                positions = await self.exchange.fetch_positions([self.symbol])
                for position in positions:
                    if position['side'] == 'long' and position['amount'] > 0:
                        order = await self.exchange.create_market_sell_order(self.symbol, position['amount'])
                        print(f"Live: Sell order executed: {order}")

        if self.is_dry_run:
            total_value = self.balance + (self.position * current_price)
            print(f"Balance: {self.balance:.2f} USDT, Position: {self.position:.8f} {self.symbol}")
            print(f"Total Value: {total_value:.2f} USDT, PNL: {total_value - self.initial_balance:.2f} USDT")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.task = asyncio.create_task(self.fetch_ohlcv())

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.task:
                self.task.cancel()

    async def close(self):
        self.stop()
        await self.exchange.close()

    def get_status(self):
        return {
            "is_running": self.is_running,
            "balance": self.balance,
            "position": self.position,
            "symbol": self.symbol,
            "strategy": self.StrategyClass.__name__,
            "is_dry_run": self.is_dry_run
        }