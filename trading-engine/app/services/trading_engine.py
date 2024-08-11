# app/trading_bot.py

import asyncio
import importlib
import traceback

import ccxt.pro as ccxt
import pandas as pd
from app.services.broker_service import BrokerService
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.core.config import settings
from app.db.models import BotStrategy


class TradingEngine:
    def __init__(self, bot_strategy: BotStrategy):
        self.bot_strategy = bot_strategy
        self.bot = bot_strategy.bot
        self.db_strategy = bot_strategy.strategy

        self.symbol = self.db_strategy.symbol
        self.timeframe = self.db_strategy.timeframe
        self.data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.is_running = False
        self.db: Session = SessionLocal()

        self.broker_service = BrokerService(
            self.db,
            cash=self.bot.cash,
            commission=self.db_strategy.commission,
            dry_run=self.bot.dry_run,
            leverage=self.db_strategy.leverage,
            trade_on_close=self.db_strategy.trade_on_close,
            hedging=self.db_strategy.hedge_mode,
            exclusive_orders=self.db_strategy.exclusive_mode
        )

        strategy_module = importlib.import_module("app.strategy.strategy")
        self.strategy = getattr(strategy_module, self.db_strategy.name)
        self._strategy = None
        self.exchange = getattr(ccxt, self.db_strategy.exchange)()

    async def simulate_ohlcv(self):
        print(f"Starting to simulate OHLCV data for {self.symbol} on {self.exchange.name}")
        last_timestamp = None
        while self.is_running:
            if self.exchange.has['watchOHLCV']:
                try:
                    candles = await self.exchange.watch_ohlcv(self.symbol, self.timeframe, None, 1)
                    for candle in candles:
                        timestamp, open_price, high, low, close, volume = candle
                        if last_timestamp is None or timestamp >= last_timestamp + self.timeframe_to_seconds() * 1000:
                            new_row = pd.DataFrame({
                                'Open': [open_price],
                                'High': [high],
                                'Low': [low],
                                'Close': [close],
                                'Volume': [volume]
                            }, index=[pd.to_datetime(timestamp, unit='ms')], dtype=float)

                            if not new_row.empty:
                                self.data = pd.concat([self.data, new_row],axis=0)
                                self.data = self.data.sort_index()
                                self.data = self.data.tail(100)

                                if len(self.data) >= 2:
                                    self.execute_strategy()

                            last_timestamp = timestamp

                except Exception as e:
                    print("Full traceback:")
                    print(traceback.format_exc())
                    print(f"An error occurred: {e}")
                    await asyncio.sleep(1)

    def timeframe_to_seconds(self):
        unit = self.timeframe[-1]
        amount = int(self.timeframe[:-1])
        if unit == 'm':
            return amount * 60
        elif unit == 'h':
            return amount * 3600
        elif unit == 'd':
            return amount * 86400
        else:
            raise ValueError("Unsupported timeframe unit")

    def execute_strategy(self, **kwargs):
        if self._strategy is None:
            self._strategy = self.strategy(self.broker_service, self.data, kwargs)
            self._strategy.init()

        self.broker_service.update_data(self.data)
        self._strategy.update_data(self.data)

        self.broker_service.process_orders()
        self._strategy.next()

        # 현재 상태 출력
        self.broker_service.print_status()

    async def run(self):
        self.is_running = True
        print("Starting the bot...")
        await self.simulate_ohlcv()

    async def stop(self):
        self.is_running = False
        print("\nFinal Results:")
        print(f"Total Return: {(self.broker_service.equity / settings.INITIAL_CASH - 1) * 100:.2f}%")
        self.db.close()
