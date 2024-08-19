import asyncio
import traceback
import pandas as pd
from app.db.models import TradingBot
from app.services.data_loader_service import DataLoaderService
from app.model.broker_interface import IBroker
from app.services.strategy_manager import StrategyManager


class TradingEngine:
    def __init__(self, trading_bot: TradingBot, broker_service: IBroker):
        self.trading_bot = trading_bot
        self.bot = trading_bot.bot
        self.db_strategy = trading_bot.strategy
        self.symbol = self.db_strategy.symbol
        self.timeframe = self.db_strategy.timeframe
        self.data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.is_running = False
        self.broker_service = broker_service
        self.strategy = StrategyManager.get_strategy(self.db_strategy.name)
        self._strategy = None

    def _timeframe_to_seconds(self):
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

    def _execute_strategy(self, **kwargs):
        if self._strategy is None:
            self._strategy = self.strategy(self.broker_service, self.data, kwargs)
            self._strategy.init()

        self.broker_service.update_data(self.data)
        self._strategy.update_data(self.data)

        self.broker_service.process_orders()
        self._strategy.next()

    async def run(self):
        self.is_running = True
        last_timestamp = None
        wait_time = self._timeframe_to_seconds()
        while self.is_running:
            try:
                new_data = await DataLoaderService.fetch_real_time_data(
                    self.db_strategy.exchange, self.symbol, self.timeframe, last_timestamp
                )

                if not new_data.empty:
                    self.data = pd.concat([self.data, new_data], axis=0)
                    self.data = self.data.sort_index()
                    self.data = self.data.tail(100)

                    if len(self.data) >= 2:
                        self._execute_strategy()

                    last_timestamp = self.data.index[-1]
                await asyncio.sleep(wait_time)

            except Exception as e:
                print("Full traceback:")
                print(traceback.format_exc())
                print(f"An error occurred: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        self.is_running = False
