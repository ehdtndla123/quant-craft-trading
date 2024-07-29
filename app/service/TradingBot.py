import asyncio
import importlib
import traceback

import ccxt.pro as ccxt
import pandas as pd
from backtesting.backtesting import _Broker, Strategy, _Data
from backtesting.backtesting import partial


class TradingBot:
    def __init__(self, exchange_name: str, symbol: str, timeframe: str, strategy_name: str, initial_cash: float,
                 commission: float = 0.001):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_cash = initial_cash
        self.commission = commission
        self.data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.is_running = False

        self._data = _Data(self.data)  # _Data 객체 생성

        self._broker = partial(
            _Broker, cash=initial_cash, commission=commission, margin=1.,
            trade_on_close=False, hedging=False,
            exclusive_orders=False, index=self.data.index,
        )

        self.broker: _Broker = self._broker(data=self._data)

        strategy_module = importlib.import_module("app.service.strategy")
        self.strategy = getattr(strategy_module, strategy_name)
        self._strategy = None
        self.exchange = getattr(ccxt, exchange_name)()

    async def simulate_ohlcv(self):
        print(f"Starting to simulate OHLCV data for {self.symbol} on {self.exchange.name}")
        last_timestamp = None
        while self.is_running:
            if self.exchange.has['watchOHLCV']:
                try:
                    candles = await self.exchange.watch_ohlcv(self.symbol, self.timeframe, None, 1)
                    print(self.exchange.iso8601(self.exchange.milliseconds()), candles)
                    for candle in candles:
                        timestamp, open_price, high, low, close, volume = candle
                        if last_timestamp is None or timestamp >= last_timestamp + self.timeframe_to_seconds() * 1000:
                            new_row = pd.DataFrame({
                                'Open': [open_price],
                                'High': [high],
                                'Low': [low],
                                'Close': [close],
                                'Volume': [volume]
                            }, index=[pd.to_datetime(timestamp, unit='ms')])

                            if not new_row.empty:
                                self.data = pd.concat([self.data, new_row])
                                self.data = self.data.sort_index()
                                self.data = self.data.tail(100)

                                # 브로커와 전략의 데이터 갱신
                                print("Updating data...")
                                self.broker.update_data(self.data)
                                print("Data updated.")

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
            print("Initializing the strategy...")
            self._strategy = self.strategy
            strategy: Strategy = self._strategy(self.broker, self._data, kwargs)
            self.strategy = strategy
            strategy.init()

        print("Executing the strategy...")
        self.broker.next()  # 브로커의 next 메서드 호출
        self.strategy.next()
        print("Strategy executed.")

        # 현재 상태 출력
        print(f"Timestamp: {self.data.index[-1]}")
        print(f"Price: {self.data['Close'].iloc[-1]}")
        print("---")
        print("=== Broker 상태 출력 ===")
        print(f"Cash: {self.broker._cash}")
        print(f"Position Size: {self.broker.position.size}")
        print(f"Equity: {self.broker.equity}")
        print(f"Open Orders: {len(self.broker.orders)}")
        print(f"Active Trades: {len(self.broker.trades)}")
        print(f"Closed Trades: {len(self.broker.closed_trades)}")
        print("=== Broker 상태 출력 ===")
        print(self.broker.orders)
        print(self.broker.trades)
        print(self.broker.closed_trades)

        self.print_all_orders()
        self.print_trades()

    def print_all_orders(self):
        orders = self.broker.orders
        if not orders:
            print("No orders available.")
            return

        for i, order in enumerate(orders, 1):
            print(f"\nOrder {i}:")
            print(f"  Size: {order.size}")
            print(f"  Limit Price: {order.limit if order.limit is not None else 'N/A'}")
            print(f"  Stop Price: {order.stop if order.stop is not None else 'N/A'}")
            print(f"  Stop Loss Price: {order.sl if order.sl is not None else 'N/A'}")
            print(f"  Take Profit Price: {order.tp if order.tp is not None else 'N/A'}")
            print(f"  Is Long: {order.is_long}")
            print(f"  Is Short: {order.is_short}")
            print(f"  Is Contingent: {order.is_contingent}")

            if order.parent_trade:
                print("  Parent Trade Info:")
                print(f"    Entry Time: {order.parent_trade.entry_time}")
                print(f"    Entry Price: {order.parent_trade.entry_price}")
                print(f"    Size: {order.parent_trade.size}")

        print(f"\nTotal Orders: {len(orders)}")

    def print_trades(self):
        trades = self.broker.trades
        if isinstance(trades, pd.DataFrame):
            trades_df = trades
        else:
            trades_df = pd.DataFrame({
                'Size': [t.size for t in trades],
                'EntryBar': [t.entry_bar for t in trades],
                'ExitBar': [t.exit_bar for t in trades],
                'EntryPrice': [t.entry_price for t in trades],
                'ExitPrice': [t.exit_price for t in trades],
                'PnL': [t.pl for t in trades],
                'ReturnPct': [t.pl_pct for t in trades],
                'EntryTime': [t.entry_time for t in trades],
                'ExitTime': [t.exit_time for t in trades],
            })
            trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']

        # 주요 정보만 선택
        summary_df = trades_df[
            ['Size', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'EntryTime', 'ExitTime', 'Duration']]

        # 결과 출력
        print("\nTrades Summary:")
        print(summary_df.to_string(index=False))

        # 추가 통계 정보
        print("\nTrade Statistics:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Profitable Trades: {sum(trades_df['PnL'] > 0)}")
        print(f"Loss-making Trades: {sum(trades_df['PnL'] < 0)}")
        print(f"Total Profit/Loss: {trades_df['PnL'].sum():.2f}")
        print(f"Average Profit/Loss per Trade: {trades_df['PnL'].mean():.2f}")
        print(f"Average Return Percentage: {trades_df['ReturnPct'].mean():.2%}")
        print(f"Average Trade Duration: {trades_df['Duration'].mean()}")

        if len(trades_df) > 0:
            print(f"Best Trade: {trades_df['PnL'].max():.2f}")
            print(f"Worst Trade: {trades_df['PnL'].min():.2f}")

    async def run(self):
        self.is_running = True
        print("Starting the bot...")
        await self.simulate_ohlcv()

    async def stop(self):
        self.is_running = False
        print("\nFinal Results:")
        print(f"Total Return: {(self.broker.equity / self.initial_cash - 1) * 100:.2f}%")
        print(f"Number of Trades: {len(self.broker.closed_trades)}")