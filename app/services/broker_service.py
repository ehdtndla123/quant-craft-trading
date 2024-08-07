# app/services/broker_service.py

from sqlalchemy.orm import Session
from app.services import order_service, trade_service
from app.db.models import Order, Trade
from app.core.config import settings
import pandas as pd
from math import copysign
from typing import List, Optional
import numpy as np
import warnings


class BrokerService:
    def __init__(self, db: Session, cash: float, commission: float, dry_run: bool, leverage: float, trade_on_close: bool, hedging: bool, exclusive_orders: bool):
        self.db = db
        self.cash = cash
        self._commission = commission
        self._leverage = leverage
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders
        self.equityList = []
        self._data = None

    def update_data(self, new_data: pd.DataFrame):
        self._data = new_data.reset_index(drop=True)

    @property
    def last_price(self) -> float:
        return self._data["Close"].iloc[-1] if self._data is not None else 0

    def adjusted_price(self, size: float = None, price: float = None) -> float:
        return (price or self.last_price) * (1 + copysign(self._commission, size))

    @property
    def equity(self) -> float:
        open_trades = trade_service.get_open_trades(self.db)
        return self.cash + sum(trade_service.calculate_pl(trade, self.last_price) for trade in open_trades)

    @property
    def margin_available(self) -> float:
        open_trades = trade_service.get_open_trades(self.db)
        margin_used = sum(
            trade_service.calculate_value(trade, self.last_price) / self._leverage for trade in open_trades)
        return max(0, self.equity - margin_used)

    def new_order(self,
                  size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  *,
                  trade: Optional[Trade] = None) -> Order:
        """
        Argument size indicates whether the order is long or short
        """
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)
        is_long = size > 0
        adjusted_price = self.adjusted_price(size)

        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})")
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})")

        order_data = {
            "size": size,
            "limit_price": limit,
            "stop_price": stop,
            "sl_price": sl,
            "tp_price": tp,
            "status": "PENDING",
            "is_contingent": trade is not None,
            "trade_id": trade.id if trade else None
        }

        # If exclusive orders (each new order auto-closes previous orders/position),
        # cancel all non-contingent orders and close all open trades beforehand
        if self._exclusive_orders and not trade:
            open_orders = order_service.get_open_orders(self.db)
            for order in open_orders:
                if not order.is_contingent:
                    order_service.cancel_order(self.db, order.id)

            open_trades = trade_service.get_open_trades(self.db)
            for open_trade in open_trades:
                trade_service.close_trade(self.db, open_trade.id, self.last_price, pd.Timestamp.now())

        new_order = order_service.create_order(self.db, order_data)

        return new_order

    # def new_order(self, order_data: dict) -> Order:
    #     size = order_data['size']
    #     is_long = size > 0
    #     adjusted_price = self.adjusted_price(size)
    #
    #     if is_long:
    #         if not (order_data.get('sl_price') or -float('inf')) < (
    #                 order_data.get('limit_price') or order_data.get('stop_price') or adjusted_price) < (
    #                        order_data.get('tp_price') or float('inf')):
    #             raise ValueError("Long orders require: SL < LIMIT < TP")
    #     else:
    #         if not (order_data.get('tp_price') or -float('inf')) < (
    #                 order_data.get('limit_price') or order_data.get('stop_price') or adjusted_price) < (
    #                        order_data.get('sl_price') or float('inf')):
    #             raise ValueError("Short orders require: TP < LIMIT < SL")
    #
    #     if self._exclusive_orders:
    #         open_orders = order_service.get_open_orders(self.db)
    #         for order in open_orders:
    #             if not order.is_contingent:
    #                 order_service.cancel_order(self.db, order.id)
    #
    #         open_trades = trade_service.get_open_trades(self.db)
    #         for trade in open_trades:
    #             trade_service.close_trade(self.db, trade.id, self.last_price, pd.Timestamp.now())
    #
    #     return order_service.create_order(self.db, order_data)

    def process_orders(self):
        if self._data is None or len(self._data) < 2:
            return

        open_price = self._data['Open'].iloc[-1]
        high = self._data['High'].iloc[-1]
        low = self._data['Low'].iloc[-1]
        close = self._data['Close'].iloc[-1]
        prev_close = self._data['Close'].iloc[-2]

        open_orders = order_service.get_open_orders(self.db)
        reprocess_orders = False

        for order in open_orders:
            if self._process_single_order(order, open_price, high, low, close, prev_close):
                reprocess_orders = True

        if reprocess_orders:
            self.process_orders()

    def _process_single_order(self, order, open_price, high, low, close, prev_close):
        reprocess = False

        # 스탑 가격 확인
        if order.stop_price:
            is_stop_hit = (high > order.stop_price) if order.is_long else (low < order.stop_price)
            if not is_stop_hit:
                return reprocess
            stop_price = order.stop_price
            order.stop_price = None
        else:
            stop_price = None

        # 지정가 확인
        if order.limit_price:
            is_limit_hit = low < order.limit_price if order.is_long else high > order.limit_price
            is_limit_hit_before_stop = (
                    is_limit_hit and
                    (order.limit_price < (stop_price or -np.inf) if order.is_long
                     else order.limit_price > (stop_price or np.inf))
            )
            if not is_limit_hit or is_limit_hit_before_stop:
                return reprocess
            price = order.limit_price
        else:
            price = prev_close if self._trade_on_close else open_price
            price = (max(price, stop_price or -np.inf) if order.is_long
                     else min(price, stop_price or np.inf))

        # 진입/청산 봉 인덱스 결정
        is_market_order = not order.limit_price and not stop_price

        # 주문이 손절/이익실현 주문인 경우 기존 거래 청산
        if order.parent_trade_id:
            trade = trade_service.get_trade(self.db, order.parent_trade_id)
            if trade:
                prev_size = trade.size
                size = copysign(min(abs(prev_size), abs(order.size)), order.size)
                if trade_service.is_trade_open(self.db, trade.id):
                    self._reduce_trade(trade, price, size)
                if order.is_sl_tp_order:
                    assert order.size == -trade.size
                    assert not order_service.is_order_open(self.db, order.id)
                else:
                    assert abs(prev_size) >= abs(size) >= 1
                    order_service.cancel_order(self.db, order.id)
                return reprocess

        # 주문 크기 계산
        size = self._calculate_order_size(order, price)
        if size == 0:
            order_service.cancel_order(self.db, order.id)
            return reprocess

        # 주문 처리
        if not self._hedging:
            size = self._close_opposite_trades(size, price)

        if size != 0:
            self._execute_order(order, price, size)
            if order.sl_price or order.tp_price:
                reprocess = True

        order_service.update_order(self.db, order.id, {"status": "FILLED"})
        return reprocess

    def _reduce_trade(self, trade: Trade, price: float, size: float):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0

        if not size_left:
            # 거래를 완전히 종료
            trade_service.close_trade(self.db, trade, price, pd.Timestamp.now())
        else:
            # 기존 거래 축소
            trade_service.update_trade(self.db, trade.id, {
                "size": size_left
            })

            # 관련된 SL/TP 주문 업데이트
            if trade.sl_order_id:
                order_service.update_order(self.db, trade.sl_order_id, {
                    "size": -size_left
                })
            if trade.tp_order_id:
                order_service.update_order(self.db, trade.tp_order_id, {
                    "size": -size_left
                })

            # 축소된 부분을 새로운 종료 거래로 생성
            close_trade_data = {
                "size": -size,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time,
                "exit_price": price,
                "exit_time": pd.Timestamp.now(),
            }
            close_trade_id = trade_service.create_trade(self.db, close_trade_data)

            # 새로 생성된 종료 거래 처리
            trade_service.close_trade(self.db, close_trade_id, price, pd.Timestamp.now())

    def _calculate_order_size(self, order: Order, price: float) -> int:
        if -1 < order.size < 1:
            size = int((self.margin_available * self._leverage * abs(order.size)) // price)
            return np.sign(order.size) * max(size, 0)
        return int(order.size)

    def _close_opposite_trades(self, size: int, price: float) -> int:
        opposite_trades = trade_service.get_opposite_trades(self.db, size)
        for trade in opposite_trades:
            if abs(size) >= abs(trade.size):
                trade_service.close_trade(self.db, trade.id, price, pd.Timestamp.now())
                size += trade.size
            else:
                self._reduce_trade(trade, price, size)
                size = 0
            if size == 0:
                break
        return size

    def _execute_order(self, order: Order, price: float, size: int):
        adjusted_price = self.adjusted_price(size, price)
        if abs(size) * adjusted_price > self.margin_available * self._leverage:
            order_service.cancel_order(self.db, order.id)
            return False  # 주문 실행 실패

        if order.trade_id:  # SL/TP order
            trade = trade_service.get_trade(self.db, order.trade_id)
            if trade:
                trade_service.close_trade(self.db, trade.id, price, pd.Timestamp.now())
        else:  # New trade
            trade_data = {
                "size": size,
                "entry_price": adjusted_price,
                "entry_time": pd.Timestamp.now(),
                "exit_price": None,
                "exit_time": None
            }
            new_trade = trade_service.create_trade(self.db, trade_data)

            reprocess_orders = False

            # Create SL order if specified
            if order.sl_price:
                sl_order_data = self._create_sl_tp_order_data(new_trade.id, -size, stop_price=order.sl_price)
                sl_order = order_service.create_order(self.db, sl_order_data)
                reprocess_orders = self._check_sl_tp_execution(sl_order, is_market_order=order.is_market_order)

            # Create TP order if specified
            if order.tp_price:
                tp_order_data = self._create_sl_tp_order_data(new_trade.id, -size, limit_price=order.tp_price)
                tp_order = order_service.create_order(self.db, tp_order_data)
                reprocess_orders = reprocess_orders or self._check_sl_tp_execution(tp_order,
                                                                                   is_market_order=order.is_market_order)

            # self.margin_available -= abs(size) * adjusted_price

        return reprocess_orders

    def _create_sl_tp_order_data(self, trade_id: int, size: float, stop_price: float = None, limit_price: float = None):
        return {
            "size": size,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "status": "PENDING",
            "is_contingent": True,
            "trade_id": trade_id
        }

    def _check_sl_tp_execution(self, order: Order, is_market_order: bool):
        low, high = self._data['Low'].iloc[-1], self._data['High'].iloc[-1]

        if is_market_order:
            return True
        elif (low <= (order.stop_price or -np.inf) <= high or
              low <= (order.limit_price or -np.inf) <= high):
            warnings.warn(
                f"({self._data.index[-1]}) A contingent SL/TP order would execute in the "
                "same bar its parent stop/limit order was turned into a trade. "
                "Since we can't assert the precise intra-candle "
                "price movement, the affected SL/TP order will instead be executed on "
                "the next (matching) price/bar, making the result (of this trade) "
                "somewhat dubious. "
                "See https://github.com/kernc/backtesting.py/issues/119",
                UserWarning)
            return False

        return False

    def get_position(self) -> dict:
        open_trades = trade_service.get_open_trades(self.db)
        size = sum(trade.size for trade in open_trades)
        pl = sum(trade_service.calculate_pl(trade, self.last_price) for trade in open_trades)
        return {
            "size": size,
            "pl": pl,
            "pl_pct": pl / self.cash if self.cash else 0,
            "is_long": size > 0,
            "is_short": size < 0
        }

    def close_position(self, portion: float = 1.0):
        open_trades = trade_service.get_open_trades(self.db)
        for trade in open_trades:
            trade_service.close_trade(self.db, trade.id, self.last_price, pd.Timestamp.now())

    def next(self):
        """시뮬레이션의 다음 단계로 진행"""
        self.process_orders()
        self._update_equity()
        self._check_margin_call()

    def _update_equity(self):
        """계정 자본을 업데이트"""
        self.equityList.append(self.equity)

    def _check_margin_call(self):
        """마진 콜 체크 및 처리"""
        if self.equity <= 0:
            self._close_all_positions()
            raise _OutOfMoneyError("Margin call: Out of money")

    def _close_all_positions(self):
        """모든 포지션 청산"""
        open_trades = trade_service.get_open_trades(self.db)
        for trade in open_trades:
            trade_service.close_trade(self.db, trade.id, self.last_price, pd.Timestamp.now())

    def print_status(self):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Broker 상태 출력 @@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Cash: {self.cash}")
        print(f"Position: {self.get_position()}")
        print(f"Equity: {self.equity}")

        print("\n=====================All Orders=====================")
        orders = order_service.get_open_orders(self.db)
        for i, order in enumerate(orders, 1):
            print(f"\nOrder {i}:")
            print(f"  Size: {order.size}")
            print(f"  Limit Price: {order.limit_price if order.limit_price is not None else 'N/A'}")
            print(f"  Stop Price: {order.stop_price if order.stop_price is not None else 'N/A'}")
            print(f"  Stop Loss Price: {order.sl_price if order.sl_price is not None else 'N/A'}")
            print(f"  Take Profit Price: {order.tp_price if order.tp_price is not None else 'N/A'}")
            print(f"  Is Long: {order.size > 0}")
            print(f"  Is Short: {order.size < 0}")
            print(f"  Is Contingent: {order.is_contingent}")

        print(f"\nTotal Orders: {len(orders)}")

        print("\n=====================Closed Trades=====================")
        self._print_trades(trade_service.get_closed_trades(self.db))

        print("\n=====================Active Trades=====================")
        self._print_trades(trade_service.get_open_trades(self.db))

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    def _print_trades(self, trades: List[Trade]):
        trades_df = pd.DataFrame({
            'Size': [t.size for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price if t.exit_price is not None else self.last_price for t in trades],
            'PnL': [trade_service.calculate_pl(t, self.last_price) for t in trades],
            'ReturnPct': [trade_service.calculate_pl_pct(t, self.last_price) for t in trades],
            'EntryTime': [pd.to_datetime(t.entry_time) for t in trades],
            'ExitTime': [pd.to_datetime(t.exit_time) if t.exit_time is not None else pd.NaT for t in trades],
        })

        # Calculate duration only for closed trades
        trades_df['Duration'] = pd.to_timedelta(trades_df['ExitTime'] - trades_df['EntryTime']).fillna(pd.Timedelta(0))

        print("\nTrades Summary:")
        print(trades_df.to_string(index=False))

        print("\nTrade Statistics:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Profitable Trades: {sum(trades_df['PnL'] > 0)}")
        print(f"Loss-making Trades: {sum(trades_df['PnL'] < 0)}")
        print(f"Total Profit/Loss: {trades_df['PnL'].sum():.2f}")
        print(f"Average Profit/Loss per Trade: {trades_df['PnL'].mean():.2f}")
        print(f"Average Return Percentage: {trades_df['ReturnPct'].mean():.2%}")

        # Calculate average duration only for closed trades
        closed_trades = trades_df[trades_df['ExitTime'].notna()]
        if len(closed_trades) > 0:
            print(f"Average Trade Duration: {closed_trades['Duration'].mean()}")
        else:
            print("Average Trade Duration: N/A (no closed trades)")

        if len(trades_df) > 0:
            print(f"Best Trade: {trades_df['PnL'].max():.2f}")
            print(f"Worst Trade: {trades_df['PnL'].min():.2f}")

    def __repr__(self):
        position = self.get_position()
        return f'<Broker: {self.cash:.0f}{position["pl"]:+.1f} ({len(trade_service.get_open_trades(self.db))} trades)>'

class _OutOfMoneyError(Exception):
    pass

def get_broker_service(db: Session) -> BrokerService:
    return BrokerService(db)