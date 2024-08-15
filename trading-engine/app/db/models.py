from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Text, Table, Enum
from sqlalchemy.orm import relationship
from app.db.database import Base
import datetime
import enum


class OrderStatus(enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"


class TradeStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class Backtesting(Base):
    __tablename__ = "backtesting"

    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, index=True)
    start_date = Column(String)
    end_date = Column(String)
    parameters = Column(Text)
    results = Column(Text)
    trades = Column(Text)
    equity_curve = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    strategy_id = Column(Integer, ForeignKey("strategies.id"), index=True)
    strategy = relationship("Strategy")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    bots = relationship("Bot", back_populates="user")


class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    leverage = Column(Float)
    trade_on_close = Column(Boolean, default=False)
    hedge_mode = Column(Boolean, default=False)
    exclusive_mode = Column(Boolean, default=False)
    timeframe = Column(String)
    symbol = Column(String)
    exchange = Column(String)
    commission = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    bot_strategies = relationship("BotStrategy", back_populates="strategy")


class Bot(Base):
    __tablename__ = "bots"
    id = Column(Integer, primary_key=True, index=True)
    dry_run = Column(Boolean, default=True)
    name = Column(String)
    cash = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("User", back_populates="bots")

    bot_strategies = relationship("BotStrategy", back_populates="bot")


class TradingBot(Base):
    __tablename__ = "trading_bots"
    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey('bots.id'))
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    bot = relationship("Bot", back_populates="bot_strategies")
    strategy = relationship("Strategy", back_populates="bot_strategies")
    orders = relationship("Order", back_populates="bot_strategy")
    trades = relationship("Trade", back_populates="bot_strategy")


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    size = Column(Float)
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus))
    is_contingent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    bot_strategy_id = Column(Integer, ForeignKey('bot_strategies.id'))
    bot_strategy = relationship("BotStrategy", back_populates="orders")

    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=True)
    parent_trade_id = Column(Integer, ForeignKey('trades.id'), nullable=True)

    trade = relationship("Trade", foreign_keys=[trade_id], back_populates="orders")
    parent_trade = relationship("Trade", foreign_keys=[parent_trade_id], back_populates="child_orders")

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return self.size < 0


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, nullable=True)
    status = Column(Enum(TradeStatus))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    bot_strategy_id = Column(Integer, ForeignKey('bot_strategies.id'))
    bot_strategy = relationship("BotStrategy", back_populates="trades")

    sl_order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)
    sl_order = relationship("Order", foreign_keys=[sl_order_id], overlaps="trade")

    tp_order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)
    tp_order = relationship("Order", foreign_keys=[tp_order_id], overlaps="trade")

    orders = relationship("Order", foreign_keys=[Order.trade_id], back_populates="trade")
    child_orders = relationship("Order", foreign_keys=[Order.parent_trade_id], back_populates="parent_trade")

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return self.size < 0
