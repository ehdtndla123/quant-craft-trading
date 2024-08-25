from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Enum, Text
from sqlalchemy.orm import relationship
from app.db.database import Base
import datetime
import enum


class OrderStatus(enum.Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"




class Backtesting(Base):
    __tablename__ = "backtesting"

    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, index=True)
    start_date = Column(String)
    end_date = Column(String)
    initial_capital = Column(Float)
    final_equity = Column(Float)
    total_return = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
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
    exclusive_orders = Column(Boolean, default=False)
    hedge_mode = Column(Boolean, default=False)
    timeframe = Column(String)
    symbol = Column(String)
    exchange = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    trading_bots = relationship("TradingBot", back_populates="strategy")


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

    trading_bots = relationship("TradingBot", back_populates="bot")


class TradingbotStatus(enum.Enum):
    # 대기중
    PENDING = "대기중"
    RUNNING = "실행중"
    STOPPING = "중지중"
    STOPPED = "중지됨"


class TradingBot(Base):
    __tablename__ = "trading_bots"
    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey('bots.id'))
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    bot = relationship("Bot", back_populates="trading_bots")
    strategy = relationship("Strategy", back_populates="trading_bots")
    orders = relationship("Order", back_populates="trading_bot")
    trades = relationship("Trade", back_populates="trading_bot")
    status = Column(Enum(TradingbotStatus), default=TradingbotStatus.PENDING)


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

    trading_bot_id = Column(Integer, ForeignKey('trading_bots.id'))
    trading_bot = relationship("TradingBot", back_populates="orders")

    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=True)
    trade = relationship("Trade", foreign_keys=[trade_id], back_populates="orders")

    parent_order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)
    parent_order = relationship("Order", remote_side=[id], backref="child_orders")

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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    trading_bot_id = Column(Integer, ForeignKey('trading_bots.id'))
    trading_bot = relationship("TradingBot", back_populates="trades")

    orders = relationship("Order", back_populates="trade")

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return self.size < 0
