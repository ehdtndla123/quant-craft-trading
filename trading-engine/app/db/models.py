from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Enum, Text
from sqlalchemy.orm import relationship
from app.db.database import Base
import datetime
import enum


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
    status = Column(Enum(TradingbotStatus), default=TradingbotStatus.PENDING)
