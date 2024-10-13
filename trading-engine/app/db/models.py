from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean, Enum, Text, BigInteger, DECIMAL, \
    func
from sqlalchemy.orm import relationship
from app.db.database import Base
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    strategy_id = Column(Integer, ForeignKey("strategies.id"), index=True)
    strategy = relationship("Strategy", back_populates="backtestings")


class User(Base):
    __tablename__ = "users"
    id = Column(BigInteger, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    trading_bots = relationship("TradingBot", back_populates="user")
    exchange_api_keys = relationship("ExchangeApiKey", back_populates="user")


class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    leverage = Column(Float)
    exclusive_orders = Column(Boolean, default=False)
    hedge_mode = Column(Boolean, default=False)
    timeframe = Column(String)
    symbol = Column(String)
    exchange = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    trading_bots = relationship("TradingBot", back_populates="strategy")
    backtestings = relationship("Backtesting", back_populates="strategy")


class TradingBotStatus(enum.Enum):
    PENDING = "대기중"
    RUNNING = "실행중"
    STOPPING = "중지중"
    STOPPED = "중지됨"


class TradingBot(Base):
    __tablename__ = "trading_bots"

    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String)
    dry_run = Column(Boolean)
    cash = Column(DECIMAL(precision=18, scale=8))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    status = Column(Enum(TradingBotStatus))

    user_id = Column(BigInteger, ForeignKey('users.id'))
    user = relationship("User", back_populates="trading_bots")

    exchange_api_key_id = Column(BigInteger, ForeignKey('exchange_api_keys.id'))
    exchange_api_key = relationship("ExchangeApiKey", back_populates="trading_bots")

    strategy_id = Column(BigInteger, ForeignKey('strategies.id'))
    strategy = relationship("Strategy", back_populates="trading_bots")

    version = Column(BigInteger, nullable=False, default=1)

    __mapper_args__ = {
        'version_id_col': version
    }


class ExchangeApiKey(Base):
    __tablename__ = "exchange_api_keys"

    id = Column(BigInteger, primary_key=True, index=True)
    exchange_type = Column(String)
    api_key = Column(String)
    secret_key = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user_id = Column(BigInteger, ForeignKey('users.id'))
    user = relationship("User", back_populates="exchange_api_keys")

    trading_bots = relationship("TradingBot", back_populates="exchange_api_key")