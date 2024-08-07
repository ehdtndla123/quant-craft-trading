from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.db.database import Base
import datetime



class Bot(Base):
    __tablename__ = "bots"

    id = Column(Integer, primary_key=True, index=True)
    dry_run = Column(Boolean, default=True)
    leverage = Column(Float)
    trade_on_close = Column(Boolean, default=False)
    hedge_mode = Column(Boolean, default=False)
    exclusive_mode = Column(Boolean, default=False)
    name = Column(String, index=True)
    timeframe = Column(String)
    symbol = Column(String)
    exchange = Column(String)
    cash = Column(Float)
    commission = Column(Float)
    strategy_name = Column(String)

    # data_list = Column(String)


    # strategy_id = Column(Integer, ForeignKey('strategies.id'))
    # user_id = Column(Integer, ForeignKey('users.id'))

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # strategy = relationship("Strategy", back_populates="bots")
    # user = relationship("User", back_populates="bots")
    #
    # orders = relationship("Order", back_populates="bot")
    # trades = relationship("Trade", back_populates="bot")

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    size = Column(Float)
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    status = Column(String)
    is_contingent = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

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
    status = Column(String)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

    sl_order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)
    tp_order_id = Column(Integer, ForeignKey('orders.id'), nullable=True)

    sl_order = relationship("Order", foreign_keys=[sl_order_id], overlaps="trade")
    tp_order = relationship("Order", foreign_keys=[tp_order_id], overlaps="trade")

    orders = relationship("Order", foreign_keys=[Order.trade_id], back_populates="trade")
    child_orders = relationship("Order", foreign_keys=[Order.parent_trade_id], back_populates="parent_trade")

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return self.size < 0