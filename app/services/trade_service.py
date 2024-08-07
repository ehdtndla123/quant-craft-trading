from sqlalchemy.orm import Session
from app.db.models import Trade
from math import copysign
from typing import List
from typing import Dict, Any


def create_trade(db: Session, trade_data: dict) -> Trade:
    trade = Trade(**trade_data)
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def get_trade(db: Session, trade_id: int) -> Trade:
    return db.query(Trade).filter(Trade.id == trade_id).first()


def get_open_trades(db: Session) -> List[Trade]:
    return db.query(Trade).filter(Trade.exit_price == None).all()


def get_closed_trades(db: Session) -> List[Trade]:
    return db.query(Trade).filter(Trade.exit_price != None).all()


def is_trade_open(db: Session, trade_id: int) -> bool:
    trade = get_trade(db, trade_id)
    return trade and trade.exit_price is None


def is_trade_closed(db: Session, trade_id: int) -> bool:
    trade = get_trade(db, trade_id)
    return trade and trade.exit_price is not None


def close_trade(db: Session, trade_id: int, exit_price: float, exit_time):
    trade = get_trade(db, trade_id)
    if trade and trade.exit_price is None:
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        db.commit()
        db.refresh(trade)
    return trade


def calculate_pl(trade: Trade, current_price: float) -> float:
    price = trade.exit_price or current_price
    return trade.size * (price - trade.entry_price)


def calculate_pl_pct(trade: Trade, current_price: float) -> float:
    price = trade.exit_price or current_price
    return copysign(1, trade.size) * (price / trade.entry_price - 1)


def calculate_value(trade: Trade, current_price: float) -> float:
    price = trade.exit_price or current_price
    return abs(trade.size) * price


def update_trade(db: Session, trade_id: int, update_data: Dict[str, Any]) -> Trade:
    trade = get_trade(db, trade_id)
    if trade:
        for key, value in update_data.items():
            if hasattr(trade, key):
                setattr(trade, key, value)
        db.commit()
        db.refresh(trade)
    return trade


def get_opposite_trades(db: Session, size: int) -> List[Trade]:
    """
    주어진 size와 반대되는 포지션의 열린 거래들을 반환합니다.
    size가 양수면 숏 포지션을, 음수면 롱 포지션의 거래들을 반환합니다.
    """
    if size > 0:  # 롱 포지션이므로, 숏 포지션 거래들을 찾습니다
        return db.query(Trade).filter(Trade.size < 0, Trade.exit_price == None).order_by(Trade.entry_time).all()
    elif size < 0:  # 숏 포지션이므로, 롱 포지션 거래들을 찾습니다
        return db.query(Trade).filter(Trade.size > 0, Trade.exit_price == None).order_by(Trade.entry_time).all()
    else:  # size가 0인 경우, 빈 리스트를 반환합니다
        return []


def get_trade_info(trade: Trade, current_price: float) -> dict:
    return {
        "id": trade.id,
        "size": trade.size,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "is_long": trade.is_long,
        "is_short": trade.is_short,
        "pl": calculate_pl(trade, current_price),
        "pl_pct": calculate_pl_pct(trade, current_price),
        "value": calculate_value(trade, current_price)
    }
