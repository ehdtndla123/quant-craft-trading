from sqlalchemy.orm import Session
from app.db.models import Order
from typing import List


def create_order(db: Session, order_data: dict, trade_id: int = None) -> Order:
    order = Order(**order_data)
    if trade_id:
        order.trade_id = trade_id
        order.is_contingent = True
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def get_order(db: Session, order_id: int) -> Order:
    return db.query(Order).filter(Order.id == order_id).first()


def get_orders(db: Session) -> List[Order]:
    return db.query(Order).all()


def get_open_orders(db: Session) -> List[Order]:
    return db.query(Order).filter(Order.status == "PENDING").all()


def update_order(db: Session, order_id: int, order_data: dict) -> Order:
    db.query(Order).filter(Order.id == order_id).update(order_data)
    db.commit()
    return get_order(db, order_id)


def cancel_order(db: Session, order_id: int) -> Order:
    order = get_order(db, order_id)
    if order:
        order.status = "CANCELLED"
        db.commit()
        db.refresh(order)
    return order


def is_order_open(db: Session, order_id: int) -> bool:
    order = get_order(db, order_id)
    return order and order.status == "PENDING"
