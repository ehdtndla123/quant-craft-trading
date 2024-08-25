from sqlalchemy.orm import Session
from app.db.models import Backtesting
from typing import List, Dict
from app.schemas.backtesting import BacktestingCreate
import json


def create_backtesting(db: Session, backtesting_data: BacktestingCreate) -> Backtesting:
    new_backtesting = Backtesting(**backtesting_data.dict())

    db.add(new_backtesting)
    db.commit()
    db.refresh(new_backtesting)
    return new_backtesting


def get_backtesting(db: Session, backtesting_id: int) -> Backtesting:
    return db.query(Backtesting).filter(Backtesting.id == backtesting_id).first()


def get_backtestings(db: Session, skip: int = 0, limit: int = 100) -> List[Backtesting]:
    return db.query(Backtesting).offset(skip).limit(limit).all()


def delete_backtesting(db: Session, backtesting_id: int) -> bool:
    db_backtesting = get_backtesting(db, backtesting_id)
    if db_backtesting:
        db.delete(db_backtesting)
        db.commit()
        return True
    return False
