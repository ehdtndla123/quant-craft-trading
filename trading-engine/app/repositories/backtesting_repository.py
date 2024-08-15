from sqlalchemy.orm import Session
from app.db.models import Backtesting
from typing import List, Dict
import json


def create_backtesting(db: Session, backtesting_data: Dict) -> Backtesting:
    new_backtesting = Backtesting(
        strategy_id=backtesting_data['strategy_id'],
        parameters=json.dumps(backtesting_data['parameters']),
        results=json.dumps(backtesting_data['results']),
        trades=json.dumps(backtesting_data['trades']),
        equity_curve=json.dumps(backtesting_data['equity_curve'])
    )
    db.add(new_backtesting)
    db.commit()
    db.refresh(new_backtesting)
    return new_backtesting


def get_backtesting(db: Session, backtesting_id: int) -> Backtesting:
    return db.query(Backtesting).filter(Backtesting.id == backtesting_id).first()


def get_backtestings(db: Session, skip: int = 0, limit: int = 100) -> List[Backtesting]:
    return db.query(Backtesting).offset(skip).limit(limit).all()


def update_backtesting(db: Session, backtesting_id: int, backtesting_data: Dict) -> Backtesting:
    db_backtesting = get_backtesting(db, backtesting_id)
    if db_backtesting:
        for key, value in backtesting_data.items():
            if key in ['parameters', 'results', 'trades', 'equity_curve']:
                setattr(db_backtesting, key, json.dumps(value))
            else:
                setattr(db_backtesting, key, value)
        db.commit()
        db.refresh(db_backtesting)
    return db_backtesting


def delete_backtesting(db: Session, backtesting_id: int) -> bool:
    db_backtesting = get_backtesting(db, backtesting_id)
    if db_backtesting:
        db.delete(db_backtesting)
        db.commit()
        return True
    return False
