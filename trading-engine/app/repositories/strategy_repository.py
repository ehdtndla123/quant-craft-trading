from sqlalchemy.orm import Session
from app.db.models import Strategy
from app.schemas.strategy import StrategyCreate, StrategyUpdate


def create_strategy(db: Session, strategy_data: StrategyCreate) -> Strategy:
    new_strategy = Strategy(**strategy_data.dict())
    db.add(new_strategy)
    db.commit()
    db.refresh(new_strategy)
    return new_strategy


def get_strategy(db: Session, strategy_id: int) -> Strategy:
    return db.query(Strategy).filter(Strategy.id == strategy_id).first()


def get_strategies(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Strategy).offset(skip).limit(limit).all()


def update_strategy(db: Session, strategy_id: int, strategy_data: StrategyUpdate) -> Strategy:
    db_strategy = get_strategy(db, strategy_id)
    if db_strategy:
        update_data = strategy_data.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_strategy, key, value)
        db.commit()
        db.refresh(db_strategy)
    return db_strategy


def delete_strategy(db: Session, strategy_id: int) -> bool:
    db_strategy = get_strategy(db, strategy_id)
    if db_strategy:
        db.delete(db_strategy)
        db.commit()
        return True
    return False
