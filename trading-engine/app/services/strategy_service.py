from app.repositories import strategy_repository
from app.schemas.strategy import StrategyCreate, StrategyUpdate
from sqlalchemy.orm import Session


def create_strategy(db: Session, strategy_data: StrategyCreate):
    return strategy_repository.create_strategy(db, strategy_data.dict())


def get_strategy(db: Session, strategy_id: int):
    return strategy_repository.get_strategy(db, strategy_id)


def get_strategies(db: Session, skip: int = 0, limit: int = 100):
    return strategy_repository.get_strategies(db, skip, limit)


def update_strategy(db: Session, strategy_id: int, strategy_data: StrategyUpdate):
    return strategy_repository.update_strategy(db, strategy_id, strategy_data.dict())


def delete_strategy(db: Session, strategy_id: int):
    return strategy_repository.delete_strategy(db, strategy_id)
