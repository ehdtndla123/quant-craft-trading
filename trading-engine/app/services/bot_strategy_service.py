from app.repositories import bot_strategy_repository
from app.schemas.bot_strategy import BotStrategyCreate
from sqlalchemy.orm import Session


def create_bot_strategy(db: Session, bot_strategy: BotStrategyCreate):
    return bot_strategy_repository.create_bot_strategy(db, bot_strategy.dict())


def get_bot_strategy(db: Session, bot_strategy_id: int):
    return bot_strategy_repository.get_bot_strategy(db, bot_strategy_id)


def get_bot_strategies(db: Session, skip: int = 0, limit: int = 100):
    return bot_strategy_repository.get_bot_strategies(db, skip, limit)


def get_bot_strategies_by_bot(db: Session, bot_id: int):
    return bot_strategy_repository.get_bot_strategies_by_bot(db, bot_id)


def get_bot_strategies_by_strategy(db: Session, strategy_id: int):
    return bot_strategy_repository.get_bot_strategies_by_strategy(db, strategy_id)


def delete_bot_strategy(db: Session, bot_strategy_id: int):
    return bot_strategy_repository.delete_bot_strategy(db, bot_strategy_id)


def get_bot_strategy_with_relations(db: Session, bot_strategy_id: int):
    return bot_strategy_repository.get_bot_strategy_with_relations(db, bot_strategy_id)
