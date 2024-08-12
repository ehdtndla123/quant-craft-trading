from sqlalchemy.orm import Session, joinedload
from app.db.models import BotStrategy


def create_bot_strategy(db: Session, bot_strategy_data: dict) -> BotStrategy:
    new_bot_strategy = BotStrategy(**bot_strategy_data)
    db.add(new_bot_strategy)
    db.commit()
    db.refresh(new_bot_strategy)
    return new_bot_strategy


def get_bot_strategy(db: Session, bot_strategy_id: int) -> BotStrategy:
    return db.query(BotStrategy).filter(BotStrategy.id == bot_strategy_id).first()


def get_bot_strategies(db: Session, skip: int = 0, limit: int = 100):
    return db.query(BotStrategy).offset(skip).limit(limit).all()


def get_bot_strategies_by_bot(db: Session, bot_id: int):
    return db.query(BotStrategy).filter(BotStrategy.bot_id == bot_id).all()


def get_bot_strategies_by_strategy(db: Session, strategy_id: int):
    return db.query(BotStrategy).filter(BotStrategy.strategy_id == strategy_id).all()


def delete_bot_strategy(db: Session, bot_strategy_id: int) -> bool:
    db_bot_strategy = get_bot_strategy(db, bot_strategy_id)
    if db_bot_strategy:
        db.delete(db_bot_strategy)
        db.commit()
        return True
    return False


def get_bot_strategy_with_relations(db: Session, bot_strategy_id: int):
    return db.query(BotStrategy).options(
        joinedload(BotStrategy.bot),
        joinedload(BotStrategy.strategy)
    ).filter(BotStrategy.id == bot_strategy_id).first()
