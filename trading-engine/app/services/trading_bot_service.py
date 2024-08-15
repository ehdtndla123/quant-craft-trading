from app.repositories import trading_bot_repository
from app.schemas.trading_bot import TradingBotCreate
from sqlalchemy.orm import Session


def create_trading_bot(db: Session, trading_bot: TradingBotCreate):
    return trading_bot_repository.create_trading_bot(db, trading_bot.dict())


def get_trading_bot(db: Session, trading_bot_id: int):
    return trading_bot_repository.get_trading_bot(db, trading_bot_id)


def get_trading_bots(db: Session, skip: int = 0, limit: int = 100):
    return trading_bot_repository.get_trading_bots(db, skip, limit)


def get_trading_bots_by_bot(db: Session, bot_id: int):
    return trading_bot_repository.get_trading_bots_by_bot(db, bot_id)


def get_trading_bots_by_strategy(db: Session, strategy_id: int):
    return trading_bot_repository.get_trading_bots_by_strategy(db, strategy_id)


def delete_trading_bot(db: Session, trading_bot_id: int):
    return trading_bot_repository.delete_trading_bot(db, trading_bot_id)


def get_trading_bot_with_relations(db: Session, trading_bot_id: int):
    return trading_bot_repository.get_trading_bot_with_relations(db, trading_bot_id)
