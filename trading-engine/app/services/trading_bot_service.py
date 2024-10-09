from app.repositories import trading_bot_repository
from app.schemas.trading_bot import TradingBotCreate
from sqlalchemy.orm import Session
from app.db.models import TradingBotStatus


def create_trading_bot(db: Session, trading_bot: TradingBotCreate):
    return trading_bot_repository.create_trading_bot(db, trading_bot.dict())


def get_trading_bot(db: Session, trading_bot_id: int):
    return trading_bot_repository.get_trading_bot(db, trading_bot_id)


def get_trading_bots(db: Session, skip: int = 0, limit: int = 100):
    return trading_bot_repository.get_trading_bots(db, skip, limit)


def get_trading_bots_by_strategy(db: Session, strategy_id: int):
    return trading_bot_repository.get_trading_bots_by_strategy(db, strategy_id)


def get_trading_bots_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return trading_bot_repository.get_trading_bots_by_user(db, user_id, skip=skip, limit=limit)


def delete_trading_bot(db: Session, trading_bot_id: int):
    return trading_bot_repository.delete_trading_bot(db, trading_bot_id)


def get_trading_bot_with_relations(db: Session, trading_bot_id: int):
    return trading_bot_repository.get_trading_bot_with_relations(db, trading_bot_id)


def update_trading_bot_status(db: Session, trading_bot_id: int, status: TradingBotStatus):
    return trading_bot_repository.update_trading_bot_status(db, trading_bot_id, status)


def get_trading_bots_by_status(db: Session, status: TradingBotStatus, skip: int = 0, limit: int = 100):
    return trading_bot_repository.get_trading_bots_by_status(db, status, skip, limit)
