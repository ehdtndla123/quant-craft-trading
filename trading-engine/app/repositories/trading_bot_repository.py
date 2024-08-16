from sqlalchemy.orm import Session, joinedload
from app.db.models import TradingBot, TradingbotStatus


def create_trading_bot(db: Session, trading_bot_data: dict) -> TradingBot:
    new_trading_bot = TradingBot(**trading_bot_data)
    db.add(new_trading_bot)
    db.commit()
    db.refresh(new_trading_bot)
    return new_trading_bot


def get_trading_bot(db: Session, trading_bot_id: int) -> TradingBot:
    return db.query(TradingBot).filter(TradingBot.id == trading_bot_id).first()


def get_trading_bots(db: Session, skip: int = 0, limit: int = 100):
    return db.query(TradingBot).offset(skip).limit(limit).all()


def get_trading_bots_by_bot(db: Session, bot_id: int):
    return db.query(TradingBot).filter(TradingBot.bot_id == bot_id).all()


def get_trading_bots_by_strategy(db: Session, strategy_id: int):
    return db.query(TradingBot).filter(TradingBot.strategy_id == strategy_id).all()


def delete_trading_bot(db: Session, trading_bot_id: int) -> bool:
    db_trading_bot = get_trading_bot(db, trading_bot_id)
    if db_trading_bot:
        db.delete(db_trading_bot)
        db.commit()
        return True
    return False


def get_trading_bot_with_relations(db: Session, trading_bot_id: int):
    return db.query(TradingBot).options(
        joinedload(TradingBot.bot),
        joinedload(TradingBot.strategy)
    ).filter(TradingBot.id == trading_bot_id).first()


def update_trading_bot_status(db: Session, trading_bot_id: int, status: TradingbotStatus):
    trading_bot = get_trading_bot(db, trading_bot_id)
    if trading_bot:
        trading_bot.status = status
        db.commit()
        db.refresh(trading_bot)
    return trading_bot


def get_trading_bots_by_status(db: Session, status: TradingbotStatus, skip: int = 0, limit: int = 100):
    return db.query(TradingBot).filter(TradingBot.status == status).offset(skip).limit(limit).all()
