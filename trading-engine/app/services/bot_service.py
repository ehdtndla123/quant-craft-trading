from app.repositories import bot_repository
from app.schemas.bot import BotCreate, BotUpdate
from sqlalchemy.orm import Session


def create_bot(db: Session, bot_data: BotCreate):
    return bot_repository.create_bot(db, bot_data.dict())


def get_bot(db: Session, bot_id: int):
    return bot_repository.get_bot(db, bot_id)


def get_bots(db: Session, skip: int = 0, limit: int = 100):
    return bot_repository.get_bots(db, skip, limit)


def update_bot(db: Session, bot_id: int, bot_data: BotUpdate):
    return bot_repository.update_bot(db, bot_id, bot_data.dict())


def delete_bot(db: Session, bot_id: int):
    return bot_repository.delete_bot(db, bot_id)
