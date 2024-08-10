# app/services/bot_service.py

from sqlalchemy.orm import Session
from app.db.models import Bot

def create_bot(db: Session, bot_data: dict) -> Bot:
    new_bot = Bot(**bot_data)
    db.add(new_bot)
    db.commit()
    db.refresh(new_bot)
    return new_bot

def get_bot(db: Session, bot_id: int) -> Bot:
    return db.query(Bot).filter(Bot.id == bot_id).first()

def get_bots(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Bot).offset(skip).limit(limit).all()

def update_bot(db: Session, bot_id: int, bot_data: dict) -> Bot:
    db_bot = get_bot(db, bot_id)
    if db_bot:
        for key, value in bot_data.items():
            setattr(db_bot, key, value)
        db.commit()
        db.refresh(db_bot)
    return db_bot

def delete_bot(db: Session, bot_id: int) -> bool:
    db_bot = get_bot(db, bot_id)
    if db_bot:
        db.delete(db_bot)
        db.commit()
        return True
    return False