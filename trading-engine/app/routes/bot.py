from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.bot import BotCreate, BotUpdate, BotInDB
from app.services import bot_service

router = APIRouter()


@router.post("/bots", response_model=BotInDB)
def create_bot(bot: BotCreate, db: Session = Depends(get_db)):
    return bot_service.create_bot(db, bot)


@router.get("/bots/{bot_id}", response_model=BotInDB)
def read_bot(bot_id: int, db: Session = Depends(get_db)):
    db_bot = bot_service.get_bot(db, bot_id)
    if db_bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return db_bot


@router.get("/bots", response_model=list[BotInDB])
def read_bots(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    bots = bot_service.get_bots(db, skip=skip, limit=limit)
    return bots


@router.put("/bots/{bot_id}", response_model=BotInDB)
def update_bot(bot_id: int, bot: BotUpdate, db: Session = Depends(get_db)):
    db_bot = bot_service.update_bot(db, bot_id, bot)
    if db_bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return db_bot


@router.delete("/bots/{bot_id}", response_model=bool)
def delete_bot(bot_id: int, db: Session = Depends(get_db)):
    result = bot_service.delete_bot(db, bot_id)
    if not result:
        raise HTTPException(status_code=404, detail="Bot not found")
    return result
