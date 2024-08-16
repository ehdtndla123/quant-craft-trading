from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.bot import BotCreate, BotUpdate, BotResponse
from app.services import bot_service

router = APIRouter()


@router.post("/bots", response_model=BotResponse)
def create_bot(bot_data: BotCreate, db: Session = Depends(get_db)):
    return bot_service.create_bot(db, bot_data)


@router.get("/bots/{bot_id}", response_model=BotResponse)
def read_bot(bot_id: int, db: Session = Depends(get_db)):
    bot = bot_service.get_bot(db, bot_id)
    if bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot


@router.get("/bots", response_model=list[BotResponse])
def read_bots(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return bot_service.get_bots(db, skip=skip, limit=limit)


@router.put("/bots/{bot_id}", response_model=BotResponse)
def update_bot(bot_id: int, bot_data: BotUpdate, db: Session = Depends(get_db)):
    updated_bot = bot_service.update_bot(db, bot_id, bot_data)
    if updated_bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    return updated_bot


@router.delete("/bots/{bot_id}", response_model=bool)
def delete_bot(bot_id: int, db: Session = Depends(get_db)):
    result = bot_service.delete_bot(db, bot_id)
    if not result:
        raise HTTPException(status_code=404, detail="Bot not found")
    return result
